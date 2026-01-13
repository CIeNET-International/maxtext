# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Pipeline layer wrapping a decoder layer(s). Supports circular pipelining """

import functools
from typing import Any

import numpy as np

from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import jax
import jax.ad_checkpoint

from flax import nnx
from flax import linen as nn
from MaxText.layers import nnx_wrappers
from MaxText import maxtext_utils


from MaxText.common_types import Config, MODEL_MODE_TRAIN, EP_AS_CONTEXT, ShardMode
from MaxText.sharding import (
    maybe_shard_with_logical,
    maybe_shard_with_name,
    create_sharding,
    logical_to_mesh_axes,
    logical_to_mesh,
)


class Pipeline(nnx.Module):
  """Module that implements pipelining across stages.

  This module will loop over microbatches and execute the main body with a vmap for both the inputs and weights.
  This will produce a pipeline pattern if the stage dimension is sharded.

  Supports circular pipelines, and multiple layers per stage are used when a module that executes multiple layers
  is passed as the layers input.

  Attributes:
    config: Importantly contains num_pipeline_microbatches, num_pipeline_repeats.
    layers: A module instance that each stage can execute. It can either be a single layer such as a
      LlamaDecoderLayer instance or scanned/looped set of decoder layers to execute multiple layers per stage.
    mesh:  The device mesh of the system.
    remat_policy: Remat policy to use for the loop iterations
  """

  def __init__(
      self,
      layers: Any, # Expects factory function
      config: Config,
      mesh: Mesh,
      rngs: nnx.Rngs,
      remat_policy: Any = None,
  ):
    self.config = config
    self.mesh = mesh
    self.remat_policy = remat_policy
    self.rngs = rngs

    self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
    self.forwarding_delay = 2 if self.config.pipeline_delay_activation_forwarding else 1
    self.pipeline_microbatch_size = self.config.micro_batch_size_to_train_on // self.config.num_pipeline_microbatches
    microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages
    self.microbatches_per_stage = microbatches_per_stage
    self.use_circ_storage = self.need_circ_storage()

    if self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      self.batch_axis_name = "activation_batch_no_exp"
      self.seq_len_axis_name = "activation_length"
    else:
      self.batch_axis_name = "activation_batch"
      self.seq_len_axis_name = "activation_length_no_exp"

    # TODO(b/470167805): replace self.spmd_axis_name with "stage" when JAX >= 0.8.2.
    self.spmd_axis_name = "stage" if self.config.shard_mode == ShardMode.AUTO else None

    self.stages_in_logical = ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed")
    self.stages_in_spec = logical_to_mesh_axes(self.stages_in_logical, self.mesh, rules=self.config.logical_axis_rules)
    self.stages_in_sharding = (
        NamedSharding(self.mesh, self.stages_in_spec) if self.config.shard_mode == ShardMode.EXPLICIT else None
    )

    self.state_io_logical = ("activation_stage", None, self.batch_axis_name, self.seq_len_axis_name, "activation_embed")
    self.state_io_spec = logical_to_mesh_axes(self.state_io_logical, self.mesh, rules=self.config.logical_axis_rules)
    self.state_io_sharding = (
        NamedSharding(self.mesh, self.state_io_spec) if self.config.shard_mode == ShardMode.EXPLICIT else None
    )
    self.input_sharding = (
        create_sharding(
            self.mesh,
            (None, self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
            rules=self.config.logical_axis_rules,
        )
        if self.config.shard_mode == ShardMode.EXPLICIT
        else None
    )
    self.output_sharding = (
        create_sharding(
            self.mesh,
            (self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
            rules=self.config.logical_axis_rules,
        )
        if self.config.shard_mode == ShardMode.EXPLICIT
        else None
    )

    # Instantiate Stages
    stage_param_keys = jax.random.split(rngs.params(), self.num_stages)
    stage_dropout_keys = jax.random.split(rngs.dropout(), self.num_stages)

    def create_stage(p_key, d_key):
        stage_rngs = nnx.Rngs(params=p_key, dropout=d_key)
        return layers(stage_rngs)

    self.layers = nnx.vmap(
        create_stage,
        in_axes=(0, 0),
        transform_metadata={nnx.PARTITION_NAME: "stage"}
    )(stage_param_keys, stage_dropout_keys)

  def need_circ_storage(self):
    return (
        self.config.num_pipeline_repeats > 1
        and self.config.num_pipeline_microbatches > self.num_stages * self.forwarding_delay
    )

  def iterations_to_complete_first_microbatch_one_repeat(self):
    return self.forwarding_delay * (self.num_stages - 1)

  def iterations_to_complete_first_microbatch(self):
    return (
        self.config.num_pipeline_microbatches * (self.config.num_pipeline_repeats - 1)
        + self.iterations_to_complete_first_microbatch_one_repeat()
    )

  def _maybe_shard_with_logical(self, inputs, logical_axes):
    return maybe_shard_with_logical(
        inputs,
        logical_axes,
        shard_mode=self.config.shard_mode,
        mesh=self.mesh,
        rules=self.config.logical_axis_rules,
    )

  def _maybe_shard_with_name(self, inputs, sharding_name):
    return maybe_shard_with_name(inputs, sharding_name, shard_mode=self.config.shard_mode)

  def init_states(self, inputs):
    # Shift
    shift = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
    shift = self._maybe_shard_with_logical(shift, self.stages_in_logical)

    # Prev outputs
    if self.config.pipeline_delay_activation_forwarding:
      prev_outputs = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
      prev_outputs = self._maybe_shard_with_logical(prev_outputs, self.stages_in_logical)
    else:
      prev_outputs = None

    # state_io
    state_io = jnp.reshape(
        inputs, (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:], out_sharding=self.state_io_sharding
    )
    state_io = self._maybe_shard_with_logical(state_io, self.state_io_logical)

    # circ_storage
    if self.use_circ_storage:
      circ_storage = jnp.zeros((self.num_stages,) + inputs.shape, dtype=inputs.dtype, out_sharding=self.state_io_sharding)
    else:
      circ_storage = None

    # circ_storage_mover
    if self.use_circ_storage:
      circ_storage_mover = shift
    else:
      circ_storage_mover = None

    init_loop_state = {
        "state_io": state_io,
        "shift": shift,
        "circ_storage": circ_storage,
        "circ_storage_mover": circ_storage_mover,
        "loop_iteration": 0,
        "prev_outputs": prev_outputs,
    }
    return init_loop_state

  def get_iteration_inputs(self, loop_iteration, state_io, circ_storage, shift):
    state_io_batch_idx = loop_iteration % self.microbatches_per_stage
    state_io_slice = state_io[:, state_io_batch_idx]
    shift = self._maybe_shard_with_logical(shift, self.stages_in_logical)

    if self.use_circ_storage:
      circ_storage_batch_idx = loop_iteration % self.config.num_pipeline_microbatches
      circular_stage_in = circ_storage[:, circ_storage_batch_idx]
    else:
      circular_stage_in = shift

    first_stage_in = jnp.where(loop_iteration < self.config.num_pipeline_microbatches, state_io_slice, circular_stage_in)
    first_stage_in = self._maybe_shard_with_logical(first_stage_in, self.stages_in_logical)

    def select_state_or_input(first_stage_in, shift):
      return jnp.where(
          jax.lax.broadcasted_iota("int32", shift.shape, 0, out_sharding=self.stages_in_sharding) == 0,
          first_stage_in,
          shift,
      )

    stages_in = select_state_or_input(first_stage_in, shift)
    stages_in = self._maybe_shard_with_logical(stages_in, self.stages_in_logical)
    return stages_in

  def shard_dim_by_stages(
      self, x, dim: int, physical_partition_spec: PartitionSpec | None, is_stage_weight: bool = False
  ):
    placeholder = None if self.config.shard_mode == ShardMode.EXPLICIT else PartitionSpec.UNCONSTRAINED
    if physical_partition_spec is None:
      dims_mapping = [placeholder] * x.ndim
    else:
      physical_partition_spec = self._remove_fsdp_from_physical_partition_spec(physical_partition_spec)
      dims_mapping = list(physical_partition_spec)
      if not is_stage_weight:
        dims_mapping = [placeholder] * (dim + 1) + dims_mapping[dim:]
    dims_mapping[dim] = "stage"
    dims_mapping = tuple(dims_mapping)
    if physical_partition_spec and is_stage_weight and self.config.shard_mode == ShardMode.EXPLICIT:
      batch_mesh_axis = ["data", "fsdp"]
      reduced_mark = [mesh_axis for mesh_axis in batch_mesh_axis if self.mesh.shape[mesh_axis] > 1]
      pspec = PartitionSpec(*dims_mapping, reduced=set(reduced_mark))
    else:
      pspec = PartitionSpec(*dims_mapping)
    sharding = jax.sharding.NamedSharding(self.mesh, pspec)
    return self._maybe_shard_with_name(x, sharding)

  def get_microbatch_and_repeat_ids(self, loop_iteration):
    microbatches_processed = jnp.maximum(loop_iteration - self.forwarding_delay * jnp.arange(self.num_stages), 0)
    microbatch_ids = microbatches_processed % self.config.num_pipeline_microbatches
    repeat_ids = microbatches_processed // self.config.num_pipeline_microbatches
    return microbatch_ids, repeat_ids

  def vmap_parallel_gather(
      self, weights, physical_partition_spec, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights
  ):
    def _gather_one(x, repeat_id):
      return jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, repeat_id, 1, repeat_dim_in_weights), repeat_dim_in_weights)

    gathered_weights_stage_dim = 0
    repeat_ids = self.shard_dim_by_stages(repeat_ids, 0, physical_partition_spec=None)
    weights = self.shard_dim_by_stages(
        weights, stages_dim_in_weights, physical_partition_spec=physical_partition_spec, is_stage_weight=False
    )
    stage_weights = jax.vmap(_gather_one, in_axes=(stages_dim_in_weights, 0), out_axes=gathered_weights_stage_dim)(
        weights, repeat_ids
    )
    stage_weights = self.shard_dim_by_stages(
        stage_weights, gathered_weights_stage_dim, physical_partition_spec=physical_partition_spec, is_stage_weight=True
    )
    return stage_weights

  def vmap_gather(self, xs, ids, ids_dim):
    def _gather_one(x, i):
      idx = tuple(i if d == ids_dim else slice(None) for d in range(x.ndim))
      replicated_sharding = NamedSharding(self.mesh, PartitionSpec())
      return x.at[idx].get(out_sharding=replicated_sharding)

    ids = self.shard_dim_by_stages(ids, 0, physical_partition_spec=None)
    outs = jax.vmap(_gather_one, in_axes=(None, 0), out_axes=ids_dim)(xs, ids)
    return self.shard_dim_by_stages(outs, 0, physical_partition_spec=None)

  def get_new_loop_state(self, output, loop_state):
    old_state_io = loop_state["state_io"]
    old_circ_storage = loop_state["circ_storage"]
    old_circ_storage_mover = loop_state["circ_storage_mover"]
    loop_iteration = loop_state["loop_iteration"]
    old_prev_outputs = loop_state["prev_outputs"]

    @jax.shard_map(
        mesh=self.mesh,
        in_specs=self.stages_in_spec,
        out_specs=self.stages_in_spec,
        check_vma=True,
    )
    def _rotate_right(arr):
      stage_size = jax.lax.axis_size("stage")
      perm = [(i, (i + 1) % stage_size) for i in range(stage_size)]
      arr = jax.lax.ppermute(arr, axis_name="stage", perm=perm)
      return arr

    @jax.shard_map(
        mesh=self.mesh,
        in_specs=self.stages_in_spec,
        out_specs=self.stages_in_spec,
        check_vma=True,
    )
    def _shift_right(arr):
      stage_idx = jax.lax.axis_index("stage")
      stage_size = jax.lax.axis_size("stage")
      perm = [(i, (i + 1) % stage_size) for i in range(stage_size)]
      arr = jax.lax.ppermute(arr, axis_name="stage", perm=perm)
      return jnp.where(stage_idx == 0, jnp.zeros_like(arr), arr)

    def _update_shift(output_in):
      if self.config.num_pipeline_repeats == 1 or self.use_circ_storage:
        return _shift_right(output_in)
      else:
        return _rotate_right(output_in)

    if self.config.pipeline_delay_activation_forwarding:
      new_shift = _update_shift(old_prev_outputs)
      new_prev_outputs = output
    else:
      new_shift = _update_shift(output)
      new_prev_outputs = None

    if self.use_circ_storage:
      def _rotate_right_and_update(circ_storage_mover_in, circ_storage_in):
        rotated = _rotate_right(circ_storage_mover_in)
        rotated = jnp.expand_dims(rotated, 1)
        offset = (
            loop_iteration - self.iterations_to_complete_first_microbatch_one_repeat() - 1
        ) % self.config.num_pipeline_microbatches
        return jax.lax.dynamic_update_slice_in_dim(circ_storage_in, rotated, offset, axis=1)

      new_circ_storage = _rotate_right_and_update(old_circ_storage_mover, old_circ_storage)
      new_circ_storage_mover = output
    else:
      new_circ_storage = None
      new_circ_storage_mover = None

    stream_buf_idx = loop_iteration % self.microbatches_per_stage
    stream_slice = old_state_io[:, stream_buf_idx]

    def _rotate_left(arr, stage_size):
      perm = [(i, (i - 1) % stage_size) for i in range(stage_size)]
      arr = jax.lax.ppermute(arr, axis_name="stage", perm=perm)
      return arr

    def _shift_left(arr, stage_size, output):
      stage_idx = jax.lax.axis_index("stage")
      arr = _rotate_left(arr, stage_size)
      return jnp.where(stage_idx == stage_size - 1, output, arr)

    @jax.shard_map(
        mesh=self.mesh,
        in_specs=(self.state_io_spec, self.stages_in_spec, self.stages_in_spec, PartitionSpec()),
        out_specs=self.state_io_spec,
    )
    def _update_state_io(state_in, stream_slice, output, stream_buf_idx):
      stage_size = jax.lax.axis_size("stage")
      stream_slice = _shift_left(stream_slice, stage_size, output)
      stream_slice = jnp.expand_dims(stream_slice, 1)
      return jax.lax.dynamic_update_slice_in_dim(state_in, stream_slice, stream_buf_idx, axis=1)

    new_state = _update_state_io(old_state_io, stream_slice, output, stream_buf_idx)

    new_loop_state = {
        "state_io": new_state,
        "shift": new_shift,
        "circ_storage": new_circ_storage,
        "circ_storage_mover": new_circ_storage_mover,
        "loop_iteration": loop_iteration + 1,
        "prev_outputs": new_prev_outputs,
    }
    return new_loop_state

  def permute_output_micro_per_stage_dim(self, output):
    microbatch_0_idx = self.iterations_to_complete_first_microbatch() % self.microbatches_per_stage
    permutation = (
        np.arange(self.microbatches_per_stage) + microbatch_0_idx
    ) % self.microbatches_per_stage
    output = output[:, permutation]
    return output

  def get_current_stage_weights(self, pipeline_weights, loop_iteration, physical_partition_spec=None):
    """
    Gets the current weights used for one iteration.
    FIX: Logic updated to handle shared weights across repeats.
    """
    if self.config.num_pipeline_repeats > 1:
        # Check if weights actually have repeat dimension (ndim check or similar)
        # For now, if we assume weights are [stages, ...], they don't have repeat.
        # We bypass gather if weights are shared.
        
        # NOTE: If we implement distinct weights per repeat later, we need to 
        # update __init__ to create [repeats, stages, ...] and restore the gather logic.
        # For current __init__ [stages, ...], we return as is.
        return pipeline_weights
    else:
      return pipeline_weights

  def get_weight_sharding(self):
    """Returns the PartitionSpec tree for the model weights, prepending 'stage' axis."""
    flat_specs = {}
    
    # Iterate over the graph to access the actual Variable objects (which hold metadata)
    # rather than just the values.
    for path, var in nnx.iter_graph(self):
        if isinstance(var, nnx.Param):
            # 1. Get the inner sharding spec defined by the layer (e.g. {'embed', 'vocab'})
            # If no sharding is defined, it defaults to None (fully replicated inner).
            inner_spec = getattr(var, 'sharding', None)
            
            # 2. Normalize inner_spec to a tuple/PartitionSpec
            if inner_spec is None:
                inner_spec = PartitionSpec() # empty tuple
            
            # 3. Prepend the "stage" axis. 
            # We know 'self.layers' is vmapped over the 'stage' axis.
            # All parameters inside 'self.layers' must have this leading axis sharded.
            if path[0] == 'layers':
                 new_spec = PartitionSpec("stage", *inner_spec)
                 flat_specs[path] = new_spec
            else:
                 # Handle non-layer parameters if any (unlikely in this Pipeline design)
                 flat_specs[path] = inner_spec

    # 4. Reconstruct the nested structure matching the parameters
    nested_specs = nnx.State(flat_specs).to_pure_dict()
    
    return {"params": nested_specs}

  def get_functional_stage_fn(self):
    """Returns pure (weights, inputs...) -> (output, new_state)"""
    graph_def, _ = nnx.split(self.layers)

    def stage_fn(weights, inputs, segment_ids, positions, deterministic, model_mode):
      model = nnx.merge(graph_def, weights)
      out = model(inputs, segment_ids, positions, deterministic, model_mode)
      # Capture updated state (metrics, etc.)
      _, new_state = nnx.split(model)
      return out, new_state

    return stage_fn

  def run_one_iteration(
      self,
      loop_state,
      pipeline_weights,
      positions,
      segment_ids,
      deterministic,
      model_mode,
      logical_partition_spec=None,
  ):
    state_io = loop_state["state_io"]
    shift = loop_state["shift"]
    circ_storage = loop_state["circ_storage"]
    loop_iteration = loop_state["loop_iteration"]

    microbatch_ids, _ = self.get_microbatch_and_repeat_ids(loop_iteration)
    physical_partition_spec = logical_to_mesh(logical_partition_spec, self.mesh, rules=self.config.logical_axis_rules)

    stages_inputs = self.get_iteration_inputs(loop_iteration, state_io, circ_storage, shift)
    
    stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, "iteration_input")
    
    stages_positions = self.vmap_gather(positions, microbatch_ids, 0) if positions is not None else None
    stages_segment_ids = self.vmap_gather(segment_ids, microbatch_ids, 0) if segment_ids is not None else None

    stage_weights = self.get_current_stage_weights(
        pipeline_weights, loop_iteration, physical_partition_spec=physical_partition_spec
    )

    stage_fn_pure = self.get_functional_stage_fn()

    # Vmap over stages (axis 0)
    # output: (stages_out, updated_weights)
    vmapped_stage_fn = jax.vmap(
        stage_fn_pure, 
        in_axes=(0, 0, 0, 0, None, None), 
        out_axes=(0, 0),
        spmd_axis_name=self.spmd_axis_name
    )

    stages_output, updated_stage_weights = vmapped_stage_fn(
        stage_weights,
        stages_inputs,
        stages_segment_ids,
        stages_positions,
        deterministic,
        model_mode,
    )

    new_loop_state = self.get_new_loop_state(stages_output, loop_state)
    
    return new_loop_state, updated_stage_weights

  def get_pipeline_remat_policy(self):
    if self.config.remat_policy == "custom":
      return self.remat_policy

    save_input_policy = jax.checkpoint_policies.save_only_these_names("iteration_input", "decoder_layer_input")
    if self.remat_policy is not None:
      remat_policy = jax.checkpoint_policies.save_from_both_policies(self.remat_policy, save_input_policy)
    else:
      remat_policy = save_input_policy
    return remat_policy

  @staticmethod
  def get_logical_spec_repeats_removed(full_logical):
    if full_logical is None: return None
    def _remove_from_spec(spec):
      return jax.sharding.PartitionSpec(*[dim for dim in spec if dim != "circular_repeats"])
    return jax.tree.map(_remove_from_spec, full_logical)

  @staticmethod
  def _remove_fsdp_from_physical_partition_spec(pps):
    if isinstance(pps, PartitionSpec):
      new_spec = []
      for axis in pps:
        if axis is None: new_spec.append(None)
        elif isinstance(axis, str):
          if axis not in ("fsdp", "fsdp_transpose"): new_spec.append(axis)
          else: new_spec.append(None)
        elif isinstance(axis, (list, tuple)):
          new_axis = [a for a in axis if a not in ("fsdp", "fsdp_transpose")]
          new_spec.append(tuple(new_axis))
        else: raise ValueError(f"Unsupported_axis_type: {type(axis)}")
      return PartitionSpec(*new_spec)
    return pps

  def all_gather_over_fsdp(self, variables, logical_partition_spec):
    physical_partition_spec = logical_to_mesh(
        logical_partition_spec, mesh=self.mesh, rules=self.config.logical_axis_rules
    )
    physical_partition_spec_no_fsdp = jax.tree.map(
        self._remove_fsdp_from_physical_partition_spec, physical_partition_spec
    )
    return jax.tree.map(
        lambda w, p: self._maybe_shard_with_name(w, NamedSharding(self.mesh, p)),
        variables,
        physical_partition_spec_no_fsdp,
    )
  
  def __call__(
      self,
      inputs: jnp.ndarray,
      segment_ids: jnp.ndarray,
      positions: jnp.ndarray,
      deterministic: bool,
      model_mode=MODEL_MODE_TRAIN,
      logical_partition_spec=None,
  ) -> jnp.ndarray:
    """The main method that maps the series of decoder layer inputs to final layer outputs."""
    with self.mesh:
      # 1. Reshape inputs to [microbatches, microbatch_size, seq_len, embed_dim]
      inputs = inputs.reshape(
          (
              self.config.num_pipeline_microbatches,
              self.pipeline_microbatch_size,
              self.config.max_target_length,
              self.config.emb_dim,
          ),
          out_sharding=self.input_sharding,
      )

      # 2. Handle Positions and Segment IDs (All Gather if needed)
      ag_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(None, None))
      
      if positions is not None:
        positions = self._maybe_shard_with_name(positions, ag_sharding)
        positions = positions.reshape(
            (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
        )

      if segment_ids is not None:
        segment_ids = self._maybe_shard_with_name(segment_ids, ag_sharding)
        segment_ids = segment_ids.reshape(
            (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
        )

      # 3. Initialize Pipeline State Buffers
      loop_state = self.init_states(inputs)

      bubble_iterations = self.forwarding_delay * (self.num_stages - 1)
      real_iterations = self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats
      total_iterations = real_iterations + bubble_iterations

      # 4. Prepare Weights (Capture once)
      # We treat weights as constant for the duration of the pipeline loop (Forward Pass).
      # This matches Linen's 'variable_broadcast' semantics and prevents OOM.
      variables = nnx.state(self.layers)

      if self.config.pipeline_fsdp_ag_once:
        all_pipeline_weights = self.all_gather_over_fsdp(variables, logical_partition_spec)
      else:
        all_pipeline_weights = variables

      logical_partition_spec = self.get_logical_spec_repeats_removed(logical_partition_spec)

      # 5. Define the Step Function
      def step_fn(loop_state, _):
          # We close over 'all_pipeline_weights', treating them as constants.
          # This tells XLA not to allocate new buffers for weights at every step.
          new_loop_state, _ = self.run_one_iteration(
              loop_state,
              all_pipeline_weights,
              positions,
              segment_ids,
              deterministic,
              model_mode,
              logical_partition_spec=logical_partition_spec,
          )
          # We discard the second return value (updated_stage_weights/metrics) here 
          # to ensure the scan loop stays efficient and memory-bound.
          return new_loop_state, None

      # 6. Apply Rematerialization (Gradient Checkpointing)
      if self.config.set_remat_policy_on_pipeline_iterations:
        prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)
        step_fn = jax.checkpoint(step_fn, policy=self.get_pipeline_remat_policy(),prevent_cse=prevent_cse)

      # 7. Execute the Loop
      if self.config.scan_pipeline_iterations:
        # Use jax.lax.scan for compilation efficiency
        scan_xs = jnp.arange(total_iterations)
        # Pass ONLY loop_state as carry. Weights are implicitly broadcasted via closure.
        loop_state, _ = jax.lax.scan(step_fn, loop_state, scan_xs)
      else:
        # Standard loop (for debugging or specific configs)
        for _ in range(total_iterations):
          loop_state, _ = step_fn(loop_state, None)

      # 8. Post-process Outputs
      # The final output is located in the state_io buffer, potentially permuted.
      final_output = self.permute_output_micro_per_stage_dim(loop_state["state_io"])

      final_output = jnp.reshape(
          final_output,
          (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim),
          out_sharding=self.output_sharding,
      )

      return final_output