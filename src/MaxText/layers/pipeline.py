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

from typing import Any, Callable

import numpy as np

from jax import numpy as jnp
from jax.sharding import Mesh
import jax
import jax.ad_checkpoint

from flax.core import meta
from flax import linen as nn
from flax import nnx

from MaxText.common_types import Config, MODEL_MODE_TRAIN, EP_AS_CONTEXT
from MaxText.layers import nnx_wrappers
from MaxText.layers.initializers import variable_to_logically_partitioned


class Pipeline(nnx.Module):
  """NNX Module that implements pipelining across stages.

  This module will loop over microbatches and execute the main body with a vmap for both the inputs and weights.
  This will produce a pipeline pattern if the stage dimension is sharded.

  Supports circular pipelines, and multiple layers per stage are used when a module that executes multiple layers
  is passed as the layers input.

  Attributes:
    config: Importantly contains num_pipeline_microbatches, num_pipeline_repeats.
    layers: A callable (NNX class or Linen class) that each stage can execute. It can either be a single layer such as a
      LlamaDecoderLayer instance or scanned/looped set of decoder layers to execute multiple layers per stage.
    mesh:  The device mesh of the system.
    remat_policy: Remat policy to use for the loop iterations
  """

  _linen_variables: nnx.Data[list[dict] | None]

  def __init__(
      self,
      layer: Callable[..., nnx.Module|nn.Module],
      config: Config,
      mesh: Mesh,
      rngs: nnx.Rngs,
      remat_policy: Any = None,
  ):
    """Initialize Pipeline with NNX or Linen decoder layers.

    Args:
      layers: Either an NNX class (type) or Linen class (type) to instantiate for each stage
      config: Model configuration
      mesh: Device mesh for sharding
      rngs: Optional NNX RNG state (passed by ToLinen wrapper)
      remat_policy: Remat policy for loop iterations
    """
    self.config = config
    self.mesh = mesh
    self.rngs = rngs
    self.remat_policy = remat_policy

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

    # TODO: Support NNX layers in addition to Linen layers
    self.layer = layer # Store the module instance
    self._linen_variables = None # Will be populated in _initialize_linen_parameters


  def need_circ_storage(self):
    return (
        self.config.num_pipeline_repeats > 1
        and self.config.num_pipeline_microbatches > self.num_stages * self.forwarding_delay
    )

  def iterations_to_complete_first_microbatch_one_repeat(self):
    """Returns iterations for microbatch 0 to complete one repeat."""
    return self.forwarding_delay * (self.num_stages - 1)

  def iterations_to_complete_first_microbatch(self):
    """Returns iterations for microbatch 0 to complete all repeats."""
    return (
        self.config.num_pipeline_microbatches * (self.config.num_pipeline_repeats - 1)
        + self.iterations_to_complete_first_microbatch_one_repeat()
    )

  def init_states(self, inputs):
    """Initialize pipeline loop state buffers.

    Assumes inputs are reshaped to [num_microbatches, micro_batch_size, sequence, embed].

    Returns:
      Dictionary containing:
        - shift: Buffer for rotating outputs [num_stages, micro_size, sequence, embed]
        - prev_outputs: Same shape as shift (only used with pipeline_delay_activation_forwarding)
        - state_io: Input/output buffer [num_stages, microbatches/stages, micro_size, sequence, embed]
        - circ_storage: Circular storage buffer (only when num_microbatches > num_stages)
        - circ_storage_mover: One-iteration delay buffer for circ_storage
        - loop_iteration: Iteration counter (starts at 0)
    """
    shift = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
    shift = self._with_logical_constraint(
        shift,
        ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
    )

    if self.config.pipeline_delay_activation_forwarding:
      prev_outputs = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
      prev_outputs = self._with_logical_constraint(
          prev_outputs,
          ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
      )
    else:
      prev_outputs = None

    state_io = jnp.reshape(inputs, (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:])
    state_io = self._with_logical_constraint(
        state_io,
        ("activation_stage", None, self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
    )

    if self.use_circ_storage:
      circ_storage = jnp.zeros((self.num_stages,) + inputs.shape, dtype=inputs.dtype)
      circ_storage_mover = shift
    else:
      circ_storage = None
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

  def _with_logical_constraint(self, tensor, logical_axis_names):
    """Applies logical sharding constraints to tensor."""
    return nn.with_logical_constraint(
        tensor,
        logical_axis_names,
        rules=self.config.logical_axis_rules,
        mesh=self.mesh,
    )

  def get_iteration_inputs(self, loop_iteration, state_io, circ_storage, shift):
    """Constructs input array for all stages for this iteration.

    Returns array of shape [stages, micro_size, sequence, embed] with rotated outputs
    from previous iteration, except stage 0 which gets new input from state_io or circ_storage.
    """
    state_io_batch_idx = loop_iteration % self.microbatches_per_stage
    state_io_slice = state_io[:, state_io_batch_idx]

    if self.use_circ_storage:
      circ_storage_batch_idx = loop_iteration % self.config.num_pipeline_microbatches
      circular_stage_in = circ_storage[:, circ_storage_batch_idx]
    else:
      circular_stage_in = shift

    first_stage_in = jnp.where(loop_iteration < self.config.num_pipeline_microbatches, state_io_slice, circular_stage_in)

    def select_state_or_input(first_stage_in, shift):
      return jnp.where(jax.lax.broadcasted_iota("int32", shift.shape, 0) == 0, first_stage_in, shift)

    stages_in = select_state_or_input(first_stage_in, shift)
    stages_in = self._with_logical_constraint(
        stages_in,
        ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
    )
    return stages_in

  def shard_dim_by_stages(self, x, dim: int):
    """Shards the specified dimension by stage."""
    dims_mapping = [jax.sharding.PartitionSpec.UNCONSTRAINED] * x.ndim
    dims_mapping[dim] = "stage"
    dims_mapping = tuple(dims_mapping)
    sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(*dims_mapping))
    return jax.lax.with_sharding_constraint(x, sharding)

  def get_microbatch_and_repeat_ids(self, loop_iteration):
    """Gets microbatch and repeat IDs for all stages at this iteration."""
    microbatches_processed = jnp.maximum(loop_iteration - self.forwarding_delay * jnp.arange(self.num_stages), 0)
    microbatch_ids = microbatches_processed % self.config.num_pipeline_microbatches
    repeat_ids = microbatches_processed // self.config.num_pipeline_microbatches
    return microbatch_ids, repeat_ids

  def vmap_parallel_gather(self, weights, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights):
    """Sharded parallel gather where each stage has its own weights and gets one slice.

    Args:
      weights: Per-stage data to gather from.
      repeat_ids: Integer tensor of shape [num_stages] with repeat indices per stage.
      repeat_dim_in_weights: Dimension where repeat_ids are applied (removed in output).
      stages_dim_in_weights: Dimension representing parallel stages.

    Returns:
      Per-stage gathered values with repeat_dim_in_weights removed.
    """
    def _gather_one(x, repeat_id):
      return jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, repeat_id, 1, repeat_dim_in_weights), repeat_dim_in_weights)

    gathered_weights_stage_dim = 0
    repeat_ids = self.shard_dim_by_stages(repeat_ids, 0)
    weights = self.shard_dim_by_stages(weights, stages_dim_in_weights)
    stage_weights = jax.vmap(_gather_one, in_axes=(stages_dim_in_weights, 0), out_axes=gathered_weights_stage_dim)(
        weights, repeat_ids
    )
    stage_weights = self.shard_dim_by_stages(stage_weights, gathered_weights_stage_dim)
    return stage_weights

  def vmap_gather(self, xs, ids, ids_dim):
    """Stage-wise sharded gather with shared input but different offsets per stage.

    Args:
      xs: Data shared by all stages.
      ids: Integer tensor of shape [num_stages] with offsets per stage.
      ids_dim: Dimension where ids are applied (output has [num_stages] size here).

    Returns:
      Per-stage gathered values with ids_dim size replaced with [num_stages].
    """
    def _gather_one(x, i):
      return jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, i, 1, ids_dim), ids_dim)

    ids = self.shard_dim_by_stages(ids, 0)
    outs = jax.vmap(_gather_one, in_axes=(None, 0), out_axes=ids_dim)(xs, ids)
    return self.shard_dim_by_stages(outs, 0)

  def get_new_loop_state(self, output, loop_state):
    """Updates all pipeline buffers after one iteration.

    Updates shift, state_io, circ_storage, circ_storage_mover, and prev_outputs
    to advance the pipeline by one step.
    """
    old_state_io = loop_state["state_io"]
    old_circ_storage = loop_state["circ_storage"]
    old_circ_storage_mover = loop_state["circ_storage_mover"]
    loop_iteration = loop_state["loop_iteration"]
    old_prev_outputs = loop_state["prev_outputs"]

    def _rotate_right(arr):
      last = jax.lax.slice_in_dim(arr, self.num_stages - 1, self.num_stages, axis=0)
      except_last = jax.lax.slice_in_dim(arr, 0, self.num_stages - 1, axis=0)
      return jnp.concatenate([last, except_last], axis=0)

    def _shift_right(arr):
      padding = [[1, 0]] + [[0, 0]] * (arr.ndim - 1)
      return jax.lax.slice(jnp.pad(arr, padding), [0] * arr.ndim, arr.shape)

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

    def _update_state_io(state_in, stream_slice, output):
      padding = [[0, 1]] + [[0, 0]] * (stream_slice.ndim - 1)
      stream_slice = jax.lax.slice_in_dim(jnp.pad(stream_slice, padding), 1, stream_slice.shape[0] + 1, axis=0)
      stream_slice = jnp.where(
          jax.lax.broadcasted_iota("int32", stream_slice.shape, 0) == self.num_stages - 1, output, stream_slice
      )
      stream_slice = jnp.expand_dims(stream_slice, 1)
      return jax.lax.dynamic_update_slice_in_dim(state_in, stream_slice, stream_buf_idx, axis=1)

    new_state = _update_state_io(old_state_io, stream_slice, output)

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
    """Permutes output to correct microbatch ordering after pipeline completion."""
    microbatch_0_idx = self.iterations_to_complete_first_microbatch() % self.microbatches_per_stage
    permutation = (
        np.arange(self.microbatches_per_stage) + microbatch_0_idx
    ) % self.microbatches_per_stage
    output = output[:, permutation]
    return output

  def _initialize_linen_parameters(self, sample_input, sample_seg_ids, sample_positions, deterministic, model_mode):
    """Initialize Linen module parameters for all stages."""
    if self._linen_variables is not None:
      return

    # Ensure inputs to init are 2D
    if sample_seg_ids is not None and sample_seg_ids.ndim == 1:
        sample_seg_ids = sample_seg_ids[None, :]
    if sample_positions is not None and sample_positions.ndim == 1:
        sample_positions = sample_positions[None, :]


    stage_params_list = []
    for _ in range(self.num_stages):
        linen_rngs = {
            'params': self.rngs.params(),
            'dropout': self.rngs.dropout()
        }
        base_params = self.layer.init(
            linen_rngs, sample_input, sample_seg_ids, sample_positions, deterministic, model_mode)
        stage_params_list.append(base_params)
    self._linen_variables = stage_params_list


  def _run_stages_linen(
      self,
      stages_inputs,
      stages_segment_ids,
      stages_positions,
      deterministic,
      model_mode,
      dropout_key
  ):
    """Run stages using Linen module with manual vmap."""
    stage_params_list = self._linen_variables if self._linen_variables is not None else []
    stacked_params = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0),
        *stage_params_list
    )

    def apply_stage(stage_idx, stage_params, stage_input, stage_seg_ids, stage_pos):
      if stage_pos is not None and stage_pos.ndim == 1: stage_pos = stage_pos[None, :]
      if stage_seg_ids is not None and stage_seg_ids.ndim == 1: stage_seg_ids = stage_seg_ids[None, :]

      # Fold in stage index to the iteration key for unique RNG per stage
      stage_dropout_key = jax.random.fold_in(dropout_key, stage_idx)

      output = self.layer.apply(
          stage_params,
          stage_input,
          stage_seg_ids,
          stage_pos,
          deterministic=deterministic,
          model_mode=model_mode,
          rngs={"dropout": stage_dropout_key},
      )
      return output[0] if isinstance(output, tuple) else output

    stage_indices = jnp.arange(self.num_stages)

    if stages_segment_ids is None:
      vmap_fn = lambda idx, p, i, pos: apply_stage(idx, p, i, None, pos)
      stages_outputs = jax.vmap(vmap_fn, in_axes=(0, 0, 0, 0), out_axes=0)(
          stage_indices, stacked_params, stages_inputs, stages_positions
      )
    else:
      vmap_fn = lambda idx, p, i, s, pos: apply_stage(idx, p, i, s, pos)
      stages_outputs = jax.vmap(vmap_fn, in_axes=(0, 0, 0, 0, 0), out_axes=0)(
        stage_indices, stacked_params, stages_inputs, stages_segment_ids, stages_positions
      )

    return stages_outputs


  def _run_stages_vmapped(
      self,
      stages_inputs,
      stages_segment_ids,
      stages_positions,
      deterministic,
      model_mode,
  ):
    """Run all stages in parallel using JAX vmap over NNX instances."""
    stage_0 = getattr(self, 'stage_0')
    graphdef, state_0 = nnx.split(stage_0)

    states = [state_0]
    for s in range(1, self.num_stages):
      instance = getattr(self, f'stage_{s}')
      _, state_s = nnx.split(instance)
      states.append(state_s)

    stacked_state = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0),
        *states
    )

    def call_stage(state, stage_input, stage_seg_ids, stage_pos):
      if stage_pos is not None and stage_pos.ndim == 1:
        stage_pos = stage_pos[None, :]
      if stage_seg_ids is not None and stage_seg_ids.ndim == 1:
        stage_seg_ids = stage_seg_ids[None, :]
      module = nnx.merge(graphdef, state)
      output = module(stage_input, stage_seg_ids, stage_pos, deterministic=deterministic, model_mode=model_mode)
      return output[0] if isinstance(output, tuple) else output

    if stages_segment_ids is None:
      def call_stage_no_seg(state, stage_input, stage_pos):
        if stage_pos is not None and stage_pos.ndim == 1:
          stage_pos = stage_pos[None, :]
        module = nnx.merge(graphdef, state)
        output = module(stage_input, None, stage_pos, deterministic=deterministic, model_mode=model_mode)
        return output[0] if isinstance(output, tuple) else output
      vmap_fn = call_stage_no_seg
      in_axes = (0, 0, 0)
      stages_outputs = jax.vmap(vmap_fn, in_axes=in_axes, out_axes=0)(
         stacked_state, stages_inputs, stages_positions
      )
    else:
      vmap_fn = call_stage
      in_axes = (0, 0, 0, 0)
      stages_outputs = jax.vmap(vmap_fn, in_axes=in_axes, out_axes=0)(
         stacked_state, stages_inputs, stages_segment_ids, stages_positions
      )
    return stages_outputs


  def run_one_iteration(
      self,
      loop_state,
      positions,
      segment_ids,
      deterministic,
      model_mode,
  ):
    """Run one loop iteration: get inputs, execute stages, update state."""
    state_io = loop_state["state_io"]
    shift = loop_state["shift"]
    circ_storage = loop_state["circ_storage"]
    loop_iteration = loop_state["loop_iteration"]

    # Get a base key for this iteration
    iter_dropout_key_base = self.rngs.dropout()
    # Fold in the loop iteration number to make the key unique per iteration
    iter_dropout_key = jax.random.fold_in(iter_dropout_key_base, loop_iteration)

 
    microbatch_ids, _ = self.get_microbatch_and_repeat_ids(loop_iteration)
    stages_inputs = self.get_iteration_inputs(loop_iteration, state_io, circ_storage, shift)
    stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, "iteration_input")
    stages_positions = self.vmap_gather(positions, microbatch_ids, 0) if positions is not None else None
    stages_segment_ids = self.vmap_gather(segment_ids, microbatch_ids, 0) if segment_ids is not None else None

    stages_output = self._run_stages_linen(stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode, iter_dropout_key)

    new_loop_state = self.get_new_loop_state(stages_output, loop_state)
    return new_loop_state


  def get_pipeline_remat_policy(self):
    """Returns the remat policy for pipeline iterations."""
    if self.config.remat_policy == "custom":
      return self.remat_policy

    save_input_policy = jax.checkpoint_policies.save_only_these_names("iteration_input", "decoder_layer_input")
    if self.remat_policy is not None:
      remat_policy = jax.checkpoint_policies.save_from_both_policies(self.remat_policy, save_input_policy)
    else:
      remat_policy = save_input_policy
    return remat_policy

  def __call__(
      self,
      inputs: jnp.ndarray,
      segment_ids: jnp.ndarray,
      positions: jnp.ndarray,
      deterministic: bool,
      model_mode=MODEL_MODE_TRAIN,
      partition_spec=None,
  ) -> jnp.ndarray:
    """Maps decoder layer inputs to outputs using pipeline parallelism.

    Reshapes inputs into microbatches, runs pipeline iterations with bubble
    handling, and returns outputs reshaped to original batch size.
    """
    inputs = inputs.reshape((self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length, self.config.emb_dim))
    if positions is not None:
      positions = positions.reshape((self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length))
    if segment_ids is not None:
      segment_ids = segment_ids.reshape((self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length))

    if self._linen_variables is None:
      example_input = inputs[0]
      example_seg_ids = segment_ids[0] if segment_ids is not None else None
      example_pos = positions[0] if positions is not None else None
      self._initialize_linen_parameters(example_input, example_seg_ids, example_pos, deterministic, model_mode)

    loop_state = self.init_states(inputs)

    bubble_iterations = self.forwarding_delay * (self.num_stages - 1)
    real_iterations = self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats
    total_iterations = real_iterations + bubble_iterations

    if self.config.scan_pipeline_iterations:
      def run_iteration_scannable(pipeline_instance, loop_state, xs):
        new_loop_state = pipeline_instance.run_one_iteration(loop_state, positions, segment_ids, deterministic, model_mode)
        return new_loop_state

      remat_fn = run_iteration_scannable
      if self.config.set_remat_policy_on_pipeline_iterations:
        remat_fn = nnx.remat(
            run_iteration_scannable,
            prevent_cse=False,
            policy=self.get_pipeline_remat_policy(),
        )

      ScannedFn = nnx.scan(
          remat_fn,
          in_axes=(None, nnx.Carry, None),  # Corresponds to (self, loop_state, xs)
          out_axes=nnx.Carry,               # Corresponds to new_loop_state
          length=total_iterations,
      )
      loop_state = ScannedFn(self, loop_state, None)
    else:
      for _ in range(total_iterations):
        loop_state = self.run_one_iteration(loop_state, positions, segment_ids, deterministic, model_mode)

    final_output = self.permute_output_micro_per_stage_dim(loop_state["state_io"])
    final_output = jnp.reshape(
        final_output, (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim)
    )
    return final_output

class PipelineToLinen(nnx_wrappers.ToLinen):
  """Wrap NNX Pipeline as a Linen module.

  This allows the NNX Pipeline to be used within the Linen Decoder module.
  """
  pass


def create_pipeline(
    config: Config,
    layer: Callable | type,
    mesh: Mesh,
    remat_policy: Any = None,
) -> nnx_wrappers.ToLinen:
  """Factory function to create a Pipeline wrapped as a Linen module.

  Args:
    config: Model configuration
    layer: NNX or Linen decoder layer class to use for pipeline stages
    mesh: Device mesh for sharding
    remat_policy: Remat policy for loop iterations
    use_nnx: Whether to use NNX pipeline (True) or Linen (False)

  Returns:
    PipelineToLinen wrapper around the NNX Pipeline
  """
  return nnx_wrappers.to_linen(
      Pipeline,
      config=config,
      mesh=mesh,
      layer=layer,
      remat_policy=remat_policy,
      name="pipeline_module",
      abstract_init=False,
      metadata_fn=variable_to_logically_partitioned,
  )
