import functools
from typing import Any, Dict, Tuple, Optional, Callable
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import jax.ad_checkpoint

from flax import nnx

from MaxText.common_types import Config, MODEL_MODE_TRAIN, EP_AS_CONTEXT, ShardMode
from MaxText.sharding import (
    maybe_shard_with_logical,
    maybe_shard_with_name,
    create_sharding,
    logical_to_mesh_axes,
    logical_to_mesh,
)

# --- DEBUG HELPER ---
def log_trace(tag, msg):
    print(f"[PIPELINE-TRACE] [{tag}] {msg}")

def log_shape(tag, name, obj):
    if hasattr(obj, 'shape'):
        print(f"[PIPELINE-DATA] [{tag}] {name}: Shape={obj.shape}, Dtype={obj.dtype}")
    elif isinstance(obj, (list, tuple)):
        print(f"[PIPELINE-DATA] [{tag}] {name}: Type={type(obj)}, Len={len(obj)}")
    elif obj is None:
        print(f"[PIPELINE-DATA] [{tag}] {name}: None")
    else:
        print(f"[PIPELINE-DATA] [{tag}] {name}: Type={type(obj)}")
# --------------------

def _remove_fsdp_from_physical_partition_spec(pps):
    if isinstance(pps, P):
        new_spec = []
        for axis in pps:
            if axis is None:
                new_spec.append(None)
            elif isinstance(axis, str):
                if axis not in ("fsdp", "fsdp_transpose"):
                    new_spec.append(axis)
                else:
                    new_spec.append(None)
            elif isinstance(axis, (list, tuple)):
                new_axis = [a for a in axis if a not in ("fsdp", "fsdp_transpose")]
                new_spec.append(tuple(new_axis))
            else:
                raise ValueError(f"Unsupported_axis_type: {type(axis)}")
        return P(*new_spec)
    return pps


class Pipeline(nnx.Module):
  """NNX implementation of Pipeline parallelism."""

  def __init__(
      self, 
      config: Config, 
      mesh: Mesh, 
      layers: nnx.Module, 
      remat_policy: Any = None,
      *,
      rngs: nnx.Rngs = None
  ):
    log_trace("__init__", "Start")
    self.config = config
    self.mesh = mesh
    self.remat_policy = remat_policy
    
    self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
    log_trace("__init__", f"Num Stages: {self.num_stages}")
    
    # -------------------------------------------------------------------------
    # 1. Lift the State (Arrays)
    # -------------------------------------------------------------------------
    log_trace("__init__", "Splitting layers into graph/state...")
    graphdef, state = nnx.split(layers)
    
    def lift_state(path, leaf):
        if not hasattr(leaf, 'shape'): 
            return leaf
        
        is_scalar = (len(leaf.shape) == 0)
        is_already_staged = False
        if not is_scalar:
            is_already_staged = (leaf.shape[0] == self.num_stages)
        
        if self.config.num_pipeline_repeats > 1:
            if is_already_staged:
                res = jnp.stack([leaf] * self.config.num_pipeline_repeats, axis=0)
            else:
                leaf_staged = jnp.stack([leaf] * self.num_stages, axis=0)
                res = jnp.stack([leaf_staged] * self.config.num_pipeline_repeats, axis=0)
        else:
            if is_already_staged:
                res = leaf
            else:
                res = jnp.stack([leaf] * self.num_stages, axis=0)
        
        return res

    log_trace("__init__", "Applying lift_state to tree...")
    lifted_state = jax.tree_util.tree_map_with_path(lift_state, state)
    
    # CRITICAL: Update the module with the lifted (Rank 4) arrays
    log_trace("__init__", "Updating layers with lifted state...")
    nnx.update(layers, lifted_state)
    
    # -------------------------------------------------------------------------
    # 2. CRITICAL FIX: Sharding Metadata Patching
    # -------------------------------------------------------------------------
    log_trace("patching", "Starting State-Guided Patching...")
    
    patched_count = 0
    scanned_count = 0
    
    def get_node_from_path(root, path_keys):
        curr = root
        try:
            for key in path_keys:
                if hasattr(key, 'key'):
                    k = key.key
                else:
                    k = key
                
                if isinstance(curr, (list, tuple, nnx.List, nnx.Sequential)):
                    curr = curr[int(k)]
                elif isinstance(curr, (dict, nnx.Dict)):
                    curr = curr[k]
                elif hasattr(curr, str(k)):
                    curr = getattr(curr, str(k))
                else:
                    return None
            return curr
        except (KeyError, IndexError, AttributeError, TypeError):
            return None

    flat_lifted_state = nnx.traversals.flatten_mapping(lifted_state)
    
    for path, val in flat_lifted_state.items():
        scanned_count += 1
        if len(path) < 1: continue
        param_path = path[:-1]
        
        node = get_node_from_path(layers, param_path)
        
        if node is not None:
            spec = getattr(node, 'sharding', None)
            
            if spec is not None and isinstance(spec, (tuple, list, jax.sharding.PartitionSpec)):
                if hasattr(val, 'ndim'):
                    array_rank = val.ndim
                    spec_len = len(spec)
                    
                    if array_rank > spec_len:
                        diff = array_rank - spec_len
                        
                        spec_list = list(spec) if isinstance(spec, (tuple, list)) else list(spec)
                        current_nones = 0
                        for x in spec_list:
                            if x is None: current_nones += 1
                            else: break
                        
                        if current_nones < diff:
                            needed = diff - current_nones
                            prefix = [None] * needed
                            new_spec_list = prefix + spec_list
                            new_spec = jax.sharding.PartitionSpec(*new_spec_list)
                            
                            try:
                                node.sharding = new_spec
                                patched_count += 1
                            except Exception as e:
                                log_trace("patching", f"Error updating {param_path}: {e}")

    log_trace("patching", f"Patch complete. Fixed {patched_count} parameters.")

    self.layers = layers 

    self.forwarding_delay = 2 if self.config.pipeline_delay_activation_forwarding else 1
    self.pipeline_microbatch_size = self.config.micro_batch_size_to_train_on // self.config.num_pipeline_microbatches
    self.microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages
    self.use_circ_storage = self.need_circ_storage()

    self.spmd_axis_name = "stage" if self.config.shard_mode == ShardMode.AUTO else None

    # Sharding Configs
    self.stages_in_logical = ("activation_stage", "activation_batch_no_exp", "activation_length", "activation_embed") if self.config.expert_shard_attention_option == EP_AS_CONTEXT else ("activation_stage", "activation_batch", "activation_length_no_exp", "activation_embed")
    self.stages_in_spec = logical_to_mesh_axes(self.stages_in_logical, self.mesh, rules=self.config.logical_axis_rules)
    self.stages_in_sharding = (
        NamedSharding(self.mesh, self.stages_in_spec) if self.config.shard_mode == ShardMode.EXPLICIT else None
    )

    self.state_io_logical = ("activation_stage", None) + self.stages_in_logical[1:]
    self.state_io_spec = logical_to_mesh_axes(self.state_io_logical, self.mesh, rules=self.config.logical_axis_rules)
    self.state_io_sharding = (
        NamedSharding(self.mesh, self.state_io_spec) if self.config.shard_mode == ShardMode.EXPLICIT else None
    )
    
    self.input_sharding = (
        create_sharding(
            self.mesh,
            (None,) + self.stages_in_logical[1:],
            rules=self.config.logical_axis_rules,
        )
        if self.config.shard_mode == ShardMode.EXPLICIT else None
    )
    self.output_sharding = self.input_sharding

    log_trace("__init__", "Complete")

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
        inputs, logical_axes, shard_mode=self.config.shard_mode,
        mesh=self.mesh, rules=self.config.logical_axis_rules,
        debug_sharding=self.config.debug_sharding,
    )

  def _maybe_shard_with_name(self, inputs, sharding_name):
    return maybe_shard_with_name(
        inputs, sharding_name, shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
    )

  def init_states(self, inputs):
    log_trace("init_states", f"Input shape: {inputs.shape}")
    shift = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
    shift = self._maybe_shard_with_logical(shift, self.stages_in_logical)

    if self.config.pipeline_delay_activation_forwarding:
      prev_outputs = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
      prev_outputs = self._maybe_shard_with_logical(prev_outputs, self.stages_in_logical)
    else:
      prev_outputs = None

    state_io = jnp.reshape(
        inputs, (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:], out_sharding=self.state_io_sharding
    )
    state_io = self._maybe_shard_with_logical(state_io, self.state_io_logical)

    if self.use_circ_storage:
      circ_storage = jnp.zeros((self.num_stages,) + inputs.shape, dtype=inputs.dtype, out_sharding=self.state_io_sharding)
      circ_storage_mover = shift
    else:
      circ_storage = None
      circ_storage_mover = None

    log_shape("init_states", "state_io", state_io)
    return {
        "state_io": state_io,
        "shift": shift,
        "circ_storage": circ_storage,
        "circ_storage_mover": circ_storage_mover,
        "loop_iteration": jnp.array(0, dtype=jnp.int32),
        "prev_outputs": prev_outputs,
    }

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
      self, 
      x, 
      dim: int, 
      physical_partition_spec: P | None, 
      is_stage_weight: bool = False,
      insert_repeat_dim: bool = False
  ):
    # GUARD: Ignore non-array-likes
    if not hasattr(x, 'ndim') or x.ndim == 0:
        return x

    # FIX: Use None instead of P.UNCONSTRAINED
    placeholder = None 
    
    if physical_partition_spec is None:
      dims_mapping = [placeholder] * x.ndim
    else:
      physical_partition_spec = _remove_fsdp_from_physical_partition_spec(physical_partition_spec)
      dims_mapping = list(physical_partition_spec)
      
      # 1. Insert Repeat Dimension (Index 0)
      if insert_repeat_dim and self.config.num_pipeline_repeats > 1:
          dims_mapping.insert(0, placeholder)
          
      # 2. Insert Stage Dimension
      if dim <= len(dims_mapping):
          dims_mapping.insert(dim, "stage")
      else:
          while len(dims_mapping) < dim:
              dims_mapping.append(placeholder)
          dims_mapping.append("stage")
          
    # Pad to match array rank
    while len(dims_mapping) < x.ndim:
        dims_mapping.append(placeholder)

    dims_mapping = tuple(dims_mapping[:x.ndim])
    
    if physical_partition_spec and is_stage_weight and self.config.shard_mode == ShardMode.EXPLICIT:
      batch_mesh_axis = ["data", "fsdp"]
      reduced_mark = [mesh_axis for mesh_axis in batch_mesh_axis if self.mesh.shape[mesh_axis] > 1]
      pspec = P(*dims_mapping, reduced=set(reduced_mark))
    else:
      pspec = P(*dims_mapping)
      
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
    # GUARD: Ignore non-array-likes
    if not hasattr(weights, 'ndim') or weights.ndim < 2:
        return weights

    def _gather_one(x, repeat_id):
      return jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, repeat_id, 1, repeat_dim_in_weights), repeat_dim_in_weights)

    gathered_weights_stage_dim = 0
    repeat_ids = self.shard_dim_by_stages(repeat_ids, 0, physical_partition_spec=None)
    
    # 1. Shard GLOBAL weights (Has Repeats)
    weights = self.shard_dim_by_stages(
        weights, 
        stages_dim_in_weights, 
        physical_partition_spec=physical_partition_spec, 
        is_stage_weight=True, 
        insert_repeat_dim=True  # <--- MUST BE TRUE
    )
    
    stage_weights = jax.vmap(_gather_one, in_axes=(stages_dim_in_weights, 0), out_axes=gathered_weights_stage_dim)(
        weights, repeat_ids
    )
    
    # 2. Shard GATHERED weights (No Repeats)
    stage_weights = self.shard_dim_by_stages(
        stage_weights, 
        gathered_weights_stage_dim, 
        physical_partition_spec=physical_partition_spec, 
        is_stage_weight=True, 
        insert_repeat_dim=False # <--- MUST BE FALSE
    )
    return stage_weights

  def vmap_gather(self, xs, ids, ids_dim):
    xs = jnp.asarray(xs)
    def _gather_one(x, i):
      idx = tuple(i if d == ids_dim else slice(None) for d in range(x.ndim))
      replicated_sharding = NamedSharding(self.mesh, P())
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
      return jax.lax.ppermute(arr, axis_name="stage", perm=perm)

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
      return jax.lax.ppermute(arr, axis_name="stage", perm=perm)

    def _shift_left(arr, stage_size, output):
      stage_idx = jax.lax.axis_index("stage")
      arr = _rotate_left(arr, stage_size)
      return jnp.where(stage_idx == stage_size - 1, output, arr)

    @jax.shard_map(
        mesh=self.mesh,
        in_specs=(self.state_io_spec, self.stages_in_spec, self.stages_in_spec, P()),
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
    permutation = (np.arange(self.microbatches_per_stage) + microbatch_0_idx) % self.microbatches_per_stage
    output = output[:, permutation]
    return output

  def get_current_stage_weights(self, pipeline_weights_state, loop_iteration, physical_partition_spec=None):
    if self.config.num_pipeline_repeats > 1:
      return self.get_current_repeat_from_stages(
          pipeline_weights_state, loop_iteration, physical_partition_spec=physical_partition_spec
      )
    else:
      return pipeline_weights_state

  def get_current_repeat_from_stages(self, weights, loop_iteration, physical_partition_spec=None):
    _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)
    
    def gather_weights_for_stages_in(w, spec=None):
      return self.vmap_parallel_gather(
          w,
          repeat_ids=repeat_ids,
          repeat_dim_in_weights=0,
          stages_dim_in_weights=1,
          physical_partition_spec=spec,
      )

    if physical_partition_spec is None:
      weights = jax.tree.map(gather_weights_for_stages_in, weights)
    else:
      weights = jax.tree.map(gather_weights_for_stages_in, weights, physical_partition_spec)
    return weights

  def all_gather_over_fsdp(self, variables, logical_partition_spec):
    log_trace("all_gather_over_fsdp", "Start")
    physical_partition_spec = logical_to_mesh(
        logical_partition_spec, mesh=self.mesh, rules=self.config.logical_axis_rules
    )
    
    def process_spec(spec):
        if spec is None: return None
        spec_no_fsdp = _remove_fsdp_from_physical_partition_spec(spec)
        
        if isinstance(spec_no_fsdp, P):
            new_spec = list(spec_no_fsdp)
        elif isinstance(spec_no_fsdp, (list, tuple)):
            new_spec = list(spec_no_fsdp)
        else:
            return spec_no_fsdp

        new_spec.insert(0, None) 
        if self.config.num_pipeline_repeats > 1:
            new_spec.insert(0, None)
            
        return P(*new_spec)

    final_specs = jax.tree.map(process_spec, physical_partition_spec)
    
    return jax.tree.map(
        lambda w, p: self._maybe_shard_with_name(w, NamedSharding(self.mesh, p)),
        variables,
        final_specs,
    )

  def run_one_iteration(
      self,
      loop_state,
      pipeline_weights_state,
      positions,
      segment_ids,
      deterministic,
      model_mode,
      graphdef,
      logical_partition_spec=None,
  ):
    # log_trace("run_one_iteration", "Start")
    state_io = loop_state["state_io"]
    shift = loop_state["shift"]
    circ_storage = loop_state["circ_storage"]
    loop_iteration = loop_state["loop_iteration"]

    microbatch_ids, _ = self.get_microbatch_and_repeat_ids(loop_iteration)

    physical_partition_spec = logical_to_mesh(logical_partition_spec, self.mesh, rules=self.config.logical_axis_rules)

    stages_inputs = self.get_iteration_inputs(loop_iteration, state_io, circ_storage, shift)
    stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, "iteration_input")
    
    if positions is not None:
        stages_positions = self.vmap_gather(positions, microbatch_ids, 0)
    else:
        stages_positions = None
        
    if segment_ids is not None:
        stages_segment_ids = self.vmap_gather(segment_ids, microbatch_ids, 0)
    else:
        stages_segment_ids = None

    stage_weights = self.get_current_stage_weights(
        pipeline_weights_state, loop_iteration, physical_partition_spec=physical_partition_spec
    )
    
    def functional_call(weights, x, seg, pos):
        model = nnx.merge(graphdef, weights)
        return model(x, seg, pos, deterministic, model_mode)

    vmap_args = [stage_weights, stages_inputs]
    vmap_in_axes = [0, 0] 
    
    if stages_segment_ids is not None:
        vmap_args.append(stages_segment_ids)
        vmap_in_axes.append(0) 
    else:
        vmap_args.append(None)
        vmap_in_axes.append(None) 

    if stages_positions is not None:
        vmap_args.append(stages_positions)
        vmap_in_axes.append(0) 
    else:
        vmap_args.append(None)
        vmap_in_axes.append(None)

    vmapped_call = jax.vmap(
        lambda w, x, s, p: functional_call(w, x, s, p), 
        in_axes=tuple(vmap_in_axes)
    )

    stages_output = vmapped_call(*vmap_args)

    if self.config.scan_layers:
      # FIX: Handle tuple vs tensor return types
      if isinstance(stages_output, (tuple, list)):
          stages_output = stages_output[0]

    new_state = self.get_new_loop_state(stages_output, loop_state)
    return new_state

  def get_pipeline_remat_policy(self):
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
      logical_partition_spec=None, 
  ) -> jnp.ndarray:
    
    log_trace("__call__", "Start")
    log_shape("__call__", "inputs", inputs)
    
    inputs = inputs.reshape(
        (
            self.config.num_pipeline_microbatches,
            self.pipeline_microbatch_size,
            self.config.max_target_length,
            self.config.emb_dim,
        ),
        out_sharding=self.input_sharding,
    )

    ag_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(None, None))

    if positions is not None:
        positions = self._maybe_shard_with_name(positions, ag_sharding)
        positions = positions.reshape(
            (
                self.config.num_pipeline_microbatches, 
                self.pipeline_microbatch_size, 
                self.config.max_target_length
            )
        )
    
    if segment_ids is not None:
        segment_ids = self._maybe_shard_with_name(segment_ids, ag_sharding)
        segment_ids = segment_ids.reshape(
            (
                self.config.num_pipeline_microbatches, 
                self.pipeline_microbatch_size, 
                self.config.max_target_length
            )
        )

    loop_state = self.init_states(inputs)
    
    graphdef, layer_state = nnx.split(self.layers)

    if self.config.pipeline_fsdp_ag_once:
      layer_state = self.all_gather_over_fsdp(layer_state, logical_partition_spec)

    bubble_iterations = self.forwarding_delay * (self.num_stages - 1)
    real_iterations = self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats
    total_iterations = real_iterations + bubble_iterations
    log_trace("__call__", f"Total Iterations: {total_iterations}")

    def scan_body(carry, _):
        new_loop_state = self.run_one_iteration(
            carry,
            layer_state,
            positions,
            segment_ids,
            deterministic,
            model_mode,
            graphdef,
            logical_partition_spec
        )
        return new_loop_state, None

    if self.config.set_remat_policy_on_pipeline_iterations:
        scan_body = jax.checkpoint(
            scan_body, 
            policy=self.get_pipeline_remat_policy()
        )

    if self.config.scan_pipeline_iterations:
         final_loop_state, _ = jax.lax.scan(scan_body, loop_state, None, length=total_iterations)
    else:
         final_loop_state = loop_state
         for i in range(total_iterations):
             final_loop_state, _ = scan_body(final_loop_state, None)

    final_output = self.permute_output_micro_per_stage_dim(final_loop_state["state_io"])
    final_output = jnp.reshape(
        final_output,
        (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim),
        out_sharding=self.output_sharding,
    )
    
    log_trace("__call__", "Complete")
    return final_output


from flax import linen as nn
from MaxText.layers import nnx_wrappers


def pipeline_as_linen(
    config: Config,
    mesh: Mesh,
    layers: nnx.Module,
    remat_policy: Any = None,
) -> nn.Module:
  """Wraps the Pipeline NNX module to behave like a Linen module."""
  return nnx_wrappers.to_linen(
      Pipeline,
      config=config,
      mesh=mesh,
      layers=layers,
      remat_policy=remat_policy,
  )


