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
def log_debug(tag, obj, show_value=False):
  """Helper to print type, shape, and optional value for debugging."""
  prefix = f"[DEBUG] {tag}:"
  type_str = str(type(obj))

  if obj is None:
    print(f"{prefix} None")
    return

  # Handle JAX/Numpy Arrays
  if hasattr(obj, "shape"):
    shape_str = str(obj.shape)
    dtype_str = str(obj.dtype) if hasattr(obj, "dtype") else "unknown"
    val_str = ""
    if show_value or (np.prod(obj.shape) < 10):
      val_str = f" | Value: {obj}"
    print(f"{prefix} Type={type_str} | Shape={shape_str} | Dtype={dtype_str}{val_str}")

  # Handle Lists/Tuples
  elif isinstance(obj, (list, tuple)):
    print(f"{prefix} Type={type_str} | Len={len(obj)}")
    for i, item in enumerate(obj):
      log_debug(f"  {tag}[{i}]", item, show_value)

  # Handle Scalars
  else:
    print(f"{prefix} Type={type_str} | Value={obj}")


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
  """NNX implementation of Pipeline parallelism with Debugging."""

  def __init__(
      self, 
      config: Config, 
      mesh: Mesh, 
      layers: nnx.Module, 
      remat_policy: Any = None,
      **kwargs,
  ):
    print(f"[DEBUG] Initializing Pipeline Module...")
    self.config = config
    self.mesh = mesh
    self.remat_policy = remat_policy
    
    self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
    
    # 1. Lift the State (Arrays) to [Repeat, Stage, ...]
    graphdef, state = nnx.split(layers)
    
    def lift_state(leaf):
        if not hasattr(leaf, 'shape'): return leaf
        if self.config.num_pipeline_repeats > 1:
            leaf_staged = jnp.stack([leaf] * self.num_stages, axis=0)
            leaf_repeated = jnp.stack([leaf_staged] * self.config.num_pipeline_repeats, axis=0)
            return leaf_repeated
        else:
            return jnp.stack([leaf] * self.num_stages, axis=0)

    lifted_state = jax.tree.map(lift_state, state)
    
    # 2. Update the Module with Lifted Arrays
    nnx.update(layers, lifted_state)
    
    # 3. CRITICAL FIX: Update Sharding Metadata (Robust Traversal)
    print("[DEBUG] Updating Sharding Metadata (Manual Recursive Traversal)...")
    
    def adjust_spec(spec):
        if spec is None: return None
        # Handle PartitionSpec or tuple/list representations
        if isinstance(spec, (tuple, list, jax.sharding.PartitionSpec)):
            new_spec = list(spec)
            new_spec.insert(0, None) # Placeholder for Stage
            if self.config.num_pipeline_repeats > 1:
                new_spec.insert(0, None) # Placeholder for Repeat
            return jax.sharding.PartitionSpec(*new_spec)
        return spec

    def recursive_update(obj, visited):
        # Prevent infinite recursion
        if id(obj) in visited:
            return
        visited.add(id(obj))
        
        # A. If it's a Variable/Param with sharding, update it
        if hasattr(obj, 'sharding') and hasattr(obj, 'value'): # Duck typing for nnx.Variable
            if obj.sharding is not None:
                old_spec = obj.sharding
                obj.sharding = adjust_spec(old_spec)
                # print(f"  > Updated {type(obj).__name__}: {old_spec} -> {obj.sharding}")
            return

        # B. If it's a Module, recurse into attributes
        if isinstance(obj, nnx.Module):
            for name, value in vars(obj).items():
                recursive_update(value, visited)
        
        # C. If it's a container, recurse into items
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                recursive_update(item, visited)
        elif isinstance(obj, dict):
            for value in obj.values():
                recursive_update(value, visited)

    # Start traversal
    visited_ids = set()
    recursive_update(layers, visited_ids)
    print(f"[DEBUG] Traversal complete. Visited {len(visited_ids)} unique objects.")

    self.layers = layers 

    self.forwarding_delay = 2 if self.config.pipeline_delay_activation_forwarding else 1
    self.pipeline_microbatch_size = self.config.micro_batch_size_to_train_on // self.config.num_pipeline_microbatches
    self.microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages
    self.use_circ_storage = self.need_circ_storage()

    if self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      self.batch_axis_name = "activation_batch_no_exp"
      self.seq_len_axis_name = "activation_length"
    else:
      self.batch_axis_name = "activation_batch"
      self.seq_len_axis_name = "activation_length_no_exp"

    self.spmd_axis_name = "stage" if self.config.shard_mode == ShardMode.AUTO else None

    # Sharding Configs
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
        if self.config.shard_mode == ShardMode.EXPLICIT else None
    )
    self.output_sharding = (
        create_sharding(
            self.mesh,
            (self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
            rules=self.config.logical_axis_rules,
        )
        if self.config.shard_mode == ShardMode.EXPLICIT else None
    )
  
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
        debug_sharding=self.config.debug_sharding,
    )

  def _maybe_shard_with_name(self, inputs, sharding_name):
    return maybe_shard_with_name(
        inputs,
        sharding_name,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
    )

  def init_states(self, inputs):
    print(f"[DEBUG] init_states called with input shape: {inputs.shape}")

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

    log_debug("init_states.state_io", state_io)

    if self.use_circ_storage:
      circ_storage = jnp.zeros((self.num_stages,) + inputs.shape, dtype=inputs.dtype, out_sharding=self.state_io_sharding)
      circ_storage_mover = shift
    else:
      circ_storage = None
      circ_storage_mover = None

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
    # --- DEBUG: Catch complex specs appearing unexpectedly ---
    if is_stage_weight and physical_partition_spec is not None:
         # Check if spec is complex (tuples/strings)
         is_complex = any(isinstance(dim, (tuple, str)) for dim in physical_partition_spec)
         if is_complex:
             print(f"[DEBUG] shard_dim_by_stages receiving COMPLEX SPEC!")
             print(f"  > x.shape: {x.shape}")
             print(f"  > spec: {physical_partition_spec}")
             print(f"  > insert_repeat: {insert_repeat_dim}")
    # -------------------------------------------------------

    placeholder = None if self.config.shard_mode == ShardMode.EXPLICIT else P.UNCONSTRAINED
    
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

    # Truncate
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
    def _gather_one(x, repeat_id):
      return jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, repeat_id, 1, repeat_dim_in_weights), repeat_dim_in_weights)

    gathered_weights_stage_dim = 0
    repeat_ids = self.shard_dim_by_stages(repeat_ids, 0, physical_partition_spec=None)
    
    # 1. Shard GLOBAL weights (Has Repeats)
    # CRITICAL: We MUST set insert_repeat_dim=True here
    weights = self.shard_dim_by_stages(
        weights, 
        stages_dim_in_weights, 
        physical_partition_spec=physical_partition_spec, 
        is_stage_weight=True, 
        insert_repeat_dim=True 
    )
    
    stage_weights = jax.vmap(_gather_one, in_axes=(stages_dim_in_weights, 0), out_axes=gathered_weights_stage_dim)(
        weights, repeat_ids
    )
    
    # 2. Shard GATHERED weights (No Repeats)
    # CRITICAL: We MUST set insert_repeat_dim=False here (Repeats stripped)
    stage_weights = self.shard_dim_by_stages(
        stage_weights, 
        gathered_weights_stage_dim, 
        physical_partition_spec=physical_partition_spec, 
        is_stage_weight=True,
        insert_repeat_dim=False
    )
    return stage_weights

  def vmap_gather(self, xs, ids, ids_dim):
    """Use vmap to implement a stage-wise sharded gather."""
    log_debug("vmap_gather input xs", xs)
    log_debug("vmap_gather input ids", ids)

    # FIX: Convert to JAX array explicitly to avoid AttributeError on .at[...]
    xs = jnp.asarray(xs)
    log_debug("vmap_gather xs converted to JAX", xs)

    def _gather_one(x, i):
      # Helper log inside vmap
      # print(f"DEBUG: _gather_one x.shape={x.shape}, i={i}")
      idx = tuple(i if d == ids_dim else slice(None) for d in range(x.ndim))
      replicated_sharding = NamedSharding(self.mesh, P())
      return x.at[idx].get(out_sharding=replicated_sharding)

    ids = self.shard_dim_by_stages(ids, 0, physical_partition_spec=None)
    outs = jax.vmap(_gather_one, in_axes=(None, 0), out_axes=ids_dim)(xs, ids)

    log_debug("vmap_gather output", outs)
    return self.shard_dim_by_stages(outs, 0, physical_partition_spec=None)

  def get_new_loop_state(self, output, loop_state):
    """
    Update the various buffers given the output of the most recent iteration
    * state_io: rotates left/up by 1 (the whole created in the last slot is filled with the most recent pipeline output)
       * Pushing inputs up from top of state_io into first stage of shift
       * Pulling outputs up from last stage of shift into bottom of state_io
    * shift: rotate output (or prev_outputs if using delay) right/down by 1 - we imagine the pipeline moves to
               right/down
    * circ_storage: pushes circ_storage_mover (the output of the previous iteration) into rotating index of circ_storage
    * circ_storage_mover: assigned to rotated output and pushed into circ_storage on the next iteration
    * prev_outputs: is set to the current output
    """
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

    # Shift either rotates or shifts depending on if the last stage immediately must send to first or not
    # For non-circular pipelines, the last stage does not need to send to first
    # For circular pipelines with #micro = #stages, last stage immediately sends to first
    # For circular pipelines with #micro > stages (circ_storage), last stage sends to circ storage
    def _update_shift(output_in):
      if self.config.num_pipeline_repeats == 1 or self.use_circ_storage:
        return _shift_right(output_in)
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
    print(f"\n[DEBUG] --- Inside all_gather_over_fsdp ---")
    
    # 1. Check Inputs
    # Inspect the first leaf of variables to see rank
    flat_vars = jax.tree.leaves(variables)
    if flat_vars:
        print(f"  > Variable[0] Shape: {flat_vars[0].shape}")
    
    # Inspect logical spec
    if logical_partition_spec is None:
        print(f"  > Input logical_partition_spec is None")
    else:
        print(f"  > Input logical_partition_spec is VALID (Not None)")
        # Print a sample
        flat_specs = jax.tree.leaves(logical_partition_spec)
        if flat_specs:
            print(f"  > Sample Logical Spec: {flat_specs[0]}")

    # 2. Convert to Physical
    physical_partition_spec = logical_to_mesh(
        logical_partition_spec, mesh=self.mesh, rules=self.config.logical_axis_rules
    )
    
    # 3. Process Spec (The Fix logic + Logging)
    def process_spec(spec):
        if spec is None: 
            return None
        
        # Original logic: remove fsdp
        spec_no_fsdp = _remove_fsdp_from_physical_partition_spec(spec)
        
        if isinstance(spec_no_fsdp, P):
            new_spec = list(spec_no_fsdp)
        elif isinstance(spec_no_fsdp, (list, tuple)):
            new_spec = list(spec_no_fsdp)
        else:
            return spec_no_fsdp

        # Log before modification
        # print(f"    >> Processing Spec: {new_spec} for Rank 4 Array")

        # INSERT STAGE (Dim 1)
        new_spec.insert(0, None) 
        
        # INSERT REPEAT (Dim 0)
        if self.config.num_pipeline_repeats > 1:
            new_spec.insert(0, None)
            
        return P(*new_spec)

    final_specs = jax.tree.map(process_spec, physical_partition_spec)
    
    # Log sample final spec
    flat_final = jax.tree.leaves(final_specs)
    if flat_final:
        print(f"  > Final Physical Spec Sample: {flat_final[0]}")
    
    print(f"[DEBUG] -----------------------------------\n")

    return jax.tree.map(
        lambda w, p: self._maybe_shard_with_name(w, NamedSharding(self.mesh, p)),
        variables,
        final_specs,
    )

  # --- CRITICAL SECTION: RUN_ONE_ITERATION WITH LOGGING ---
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
    """Run one loop iteration with robust None handling and logging."""
    # print(f"[DEBUG] --- Starting run_one_iteration ---")
    state_io = loop_state["state_io"]
    shift = loop_state["shift"]
    circ_storage = loop_state["circ_storage"]
    loop_iteration = loop_state["loop_iteration"]

    microbatch_ids, _ = self.get_microbatch_and_repeat_ids(loop_iteration)

    physical_partition_spec = logical_to_mesh(logical_partition_spec, self.mesh, rules=self.config.logical_axis_rules)

    stages_inputs = self.get_iteration_inputs(loop_iteration, state_io, circ_storage, shift)
    stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, "iteration_input")

    # log_debug("stages_inputs", stages_inputs)

    # Gather inputs or set to None
    if positions is not None:
      stages_positions = self.vmap_gather(positions, microbatch_ids, 0)
      # log_debug("stages_positions (gathered)", stages_positions)
    else:
      # print("[DEBUG] positions is None")
      stages_positions = None

    if segment_ids is not None:
      stages_segment_ids = self.vmap_gather(segment_ids, microbatch_ids, 0)
      # log_debug("stages_segment_ids (gathered)", stages_segment_ids)
    else:
      # print("[DEBUG] segment_ids is None")
      stages_segment_ids = None

    stage_weights = self.get_current_stage_weights(
        pipeline_weights_state, loop_iteration, physical_partition_spec=physical_partition_spec
    )

    # --- FUNCTIONAL CALL DEFINITION ---
    # Define this clearly to avoid lambda confusion
    def functional_call(weights, x, seg, pos):
      # DEBUG LOG inside the vmapped function
      # This will print dimensions AFTER vmap slicing (should be Batch x Seq, not Stage x Batch x Seq)
      print(f"[DEBUG] Inside functional_call:")
      log_debug("  Input x", x)
      log_debug("  Input seg", seg)
      log_debug("  Input pos", pos)

      # Merge state back into graph to get callable module
      model = nnx.merge(graphdef, weights)
      return model(x, seg, pos, deterministic, model_mode)

    # --- DYNAMIC VMAP CONSTRUCTION ---
    vmap_args = [stage_weights, stages_inputs]
    vmap_in_axes = [0, 0]

    # Handle segment_ids
    if stages_segment_ids is not None:
      vmap_args.append(stages_segment_ids)
      vmap_in_axes.append(0)
    else:
      vmap_args.append(None)
      vmap_in_axes.append(None)

    # Handle positions
    if stages_positions is not None:
      vmap_args.append(stages_positions)
      vmap_in_axes.append(0)
    else:
      vmap_args.append(None)
      vmap_in_axes.append(None)

    # Log what we are about to call
    # log_debug("VMAP Arguments count", len(vmap_args))
    # log_debug("VMAP Axes", vmap_in_axes)

    # Construct the vmap
    # We must match the args to (weights, x, seg, pos)
    vmapped_call = jax.vmap(lambda w, x, s, p: functional_call(w, x, s, p), in_axes=tuple(vmap_in_axes))

    # Execute
    stages_output = vmapped_call(*vmap_args)

    if self.config.scan_layers:
      stages_output = stages_output[0]

    new_state = self.get_new_loop_state(stages_output, loop_state)
    return new_state

  def get_pipeline_remat_policy(self):
    """Returns the pipeline remat policy for this pipeline."""
    # We ensure that the decoder layer inputs are saved, although we leave it to a custom
    # policy if they should be saved to device or offloaded.
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

    print(f"\n[DEBUG] Pipeline.__call__ Started")
    # log_debug("Inputs Raw", inputs)

    print("\n[DEBUG] --- Logical Partition Spec in __call__ ---")
    if logical_partition_spec is not None:
        def log_spec(path, spec):
            print(f"Spec: {path} | PartitionSpec: {spec}")
        jax.tree_util.tree_map_with_path(lambda p, x: log_spec(p, x), logical_partition_spec)
    else:
        print("logical_partition_spec is None")
    print("[DEBUG] ------------------------------------------\n")


    # 1. Reshape Inputs (Existing correct code)
    inputs = inputs.reshape(
        (
            self.config.num_pipeline_microbatches,
            self.pipeline_microbatch_size,
            self.config.max_target_length,
            self.config.emb_dim,
        ),
        out_sharding=self.input_sharding,
    )

    # 2. FIX: Reshape Positions and Segment IDs
    # We must break the global batch (e.g. 16) into (NumMicro, MicroSize) -> (4, 4)
    # The sharding logic ensures they are distributed correctly before reshape.

    ag_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(None, None))

    if positions is not None:
      positions = self._maybe_shard_with_name(positions, ag_sharding)
      positions = positions.reshape(
          (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
      )
      # log_debug("Positions Reshaped", positions)

    if segment_ids is not None:
      segment_ids = self._maybe_shard_with_name(segment_ids, ag_sharding)
      segment_ids = segment_ids.reshape(
          (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
      )
      # log_debug("Segment IDs Reshaped", segment_ids)

    # 3. Initialize Loop
    loop_state = self.init_states(inputs)

    graphdef, layer_state = nnx.split(self.layers)

    if self.config.pipeline_fsdp_ag_once:
      layer_state = self.all_gather_over_fsdp(layer_state, logical_partition_spec)

    # Each microbatch should go through each stage (with repeats) - so there is num_micro * (num_stages * repeats)
    # compute to perform
    # Each iteration is vmapped by num_stages, so the number of iterations should be
    # num_micro * num_stages * repeats / num_stages = num_micro * repeats
    # However due to the pipeline bubble some iterations process less than num_stages microbatches. It takes
    # num_micro * repeat iterations for the last microbatch to start the final repeat, then an additional
    # num_stages - 1 to finish the final repeat.
    # Thus the total iterations is num_micro * repeat + num_stages - 1, & we may consider the num_stages - 1 as bubble.
    # The bubble doubles when we use forwarding delay.
    bubble_iterations = self.forwarding_delay * (self.num_stages - 1)
    real_iterations = self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats
    total_iterations = real_iterations + bubble_iterations

    print(f"[DEBUG] Scan Configuration: Total Iterations={total_iterations}")

    def scan_body(carry, _):
      new_loop_state = self.run_one_iteration(
          carry, layer_state, positions, segment_ids, deterministic, model_mode, graphdef, logical_partition_spec
      )
      return new_loop_state, None

    if self.config.set_remat_policy_on_pipeline_iterations:
      scan_body = jax.checkpoint(scan_body, policy=self.get_pipeline_remat_policy())

    if self.config.scan_pipeline_iterations:
      loop_state, _ = jax.lax.scan(scan_body, loop_state, None, length=total_iterations)
    else:
      for _ in range(total_iterations):
        loop_state, _ = scan_body(loop_state, None)

    final_output = self.permute_output_micro_per_stage_dim(loop_state["state_io"])
    final_output = jnp.reshape(
        final_output,
        (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim),
        out_sharding=self.output_sharding,
    )

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


