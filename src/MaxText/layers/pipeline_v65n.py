# Copyright 2023-2026 Google LLC
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

"""V65n: Sharded-carry overlap variant of V65i.

Design rationale
================
V65i computes BOTH w_curr and w_next *before* the inner microbatch scan begins.
Both use all-gather, and both must complete before the first microbatch can
start.  The XLA scheduler has no opportunity to overlap the all-gathers with
compute because they are strict data dependencies of the scan body.

V65n restructures the outer repeat scan to carry **sharded** repeat weights
(pre-all-gather) from the previous repeat.  At the start of each repeat:

  1. w_curr = all-gather(carried_sharded_prev)    -- 1 all-gather
  2. sharded_next = dynamic_slice(all_params, next_repeat)  -- no communication
  3. w_next = all-gather(sharded_next)             -- 1 all-gather

The key difference from V65i: the **carry** is the sharded (pre-all-gather)
weight slice, NOT the fully gathered BSW.  This keeps carry size proportional
to per-device weight size (same as loop_state), NOT per-repeat all-gathered
size.

Memory profile expectation:
  - V65i: 21.3 GB (no weight carry, 2 all-gathers per repeat, cond overhead)
  - V65n: ~21-22 GB (sharded carry is small, 2 all-gathers per repeat,
                      no cond -- always all-gather for w_curr)
  - V65 (original): 25.1 GB (unsharded BSW carry blows up memory)

Performance expectation:
  - Same throughput as V65i (same number of all-gathers)
  - Eliminates jax.lax.cond for iteration-0 zeros (unconditional all-gather
    produces cleaner HLO, avoids cond overhead)
  - XLA may overlap the two all-gathers better since w_curr comes from carry
    (already materialized) and w_next is a fresh slice (independent)

Implementation notes:
  - Based on pipeline_v65i.py NNXCircularPipeline
  - Same NNXPipelineBase, NNXPipeline (non-circular) unchanged
  - Only NNXCircularPipeline.__call__ is modified for the sharded-carry pattern
  - No custom_vjp anywhere -- pure jax.lax.scan with SPMD-friendly carries
  - If this converges to the same HLO as V65i (XLA eliminates the cond
    anyway), the variant documents the equivalence for future reference
"""

from typing import Any

import numpy as np

from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import jax
import jax.ad_checkpoint

from aqt.jax.v2 import aqt_tensor
from flax import linen as nn
from flax import nnx
from maxtext.layers import initializers
from maxtext.layers.nnx_wrappers import is_linen_initializing, to_linen_class

from maxtext.common.common_types import Config, MODEL_MODE_TRAIN, ShardMode
from maxtext.utils.sharding import (
    maybe_shard_with_logical,
    maybe_shard_with_name,
    create_sharding,
    logical_to_mesh_axes,
    logical_to_mesh,
)
from maxtext.utils import pipeline_utils

# Import shared base and helpers from V65i to avoid duplication.
from maxtext.layers.pipeline_v65i import (
    _is_static_param,
    _advance_rng_state,
    is_spec_leaf,
    NNXPipelineBase,
    NNXPipeline,
)


class NNXCircularPipelineShardedCarry(NNXPipelineBase):
  """NNX circular pipeline with sharded-carry overlap pattern.

  Key difference from NNXCircularPipeline (V65i):
    - Carries **sharded** repeat weights (pre-all-gather) across outer scan
      iterations instead of using jax.lax.cond to select between zeros and
      a prefetched BSW for w_curr.
    - At each repeat: all-gather the carried sharded weights to produce w_curr,
      then slice+all-gather for w_next.  Two all-gathers per repeat, same as
      V65i, but the carry is small (sharded) and no cond is needed.
  """

  def get_main_vmap_func_for_iterations(self):
    """Override: vmap returns only non-param state to avoid stacking params across stages."""

    def func_to_vmap(graph, state, stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode):
      module = nnx.merge(graph, state)
      out = module(stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode)
      _, _, updated_metrics, updated_mutables = nnx.split(module, _is_static_param, nnx.Intermediate, ...)
      return out, (updated_metrics, updated_mutables)

    vmap_kwargs = {
        "in_axes": (None, 0, 0, 0, 0, None, None),
        "out_axes": (0, (0, 0)),
    }
    if self.spmd_axis_name is not None:
      vmap_kwargs["spmd_axis_name"] = self.spmd_axis_name
    return jax.vmap(func_to_vmap, **vmap_kwargs)

  def gather_microbatch_inputs_vmap(self, xs, ids, ids_dim):
    """Slices out the specific sequence inputs (e.g., positions, segments) for the current microbatch."""
    if xs is None:
      return None

    xs = jnp.asarray(xs)
    ndim = xs.ndim

    def _gather_one(x, i):
      idx = tuple(i if d == ids_dim else slice(None) for d in range(ndim))
      positions_sharding = (
          create_sharding(self.mesh, (None, "layers", "activation_length"))
          if self.config.shard_mode == ShardMode.EXPLICIT
          else None
      )
      return x.at[idx].get(out_sharding=positions_sharding)

    return jax.vmap(_gather_one, in_axes=(None, 0), out_axes=ids_dim)(xs, ids)

  def gather_weights_across_stages_vmap(self, weights_state, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights):
    """Uses jax.vmap to dynamically slice and gather weights for specific pipeline repeats."""

    def _gather_repeat_leaf(w_leaf, rep_id):
      if w_leaf is None:
        return None
      return jnp.squeeze(
          jax.lax.dynamic_slice_in_dim(w_leaf, rep_id, 1, axis=repeat_dim_in_weights), axis=repeat_dim_in_weights
      )

    vmap_gather = jax.vmap(_gather_repeat_leaf, in_axes=(stages_dim_in_weights, 0), out_axes=0)
    return jax.tree.map(lambda w: vmap_gather(w, repeat_ids) if w is not None else None, weights_state)

  def from_all_variables_to_repeat_weights(self, weights_state, loop_iteration):
    """Slices out the specific repeat's weights from the full weights state."""
    if self.config.num_pipeline_repeats == 1:
      return weights_state

    _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)

    return self.gather_weights_across_stages_vmap(
        weights_state, repeat_ids=repeat_ids, repeat_dim_in_weights=0, stages_dim_in_weights=1
    )

  def from_repeat_weights_to_bsw(
      self,
      repeat_weights,
      physical_partition_spec,
      axes_to_gather=("fsdp", "fsdp_transpose", "context", "expert"),
      use_shardmap=False,
  ):
    """Executes the FSDP-like all-gathers to fully materialize a block of weights for the BSW."""
    axes_to_remove = ["fsdp", "fsdp_transpose", "context"]
    if physical_partition_spec is not None:
      bsw_pps = pipeline_utils.derive_stage_weight_partition_specs(physical_partition_spec, axes_to_remove)
    else:
      bsw_pps = None

    def _from_repeat_weights_to_bsw_shardmap(
        repeat_weights,
        physical_partition_spec,
        axes_to_gather,
    ):
      repeat_weights_pps = jax.tree.map(
          lambda p: P(*p[1:]) if isinstance(p, P) else p,
          physical_partition_spec,
          is_leaf=is_spec_leaf,
      )

      axis_indices_dict = {
          axis: pipeline_utils.get_mesh_axis_dim_indices(physical_partition_spec, axis) for axis in axes_to_gather
      }

      axis_names = list(axis_indices_dict.keys())
      axis_pytrees = list(axis_indices_dict.values())

      def should_skip_gather(axis_name, path_keys):
        if axis_name == "expert" and "MoeBlock_0" in path_keys:
          return True
        return False

      weights_treedef = jax.tree.structure(repeat_weights)
      pps_treedef = jax.tree.structure(repeat_weights_pps, is_leaf=is_spec_leaf)
      weights_leaves = jax.tree.leaves(repeat_weights)
      assert pps_treedef.num_leaves == len(weights_leaves), (
          f"repeat_weights/spec leaf count mismatch: specs={pps_treedef.num_leaves}, " f"weights={len(weights_leaves)}"
      )
      raw_weights = pps_treedef.unflatten(weights_leaves)

      @jax.shard_map(
          mesh=self.mesh,
          in_specs=(repeat_weights_pps, None),
          out_specs=bsw_pps,
          check_vma=False,
      )
      def _shard_map_gather_weights(sharded_weights, indices_pytrees_list):

        def _gather_tensor_along_axes(path, x, *indices):
          path_keys = [getattr(p, "key", str(p)) for p in path]

          for axis_name, axis_idx in zip(axis_names, indices):
            if axis_idx >= 0 and not should_skip_gather(axis_name, path_keys):
              x = jax.lax.all_gather(x, axis_name=axis_name, axis=axis_idx - 1, tiled=True)
          return x

        return jax.tree_util.tree_map_with_path(_gather_tensor_along_axes, sharded_weights, *indices_pytrees_list)

      raw_bsw = _shard_map_gather_weights(raw_weights, axis_pytrees)
      return weights_treedef.unflatten(jax.tree.leaves(raw_bsw))

    def _from_repeat_weights_to_bsw_hint(repeat_weights):
      def _apply_sharding_hint(weight, pspec):
        if pspec is None or weight is None:
          return weight
        sharding_name = NamedSharding(self.mesh, pspec)
        return maybe_shard_with_name(
            weight,
            sharding_name,
            shard_mode=self.config.shard_mode,
            debug_sharding=self.config.debug_sharding,
            extra_stack_level=0,
        )

      spec_leaves = jax.tree_util.tree_leaves(bsw_pps, is_leaf=is_spec_leaf)
      spec_iter = iter(spec_leaves)
      return jax.tree.map(lambda w: _apply_sharding_hint(w, next(spec_iter)), repeat_weights)

    if bsw_pps is None:
      return repeat_weights

    if use_shardmap:
      return _from_repeat_weights_to_bsw_shardmap(repeat_weights, physical_partition_spec, axes_to_gather=axes_to_gather)
    return _from_repeat_weights_to_bsw_hint(repeat_weights)

  def weight_prefetching(self, weights_state, physical_partition_spec, loop_iteration):
    """Prefetch next repeat's weights for the Buffer Sliding Window."""
    nxt_repeat_weights = self.from_all_variables_to_repeat_weights(weights_state, loop_iteration + 1)
    return self.from_repeat_weights_to_bsw(nxt_repeat_weights, physical_partition_spec)

  def fetch_active_stage_weights(self, bsw, loop_iteration, physical_partition_spec=None):
    """Fetches the actively prefetched weights from the BSW."""
    return self.get_current_weights_from_bsw(bsw, loop_iteration, physical_partition_spec)

  def get_current_weights_from_bsw(self, bsw, loop_iteration, physical_partition_spec):
    """Pulls the fully gathered parameters for the current repeat from the BSW dual-buffer."""
    # Fast path: both BSW slots are the same object -> skip shard_map select.
    if bsw[0] is bsw[1]:
      treedef = jax.tree.structure(bsw[0])
      leaves = jax.tree.leaves(bsw[0])
      return treedef.unflatten(leaves)

    bsw_pps = jax.tree.map(self._remove_fsdp_from_physical_partition_spec, physical_partition_spec)
    _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)
    stage0_repeat_id = jnp.maximum(loop_iteration, 0) // self.config.num_pipeline_microbatches

    if bsw_pps is not None:
      bsw_treedef = jax.tree.structure(bsw[0])

      pps_treedef = jax.tree.structure(bsw_pps, is_leaf=is_spec_leaf)
      bsw0_leaves = jax.tree.leaves(bsw[0])
      bsw1_leaves = jax.tree.leaves(bsw[1])
      assert bsw_treedef == jax.tree.structure(
          bsw[1]
      ), "BSW half-tree structure mismatch: bsw[0] and bsw[1] must be structurally identical but differ."
      assert pps_treedef.num_leaves == len(bsw0_leaves) == len(bsw1_leaves), (
          f"BSW/spec leaf count mismatch: specs={pps_treedef.num_leaves}, "
          f"bsw0={len(bsw0_leaves)}, bsw1={len(bsw1_leaves)}"
      )
      raw_bsw_0 = pps_treedef.unflatten(bsw0_leaves)
      raw_bsw_1 = pps_treedef.unflatten(bsw1_leaves)

      @jax.shard_map(
          mesh=self.mesh,
          in_specs=((bsw_pps, bsw_pps), P("stage")),
          out_specs=bsw_pps,
          check_vma=True,
      )
      def select_weights_from_bsw(bsw_inner, repeat_id):
        return jax.tree.map(
            lambda x, y: jax.lax.select(repeat_id[0] == stage0_repeat_id, y, x),
            bsw_inner[0],
            bsw_inner[1],
        )

      raw_weights = select_weights_from_bsw((raw_bsw_0, raw_bsw_1), repeat_ids)
      weights = bsw_treedef.unflatten(jax.tree.leaves(raw_weights))
    else:
      def select_weights_from_bsw(bsw_inner, repeat_id):
        return jax.tree.map(
            lambda x, y: jax.lax.select(repeat_id == stage0_repeat_id, y, x) if x is not None else None,
            bsw_inner[0],
            bsw_inner[1],
        )

      weights = jax.vmap(select_weights_from_bsw, in_axes=((0, 0), 0), out_axes=0)(bsw, repeat_ids)

    return weights

  def run_one_iteration(
      self,
      loop_state,
      bsw,
      pipeline_weights_graph,
      layers_metrics,
      current_layer_mutables,
      positions,
      segment_ids,
      deterministic,
      model_mode,
      logical_partition_spec,
  ):
    """Executes forward/backward for a single microbatch inside the circular pipeline."""
    state_io = loop_state["state_io"]
    shift = loop_state["shift"]
    circ_storage = loop_state["circ_storage"]
    loop_iteration = loop_state["loop_iteration"]

    microbatch_ids, _ = self.get_microbatch_and_repeat_ids(loop_iteration)
    physical_partition_spec = logical_to_mesh(logical_partition_spec, self.mesh, rules=self.config.logical_axis_rules)

    stages_inputs = self.get_iteration_inputs(loop_iteration, state_io, circ_storage, shift)
    stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, "iteration_input")

    stages_positions = self.gather_microbatch_inputs_vmap(positions, microbatch_ids, 0) if positions is not None else None
    stages_segment_ids = (
        self.gather_microbatch_inputs_vmap(segment_ids, microbatch_ids, 0) if segment_ids is not None else None
    )

    vmap_func = self.get_main_vmap_func_for_iterations()

    # 1. Fetch params from BSW (params-only, tree matches physical_partition_spec)
    stage_params = self.fetch_active_stage_weights(
        bsw,
        loop_iteration,
        physical_partition_spec=physical_partition_spec,
    )

    # 2. Gather non-params (metrics, mutables) for current repeat directly
    _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)
    if self.config.num_pipeline_repeats > 1:
      stage_metrics = self.gather_weights_across_stages_vmap(
          layers_metrics, repeat_ids=repeat_ids, repeat_dim_in_weights=0, stages_dim_in_weights=1
      )
      stage_mutables = self.gather_weights_across_stages_vmap(
          current_layer_mutables, repeat_ids=repeat_ids, repeat_dim_in_weights=0, stages_dim_in_weights=1
      )
    else:
      stage_metrics = self._stamp_at_current_trace(layers_metrics)
      stage_mutables = current_layer_mutables

    # 3. Merge into full state for forward pass
    stage_weights_state = nnx.State.merge(stage_params, stage_metrics, stage_mutables)

    stages_output, (updated_stage_metrics, updated_stage_mutables) = vmap_func(
        pipeline_weights_graph,
        stage_weights_state,
        stages_inputs,
        stages_segment_ids,
        stages_positions,
        deterministic,
        model_mode,
    )

    if self.config.scan_layers:
      stages_output = stages_output[0]

    # Scatter-back: only update mutables
    if self.config.num_pipeline_repeats > 1:

      def _scatter_update_mutables(fw, uw):
        if fw is None or uw is None:
          return fw

        def _update_one_stage(f_s, u_s, r_id):
          return jax.lax.dynamic_update_slice_in_dim(f_s, jnp.expand_dims(u_s, 0), r_id, axis=0)

        r_ids = self.shard_dim_by_stages(repeat_ids, 0, physical_partition_spec=None)
        updated_fw = jax.vmap(_update_one_stage, in_axes=(1, 0, 0), out_axes=1)(fw, uw, r_ids)
        return self.shard_dim_by_stages(updated_fw, 1, physical_partition_spec=None, is_stage_weight=False)

      new_mutables_state = jax.tree.map(_scatter_update_mutables, current_layer_mutables, updated_stage_mutables)
      new_layer_state = nnx.State.merge(updated_stage_metrics, new_mutables_state)
    else:
      new_layer_state = nnx.State.merge(updated_stage_metrics, updated_stage_mutables)

    new_state = self.get_new_loop_state(stages_output, loop_state)
    return new_state, new_layer_state

  def __call__(
      self,
      inputs: jnp.ndarray,
      segment_ids: jnp.ndarray,
      positions: jnp.ndarray,
      deterministic: bool,
      model_mode=MODEL_MODE_TRAIN,
      logical_partition_spec=None,
  ) -> jnp.ndarray:
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
      positions = self._maybe_shard_with_name(positions, ag_sharding).reshape(
          (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
      )
    if segment_ids is not None:
      segment_ids = self._maybe_shard_with_name(segment_ids, ag_sharding).reshape(
          (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
      )

    loop_state = self.init_states(inputs)

    if is_linen_initializing():
      return jnp.zeros(
          (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim),
          dtype=inputs.dtype,
      )

    # Two spec variants needed:
    # - Full spec (with circular_repeats axis) -> BSW creation
    # - Stripped logical spec (circular_repeats removed) -> BSW consumption
    physical_partition_spec_full = logical_to_mesh(
        logical_partition_spec, mesh=self.mesh, rules=self.config.logical_axis_rules
    )
    logical_partition_spec_stripped = pipeline_utils.strip_pipeline_repeat_logical_axis(logical_partition_spec)

    bubble_iterations = self.forwarding_delay * (self.num_stages - 1)

    layers_graph, layers_state = nnx.split(self.layers)

    def is_lp(x):
      return isinstance(x, nn.spmd.LogicallyPartitioned)

    def unbox_val(x):
      return x.value if is_lp(x) else x

    layers_state = jax.tree.map(unbox_val, layers_state, is_leaf=is_lp)

    _, layers_params, layers_metrics, layers_mutables = nnx.split(layers_state, _is_static_param, nnx.Intermediate, ...)

    assert all(
        isinstance(v, nnx.RngState)
        for v in jax.tree.leaves(layers_mutables, is_leaf=lambda x: isinstance(x, nnx.Variable))
        if isinstance(v, nnx.Variable)
    ), (
        "Non-RngState variable found in layers_mutables catch-all partition. "
        "Only RngState variables (RngKey/RngCount) should be present."
    )

    # ---- Nested scan structure ----
    num_microbatches = self.config.num_pipeline_microbatches

    def run_single_microbatch(carry, _unused_xs, buffer_sliding_window):
      loop_state, rng_mutables = carry
      rng_mutables = jax.lax.stop_gradient(rng_mutables)
      iteration = loop_state["loop_iteration"]
      advanced_rng_mutables = _advance_rng_state(rng_mutables, iteration)
      new_loop_state, new_layer_state = self.run_one_iteration(
          loop_state,
          buffer_sliding_window,
          layers_graph,
          layers_metrics,
          advanced_rng_mutables,
          positions,
          segment_ids,
          deterministic,
          model_mode,
          logical_partition_spec_stripped,
      )
      _, _, per_step_metrics, updated_rng_mutables = nnx.split(
          new_layer_state, _is_static_param, nnx.Intermediate, ...
      )
      return (new_loop_state, updated_rng_mutables), per_step_metrics

    # ---- V65n key change: carry sharded repeat weights across outer scan ----
    #
    # Instead of jax.lax.cond(repeat_index == 0, zeros, prefetch(prev)),
    # we carry the SHARDED repeat weights from the previous iteration.
    # For repeat 0: carried sharded weights are initialized as zeros
    # (same shape as a single-repeat slice of layers_params, sharded).
    # For repeat N>0: carried sharded weights = the "next" slice from
    # the previous repeat iteration, already at the correct trace level.
    #
    # This eliminates the cond and keeps carry size small (sharded).

    # Create initial sharded zeros matching a single-repeat slice shape.
    # layers_params has shape [num_repeats, num_stages, ...] per leaf.
    # A single-repeat slice has shape [num_stages, ...].
    def _make_sharded_zeros(x):
      """Create zeros with shape of a single-repeat slice (repeat dim removed)."""
      if hasattr(x, 'value'):
        arr = x.value
      else:
        arr = x
      # arr shape: [num_repeats, num_stages, ...] -> slice shape: [num_stages, ...]
      if self.config.num_pipeline_repeats > 1:
        slice_shape = arr.shape[1:]
      else:
        slice_shape = arr.shape
      return jnp.zeros(slice_shape, dtype=arr.dtype)

    initial_sharded_prev = jax.tree.map(_make_sharded_zeros, layers_params)

    def execute_one_repeat(carry, repeat_index):
      loop_state, rng_mutables, sharded_prev_weights = carry

      total_iteration_at_repeat_start = repeat_index * num_microbatches

      # --- V65n: all-gather carried sharded weights to produce w_curr ---
      # For repeat 0: sharded_prev_weights is zeros -> w_curr is zeros (gathered)
      # For repeat N>0: sharded_prev_weights is the sharded slice from previous
      #   repeat's "next" -> all-gather produces the correct w_curr.
      #
      # No jax.lax.cond needed. The zeros case produces gathered zeros, which
      # get_current_weights_from_bsw handles correctly (BSW select will pick
      # w_next for repeat 0 anyway since stage0_repeat_id == 0).
      gathered_weights_current = self.from_repeat_weights_to_bsw(
          sharded_prev_weights, physical_partition_spec_full
      )

      # --- Slice next repeat's sharded weights (dynamic_slice, no all-gather yet) ---
      sharded_next_weights = self.from_all_variables_to_repeat_weights(
          layers_params, total_iteration_at_repeat_start
      )

      # --- All-gather next repeat's weights ---
      gathered_weights_next = self.from_repeat_weights_to_bsw(
          sharded_next_weights, physical_partition_spec_full
      )

      buffer_sliding_window = (gathered_weights_current, gathered_weights_next)

      def run_microbatch_scan(loop_state_in, rng_mutables_in, bsw):
        def microbatch_step(microbatch_carry, _unused):
          return run_single_microbatch(microbatch_carry, None, bsw)

        if self.config.set_remat_policy_on_pipeline_iterations:
          microbatch_step = jax.checkpoint(
              microbatch_step,
              policy=self.get_pipeline_remat_policy(),
              prevent_cse=not self.config.scan_pipeline_iterations,
          )

        if self.config.scan_pipeline_iterations:
          (final_loop_state, final_rng_mutables), microbatch_metrics = jax.lax.scan(
              microbatch_step, (loop_state_in, rng_mutables_in), None, length=num_microbatches
          )
        else:
          scan_carry = (loop_state_in, rng_mutables_in)
          microbatch_metrics_list = []
          for _ in range(num_microbatches):
            scan_carry, step_metrics = microbatch_step(scan_carry, None)
            microbatch_metrics_list.append(step_metrics)
          final_loop_state, final_rng_mutables = scan_carry
          microbatch_metrics = (
              jax.tree.map(lambda *xs: jnp.stack(xs), *microbatch_metrics_list)
              if microbatch_metrics_list
              else layers_metrics
          )
        return (final_loop_state, final_rng_mutables), microbatch_metrics

      loop_state = {**loop_state, "loop_iteration": total_iteration_at_repeat_start}
      (new_loop_state, updated_rng_mutables), repeat_step_metrics = run_microbatch_scan(
          loop_state, rng_mutables, buffer_sliding_window
      )

      # --- V65n: carry sharded_next_weights forward (small, sharded) ---
      return (new_loop_state, updated_rng_mutables, sharded_next_weights), repeat_step_metrics

    if self.config.set_remat_policy_on_pipeline_iterations:
      checkpointed_repeat_body = jax.checkpoint(
          execute_one_repeat, policy=self.get_pipeline_remat_policy()
      )
    else:
      checkpointed_repeat_body = execute_one_repeat

    # ---- Execute: outer scan over repeats ----
    num_repeats = self.config.num_pipeline_repeats

    (loop_state, final_rng_mutables, _final_sharded_carry), repeat_metrics = jax.lax.scan(
        checkpointed_repeat_body,
        (loop_state, layers_mutables, initial_sharded_prev),
        jnp.arange(num_repeats),
        length=num_repeats,
    )

    # ---- Bubble iterations (pipeline drain) ----
    if bubble_iterations > 0:
      total_iteration_after_repeats = num_repeats * num_microbatches
      last_repeat_weights = self.weight_prefetching(
          layers_params, physical_partition_spec_full, total_iteration_after_repeats - 1
      )
      bubble_bsw = (last_repeat_weights, last_repeat_weights)

      def run_bubble_microbatch(bubble_carry, _unused):
        return run_single_microbatch(bubble_carry, None, bubble_bsw)

      if self.config.set_remat_policy_on_pipeline_iterations:
        checkpointed_bubble_step = jax.checkpoint(
            run_bubble_microbatch,
            policy=self.get_pipeline_remat_policy(),
            prevent_cse=not self.config.scan_pipeline_iterations,
        )
      else:
        checkpointed_bubble_step = run_bubble_microbatch

      loop_state = {**loop_state, "loop_iteration": jnp.int32(total_iteration_after_repeats)}
      if self.config.scan_pipeline_iterations:
        (loop_state, final_rng_mutables), bubble_metrics = jax.lax.scan(
            checkpointed_bubble_step, (loop_state, final_rng_mutables), None, length=bubble_iterations
        )
      else:
        bubble_carry = (loop_state, final_rng_mutables)
        bubble_metrics_list = []
        for _ in range(bubble_iterations):
          bubble_carry, bubble_step_metrics = checkpointed_bubble_step(bubble_carry, None)
          bubble_metrics_list.append(bubble_step_metrics)
        loop_state, final_rng_mutables = bubble_carry
        bubble_metrics = (
            jax.tree.map(lambda *xs: jnp.stack(xs), *bubble_metrics_list)
            if bubble_metrics_list
            else layers_metrics
        )

      stacked_metrics = jax.tree.map(
          lambda repeat_m, bubble_m: jnp.concatenate([repeat_m, bubble_m], axis=0),
          repeat_metrics, bubble_metrics,
      )
    else:
      stacked_metrics = repeat_metrics

    final_layer_state = nnx.State.merge(layers_params, stacked_metrics, final_rng_mutables)
    nnx.update(self.layers, final_layer_state)

    final_output = self.realign_output_microbatches(loop_state["state_io"])
    return jnp.reshape(
        final_output,
        (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim),
        out_sharding=self.output_sharding,
    )


def create_nnx_pipeline(
    config: Config, stage_factory: Any, mesh: Mesh, remat_policy: Any = None, *, rngs: nnx.Rngs
) -> NNXPipeline | NNXCircularPipelineShardedCarry:
  """Factory function to instantiate the NNX Pipeline module."""
  if config.pipeline_fsdp_ag_per_repeat:
    return NNXCircularPipelineShardedCarry(
        config=config, stage_factory=stage_factory, mesh=mesh, remat_policy=remat_policy, rngs=rngs
    )
  return NNXPipeline(config=config, stage_factory=stage_factory, mesh=mesh, remat_policy=remat_policy, rngs=rngs)


Pipeline = to_linen_class(
    NNXPipeline,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
CircularPipeline = to_linen_class(
    NNXCircularPipelineShardedCarry,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


def create_pipeline(
    config: Config,
    layers=None,
    mesh: Mesh = None,
    remat_policy: Any = None,
) -> nn.Module:
  """Returns the ToLinen-wrapped NNX pipeline appropriate for the config.

  For raw NNX pipeline classes (no Linen wrapping), use create_nnx_pipeline() instead.

  Args:
    config: Model configuration.
    layers: Callable[[nnx.Rngs], nnx.Module] constructing one pipeline stage.
    mesh: JAX device mesh for sharding.
    remat_policy: Optional rematerialization policy.
  """
  cls = CircularPipeline if config.pipeline_fsdp_ag_per_repeat else Pipeline
  return cls(config=config, stage_factory=layers, mesh=mesh, remat_policy=remat_policy)
