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

"""Pipeline V57: NNXCircularPipeline with inline Linen nn.scan(nn.remat) adapter.

This variant replaces the inner jax.lax.scan + jax.checkpoint in
NNXCircularPipeline.__call__ with a Linen nn.scan(nn.remat(...)) wrapper.
The Linen scan is invoked via .apply() on state extracted from NNX variables,
getting the same XLA scan + remat behavior that the original Linen pipeline
achieves -- which is known to produce better memory profiles than the
jax.lax.scan + jax.checkpoint combination in the pure NNX path.

Key design:
  - A temporary nn.Module (_LinenInnerScanBody) wraps the NNX inner_body logic.
  - nn.scan(nn.remat(_LinenInnerScanBody)) gives us Linen's scan lifting with
    proper variable_broadcast for params (no per-iteration stacking).
  - NNX state is converted to Linen variable dicts before .apply(), and
    converted back afterward.
  - Everything is self-contained inside NNXCircularPipelineV57.__call__.
"""

from typing import Any, Callable

import numpy as np

from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import jax
import jax.ad_checkpoint

from aqt.jax.v2 import aqt_tensor
from flax.core import meta
from flax import linen as nn
from flax.linen.spmd import LogicallyPartitioned
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

# Re-export everything from pipeline.py that is NOT overridden
from maxtext.layers.pipeline import (
    _is_static_param,
    _advance_rng_state,
    is_spec_leaf,
    NNXPipelineBase,
    NNXPipeline,
    Pipeline,
    create_nnx_pipeline as _orig_create_nnx_pipeline,
    create_pipeline as _orig_create_pipeline,
)


class NNXCircularPipelineV57(NNXPipelineBase):
  """V57: NNXCircularPipeline with inline Linen nn.scan(nn.remat) adapter.

  Identical to NNXCircularPipeline for all methods EXCEPT __call__, where
  the inner scan loop uses nn.scan(nn.remat(...)) via a temporary Linen
  module instead of jax.lax.scan + jax.checkpoint.

  This gives the same XLA scan + remat behavior as the original Linen
  CircularPipeline, which is known to produce superior memory profiles
  because nn.scan handles variable_broadcast (params not stacked in carry)
  and nn.remat uses Flax's checkpoint policy integration natively.
  """

  # ---- All helper methods copied from NNXCircularPipeline ----

  def get_main_vmap_func_for_iterations(self):
    """Override: vmap returns only non-param state to avoid stacking params across stages."""
    def func_to_vmap(graph, state, stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode):
      module = nnx.merge(graph, state)
      out = module(stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode)
      _, _, updated_metrics, updated_mutables = nnx.split(
          module, _is_static_param, nnx.Intermediate, ...
      )
      return out, (updated_metrics, updated_mutables)

    vmap_kwargs = dict(
        in_axes=(None, 0, 0, 0, 0, None, None),
        out_axes=(0, (0, 0)),
    )
    if self.spmd_axis_name is not None:
      vmap_kwargs["spmd_axis_name"] = self.spmd_axis_name
    return jax.vmap(func_to_vmap, **vmap_kwargs)

  def gather_microbatch_inputs_vmap(self, xs, ids, ids_dim):
    """Slices out the specific sequence inputs for the current microbatch."""
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
    """Executes the forward/backward logic for a single microbatch inside the circular pipeline."""
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

  # =====================================================================
  # __call__: The V57 variant with inline Linen nn.scan(nn.remat) adapter
  # =====================================================================

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

    # ---- V57: Inline Linen nn.scan(nn.remat) adapter ----
    #
    # Instead of using jax.lax.scan + jax.checkpoint for the inner microbatch
    # loop, we create a temporary Linen nn.Module that wraps the inner body,
    # then apply nn.scan(nn.remat(...)) to it. This gives us:
    #   1. variable_broadcast for params (not stacked in carry)
    #   2. Linen's native remat integration
    #   3. split_rngs for proper dropout mask variation
    #
    # The conversion flow:
    #   NNX state -> strip nnx.Variable wrappers -> Linen variable dicts
    #   -> nn.scan(nn.remat(body)).apply(variables, carry)
    #   -> extract results -> re-wrap into NNX state

    bsw_ref = [None]
    num_microbatches = self.config.num_pipeline_microbatches
    pipeline_self = self  # Capture for use inside Linen module

    # -- Convert NNX state trees to flat arrays for Linen variable dicts --

    def _nnx_state_to_raw_arrays(state):
      """Strip nnx.Variable wrappers, returning raw JAX arrays."""
      return jax.tree.map(
          lambda x: x.value if isinstance(x, nnx.Variable) else x,
          state,
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )

    def _get_treedef(state):
      """Get the treedef of an nnx.State, treating Variables as leaves."""
      return jax.tree.structure(state, is_leaf=lambda x: isinstance(x, nnx.Variable))

    # Capture treedefs for reconstruction after Linen .apply()
    mutables_treedef = _get_treedef(layers_mutables)
    metrics_treedef = _get_treedef(layers_metrics)

    # ---- Build the Linen inner scan body module ----

    class _LinenInnerScanBody(nn.Module):
      """Temporary Linen module wrapping a single inner microbatch iteration.

      This module is designed to be used with nn.scan(nn.remat(...)) to get
      proper variable_broadcast behavior for params and remat for activations.

      The carry is a tuple of:
        - loop_state dict (JAX arrays)
        - mutables_flat: flat list of raw arrays from layers_mutables

      The module accesses the BSW params through bsw_ref (closure), which is
      set by the outer loop before each repeat. This avoids putting the large
      BSW in the scan carry.
      """

      @nn.compact
      def __call__(self, carry):
        current_loop_state, mutables_flat = carry

        # Reconstruct NNX mutables from flat arrays
        current_layer_mutables = mutables_treedef.unflatten(mutables_flat)
        current_layer_mutables = jax.lax.stop_gradient(current_layer_mutables)

        iteration = current_loop_state["loop_iteration"]
        advanced_mutables = _advance_rng_state(current_layer_mutables, iteration)

        new_loop_state, new_layer_state = pipeline_self.run_one_iteration(
            current_loop_state, bsw_ref[0], layers_graph, layers_metrics,
            advanced_mutables, positions, segment_ids, deterministic,
            model_mode, logical_partition_spec_stripped,
        )

        _, _, new_layer_metrics, new_layer_mutables = nnx.split(
            new_layer_state, _is_static_param, nnx.Intermediate, ...
        )

        # Flatten mutables back to raw arrays for carry
        new_mutables_flat = jax.tree.leaves(
            _nnx_state_to_raw_arrays(new_layer_mutables)
        )

        # Flatten metrics for scan output (will be stacked along axis 0)
        new_metrics_flat = jax.tree.leaves(
            _nnx_state_to_raw_arrays(new_layer_metrics)
        )

        new_carry = (new_loop_state, new_mutables_flat)
        return new_carry, new_metrics_flat

    # Build the scanned + remat'd module
    remat_policy = self.get_pipeline_remat_policy()

    # nn.scan wraps nn.remat for the inner microbatch loop.
    # - No variable collections are used (we pass everything through carry/scan outputs).
    # - split_rngs is empty since RNG advancement is handled manually via _advance_rng_state.
    # - The module has no Linen parameters; all state flows through carry.
    ScannedInnerBody = nn.scan(
        nn.remat(
            _LinenInnerScanBody,
            policy=remat_policy,
        ) if self.config.set_remat_policy_on_pipeline_iterations else _LinenInnerScanBody,
        variable_axes={},
        variable_broadcast=[],
        variable_carry=[],
        split_rngs={},
        length=num_microbatches,
    )

    def _run_inner_scan(current_loop_state, current_layer_mutables):
      """Execute the inner microbatch scan using the Linen adapter."""
      # Flatten mutables to raw arrays for carry
      mutables_flat = jax.tree.leaves(_nnx_state_to_raw_arrays(current_layer_mutables))

      carry_in = (current_loop_state, mutables_flat)

      # Instantiate and apply the scanned Linen module
      # No variables needed -- everything flows through carry
      scanned_module = ScannedInnerBody()
      result = scanned_module.apply(
          {},  # No Linen variables
          carry_in,
      )

      new_carry, stacked_metrics_flat = result
      new_loop_state, new_mutables_flat = new_carry

      # Reconstruct NNX state from flattened outputs
      new_layer_mutables = mutables_treedef.unflatten(new_mutables_flat)
      # Metrics are stacked along axis 0 by nn.scan
      # stacked_metrics_flat is a list of arrays, each with leading dim = num_microbatches
      inner_metrics = metrics_treedef.unflatten(stacked_metrics_flat)

      return new_loop_state, new_layer_mutables, inner_metrics

    def _run_inner_unrolled(current_loop_state, current_layer_mutables):
      """Fallback: unrolled inner loop (no scan) for debugging."""
      inner_carry = (current_loop_state, current_layer_mutables)
      inner_metrics_list = []

      def _one_step(carry, _unused):
        ls, muts = carry
        muts_stopped = jax.lax.stop_gradient(muts)
        iteration = ls["loop_iteration"]
        advanced = _advance_rng_state(muts_stopped, iteration)
        new_ls, new_layer_state = self.run_one_iteration(
            ls, bsw_ref[0], layers_graph, layers_metrics,
            advanced, positions, segment_ids, deterministic,
            model_mode, logical_partition_spec_stripped,
        )
        _, _, new_metrics, new_muts = nnx.split(
            new_layer_state, _is_static_param, nnx.Intermediate, ...
        )
        return (new_ls, new_muts), new_metrics

      for _ in range(num_microbatches):
        inner_carry, step_metrics = _one_step(inner_carry, None)
        inner_metrics_list.append(step_metrics)

      new_loop_state, new_layer_mutables = inner_carry
      inner_metrics = (
          jax.tree.map(lambda *xs: jnp.stack(xs), *inner_metrics_list)
          if inner_metrics_list else layers_metrics
      )
      return new_loop_state, new_layer_mutables, inner_metrics

    # ---- Outer scan (repeats) + bubble scan ----

    def outer_body(carry, _):
      """One repeat: prefetch w_next (1 all-gather), dual BSW, then run inner scan."""
      current_loop_state, current_layer_mutables, w_curr = carry
      iteration = current_loop_state["loop_iteration"]

      w_next = self.weight_prefetching(
          layers_params, physical_partition_spec_full, iteration
      )
      bsw_ref[0] = (w_curr, w_next)

      if self.config.scan_pipeline_iterations:
        new_loop_state, new_layer_mutables, inner_metrics = _run_inner_scan(
            current_loop_state, current_layer_mutables
        )
      else:
        new_loop_state, new_layer_mutables, inner_metrics = _run_inner_unrolled(
            current_loop_state, current_layer_mutables
        )

      return (new_loop_state, new_layer_mutables, w_next), inner_metrics

    num_repeats = self.config.num_pipeline_repeats
    initial_w_curr = jax.tree.map(lambda x: jnp.zeros(x.shape[1:], dtype=x.dtype), layers_params)

    if self.config.scan_pipeline_iterations:
      unroll_repeats = num_repeats if not self.config.scan_pipeline_repeats else 1
      (loop_state, final_layer_mutables, final_w_curr), repeat_metrics = jax.lax.scan(
          outer_body, (loop_state, layers_mutables, initial_w_curr), None,
          length=num_repeats, unroll=unroll_repeats,
      )
      repeat_metrics = jax.tree.map(
          lambda x: x.reshape((num_repeats * num_microbatches,) + x.shape[2:]),
          repeat_metrics,
      )
    else:
      outer_carry = (loop_state, layers_mutables, initial_w_curr)
      repeat_metrics_list = []
      for _ in range(num_repeats):
        outer_carry, rep_metrics = outer_body(outer_carry, None)
        repeat_metrics_list.append(rep_metrics)
      loop_state, final_layer_mutables, final_w_curr = outer_carry
      repeat_metrics = (
          jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *repeat_metrics_list)
          if repeat_metrics_list else layers_metrics
      )

    # ---- Bubble iterations (pipeline drain) ----
    if bubble_iterations > 0:
      bsw_ref[0] = (final_w_curr, final_w_curr)

      if self.config.scan_pipeline_iterations:
        # For bubble, use a separate scan with the same Linen adapter
        # but with bubble_iterations length
        BubbleScannedBody = nn.scan(
            nn.remat(
                _LinenInnerScanBody,
                policy=remat_policy,
            ) if self.config.set_remat_policy_on_pipeline_iterations else _LinenInnerScanBody,
            variable_axes={},
            variable_broadcast=[],
            variable_carry=[],
            split_rngs={},
            length=bubble_iterations,
        )

        mutables_flat = jax.tree.leaves(_nnx_state_to_raw_arrays(final_layer_mutables))
        carry_in = (loop_state, mutables_flat)

        bubble_module = BubbleScannedBody()
        result = bubble_module.apply({}, carry_in)

        (loop_state, new_mutables_flat), bubble_metrics_flat = result
        final_layer_mutables = mutables_treedef.unflatten(new_mutables_flat)
        bubble_metrics = metrics_treedef.unflatten(bubble_metrics_flat)
      else:
        bubble_carry = (loop_state, final_layer_mutables)
        bubble_metrics_list = []

        def _bubble_step(carry, _unused):
          ls, muts = carry
          muts_stopped = jax.lax.stop_gradient(muts)
          iteration = ls["loop_iteration"]
          advanced = _advance_rng_state(muts_stopped, iteration)
          new_ls, new_layer_state = self.run_one_iteration(
              ls, bsw_ref[0], layers_graph, layers_metrics,
              advanced, positions, segment_ids, deterministic,
              model_mode, logical_partition_spec_stripped,
          )
          _, _, new_metrics, new_muts = nnx.split(
              new_layer_state, _is_static_param, nnx.Intermediate, ...
          )
          return (new_ls, new_muts), new_metrics

        for _ in range(bubble_iterations):
          bubble_carry, bub_metrics = _bubble_step(bubble_carry, None)
          bubble_metrics_list.append(bub_metrics)
        loop_state, final_layer_mutables = bubble_carry
        bubble_metrics = (
            jax.tree.map(lambda *xs: jnp.stack(xs), *bubble_metrics_list)
            if bubble_metrics_list else layers_metrics
        )

      stacked_metrics = jax.tree.map(
          lambda r, b: jnp.concatenate([r, b], axis=0), repeat_metrics, bubble_metrics
      )
    else:
      stacked_metrics = repeat_metrics

    final_layer_state = nnx.State.merge(layers_params, stacked_metrics, final_layer_mutables)
    nnx.update(self.layers, final_layer_state)

    final_output = self.realign_output_microbatches(loop_state["state_io"])
    return jnp.reshape(
        final_output,
        (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim),
        out_sharding=self.output_sharding,
    )


# ---- V57 wiring: factory + Linen wrappers ----

CircularPipelineV57 = to_linen_class(
    NNXCircularPipelineV57,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


def create_nnx_pipeline_v57(
    config: Config, stage_factory: Any, mesh: Mesh, remat_policy: Any = None, *, rngs: nnx.Rngs
) -> NNXPipeline | NNXCircularPipelineV57:
  """Factory function to instantiate the V57 NNX Pipeline module."""
  if config.pipeline_fsdp_ag_per_repeat:
    return NNXCircularPipelineV57(
        config=config, stage_factory=stage_factory, mesh=mesh, remat_policy=remat_policy, rngs=rngs
    )
  return NNXPipeline(config=config, stage_factory=stage_factory, mesh=mesh, remat_policy=remat_policy, rngs=rngs)


def create_pipeline_v57(
    config: Config,
    layers=None,
    mesh: Mesh = None,
    remat_policy: Any = None,
) -> nn.Module:
  """Returns the ToLinen-wrapped V57 pipeline appropriate for the config.

  Args:
    config: Model configuration.
    layers: Callable[[nnx.Rngs], nnx.Module] constructing one pipeline stage.
    mesh: JAX device mesh for sharding.
    remat_policy: Optional rematerialization policy.
  """
  cls = CircularPipelineV57 if config.pipeline_fsdp_ag_per_repeat else Pipeline
  return cls(config=config, stage_factory=layers, mesh=mesh, remat_policy=remat_policy)
