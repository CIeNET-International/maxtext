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

"""Pipeline layer wrapping a decoder layer(s). Supports circular pipelining.

V65e: V65 + identity pass-through variable_axes collections to match V46 PE split.

Hypothesis
----------
V46 (Linen) has ``variable_axes={"summaries": 0, "aux_loss": 0, "intermediates": 0,
"hyper_params": 0}`` and ``variable_broadcast=["_overwrite_with_gradient", "non_trainable"]``.
These collections produce KNOWN outputs in partial eval even when carry outputs
are UNKNOWN (from custom_vjp). The known/unknown split triggers
``_scan_partial_eval_custom`` to create 2 scan groups -> 2 recomp pairs.

V65 has ``variable_axes={}`` -- no KNOWN outputs -> 1 scan group -> 1 recomp pair
-> higher memory.

Fix: add 4 DUMMY variable_axes collections that are NEVER WRITTEN inside the
body. They are identity pass-throughs: initialized before the scan, passed
through each repeat iteration unchanged (scope never calls put_variable for
them). lift.scan stacks their outputs along axis 0, producing KNOWN output
tensors in PE.

Additionally, add ``_overwrite_with_gradient`` and ``non_trainable`` to
variable_broadcast to match V46's broadcast list.

The body function does NOT touch these collections -- they pass through unchanged.
Previous V65 attempt with ``scan_metrics`` FAILED because we WROTE to the
collection (making it UNKNOWN). This time: pure identity pass-through.
"""

import functools
from typing import Any

import numpy as np

from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import jax
import jax.ad_checkpoint

from aqt.jax.v2 import aqt_tensor
from flax.core import lift as flax_lift
from flax.core import scope as flax_scope
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

# Reuse V65 utilities and base classes
from maxtext.layers.pipeline_v65 import (
    _is_static_param,
    _advance_rng_state,
    is_spec_leaf,
    _flatten_nnx_state,
    _unflatten_nnx_state,
    _arrays_to_linen_collection,
    _linen_collection_to_arrays,
    _PipelineContext,
)

from maxtext.layers.pipeline import (
    NNXPipelineBase,
    NNXPipeline,
)


class NNXCircularPipeline(NNXPipelineBase):
  """V65e: V65 + identity pass-through variable_axes to match V46 PE split.

  Identical to V65's NNXCircularPipeline except:
    1. variable_axes={'_scan_out_0': 0, '_scan_out_1': 0,
                      '_scan_out_2': 0, '_scan_out_3': 0}
       -- 4 dummy collections matching V46's 4 variable_axes collections
    2. variable_broadcast adds '_overwrite_with_gradient' and 'non_trainable'
       -- matching V46's broadcast list
    3. linen_variables includes initial values for all dummy collections
    4. apply_fn mutable list includes all dummy collections
    5. Body function does NOT write to dummy collections (identity pass-through)

  Memory target: match V46's 23.3 GB via PE split from known outputs.
  """

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
    """Slices out specific sequence inputs for the current microbatch."""
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
    """Fetches actively prefetched weights from BSW."""
    return self.get_current_weights_from_bsw(bsw, loop_iteration, physical_partition_spec)

  def get_current_weights_from_bsw(self, bsw, loop_iteration, physical_partition_spec):
    """Pulls the fully gathered parameters for the current repeat from BSW dual-buffer."""
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

    # 1. Fetch params from BSW
    stage_params = self.fetch_active_stage_weights(
        bsw,
        loop_iteration,
        physical_partition_spec=physical_partition_spec,
    )

    # 2. Gather non-params for current repeat
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

    # Scatter-back
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
    """V65e: V65 + identity pass-through variable_axes for PE split.

    Architecture:
      lift.scan(REPEATS) -> lift.checkpoint(stage_fn containing jax.lax.scan(MICROBATCHES))

    Key difference from V65:
      - 4 dummy variable_axes collections: _scan_out_0..3
        Initialized with zeros(num_repeats), NEVER written inside body.
        Creates KNOWN outputs in partial eval -> 2 scan groups -> 2 recomp pairs.
      - variable_broadcast adds '_overwrite_with_gradient' and 'non_trainable'
        to match V46's broadcast list.
    """
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
    num_microbatches = self.config.num_pipeline_microbatches
    num_repeats = self.config.num_pipeline_repeats
    remat_policy = self.get_pipeline_remat_policy()

    layers_graph, layers_state = nnx.split(self.layers)

    def is_lp(x):
      return isinstance(x, nn.spmd.LogicallyPartitioned)

    def unbox_val(x):
      return x.value if is_lp(x) else x

    layers_state = jax.tree.map(unbox_val, layers_state, is_leaf=is_lp)

    _, layers_params, layers_metrics, layers_mutables = nnx.split(
        layers_state, _is_static_param, nnx.Intermediate, ...
    )

    # Validate: layers_mutables should contain ONLY RngState variables
    assert all(
        isinstance(v, nnx.RngState)
        for v in jax.tree.leaves(layers_mutables, is_leaf=lambda x: isinstance(x, nnx.Variable))
        if isinstance(v, nnx.Variable)
    ), (
        "Non-RngState variable found in layers_mutables catch-all partition. "
        "Only RngState variables (RngKey/RngCount) should be present."
    )

    scan_pipeline_iters = self.config.scan_pipeline_iterations

    # ---- Flatten all NNX state into indexed arrays for Linen scope ----

    # Params -> 'params' collection (broadcast)
    param_arrays, param_treedef, param_is_var, param_var_types, param_meta = _flatten_nnx_state(layers_params)
    param_keys = [f'p{i}' for i in range(len(param_arrays))]

    # Mutables -> 'carry_state' collection (carry)
    mut_arrays, mut_treedef, mut_is_var, mut_var_types, mut_meta = _flatten_nnx_state(layers_mutables)
    mut_keys = [f'm{i}' for i in range(len(mut_arrays))]

    # Metrics -> 'metrics_template' collection (broadcast)
    met_arrays, met_treedef, met_is_var, met_var_types, met_meta = _flatten_nnx_state(layers_metrics)
    met_keys = [f'k{i}' for i in range(len(met_arrays))]

    # Input arrays -> 'broadcast_inputs' collection (broadcast)
    input_arrays = []
    input_keys = []
    if positions is not None:
      input_arrays.append(positions)
      input_keys.append('positions')
    if segment_ids is not None:
      input_arrays.append(segment_ids)
      input_keys.append('segment_ids')

    # ---- V65e: Define dummy variable_axes collection names ----
    # These match V46's 4 variable_axes collections (summaries, aux_loss,
    # intermediates, hyper_params). We use generic names since the actual
    # collection names don't matter -- what matters is having 4 collections
    # with axis=0 that produce KNOWN outputs in partial eval.
    DUMMY_AXES_COLLECTIONS = ['_scan_out_0', '_scan_out_1', '_scan_out_2', '_scan_out_3']
    # Each dummy collection contains a single scalar per repeat.
    # Shape in linen_variables: (num_repeats,) -- lift.scan indexes axis 0.
    # After indexing: scalar () -- identity pass-through per repeat.
    DUMMY_KEY = '_dummy'

    # ---- Non-pytree context ----
    ctx = _PipelineContext(self)

    # ---- BSW mutable ref ----
    bsw_ref = [None]

    # ---- Stage function for lift.scan + lift.checkpoint ----
    #
    # CLOSURE AUDIT (V65e):
    #   Same as V65 plus:
    #   - DUMMY_AXES_COLLECTIONS: list of strings. SAFE.
    #   - DUMMY_KEY: string. SAFE.
    #   Body does NOT call scope.put_variable for any DUMMY collection.

    def _stage_fn_for_scope(scope, carry):
      """One repeat: prefetch w_next, run microbatch scan, return updated carry.

      All JAX arrays read from scope. No NNX module accessed.
      CRITICAL: dummy variable_axes collections are NOT written (identity pass-through).
      """
      loop_state_carry, w_curr = carry

      # ---- Read ALL JAX arrays from scope ----

      # Read params from scope (broadcast)
      local_param_arrays = [scope.get_variable('params', k) for k in param_keys]
      local_layers_params = _unflatten_nnx_state(
          local_param_arrays, param_treedef, param_is_var, param_var_types, param_meta
      )

      # Read broadcast inputs from scope
      local_positions = scope.get_variable('broadcast_inputs', 'positions') if 'positions' in input_keys else None
      local_segment_ids = scope.get_variable('broadcast_inputs', 'segment_ids') if 'segment_ids' in input_keys else None

      # Read metrics template from scope
      local_met_arrays = [scope.get_variable('metrics_template', k) for k in met_keys]
      local_layers_metrics = _unflatten_nnx_state(
          local_met_arrays, met_treedef, met_is_var, met_var_types, met_meta
      )

      # Read mutables from scope (carry)
      cur_mut_arrays = [scope.get_variable('carry_state', k) for k in mut_keys]
      current_layer_mutables = _unflatten_nnx_state(
          cur_mut_arrays, mut_treedef, mut_is_var, mut_var_types, mut_meta
      )

      # NOTE: We deliberately do NOT read or write the dummy collections
      # (_scan_out_0..3). They exist in scope as variable_axes collections,
      # and lift.scan handles their stacking automatically. Since the body
      # never calls scope.put_variable for them, they pass through unchanged
      # (identity). This produces KNOWN outputs in partial eval.

      # ---- Level 1: Per-microbatch custom_vjp ----

      @jax.custom_vjp
      def _run_iter_local(loop_state_arg, bsw_arg, mutables_arg, met_arg, pos_arg, seg_arg):
        return _run_iter_local_fwd(loop_state_arg, bsw_arg, mutables_arg, met_arg, pos_arg, seg_arg)[0]

      def _run_iter_local_fwd(loop_state_arg, bsw_arg, mutables_arg, met_arg, pos_arg, seg_arg):
        def _run(ls, b):
          return ctx.run_one_iteration(
              ls, b, layers_graph, met_arg,
              mutables_arg, pos_arg, seg_arg,
              deterministic, model_mode, logical_partition_spec_stripped,
          )
        _run_remat = jax.remat(_run, policy=remat_policy)
        (new_ls, new_layer_state), vjp_fn = jax.vjp(_run_remat, loop_state_arg, bsw_arg)
        return (new_ls, new_layer_state), vjp_fn

      def _run_iter_local_bwd(vjp_fn, g_out):
        g_ls, g_layer_state = g_out
        d_ls, d_bsw = vjp_fn((g_ls, g_layer_state))
        return d_ls, d_bsw, None, None, None, None

      _run_iter_local.defvjp(_run_iter_local_fwd, _run_iter_local_bwd)

      # ---- Level 2: Microbatch scan custom_vjp ----

      @jax.custom_vjp
      def _run_microbatches(loop_state_arg, bsw_arg, muts_arg,
                            met_arg, pos_arg, seg_arg):
        return _run_microbatches_fwd(loop_state_arg, bsw_arg, muts_arg,
                                     met_arg, pos_arg, seg_arg)[0]

      def _run_microbatches_fwd(loop_state_arg, bsw_arg, muts_arg,
                                 met_arg, pos_arg, seg_arg):
        def inner_body(inner_carry, _):
          ls, muts = inner_carry
          muts = jax.lax.stop_gradient(muts)
          iteration_inner = ls["loop_iteration"]
          advanced_muts = _advance_rng_state(muts, iteration_inner)
          new_ls, new_layer_state = _run_iter_local(
              ls, bsw_arg, advanced_muts,
              met_arg, pos_arg, seg_arg
          )
          _, _, new_metrics, new_muts = nnx.split(
              new_layer_state, _is_static_param, nnx.Intermediate, ...
          )
          new_met_arrays_inner, _, _, _, _ = _flatten_nnx_state(new_metrics)
          return (new_ls, new_muts), _arrays_to_linen_collection(new_met_arrays_inner, met_keys)

        def scan_fn(ls_arg, b_arg):
          if scan_pipeline_iters:
            (final_ls, final_muts), metrics = jax.lax.scan(
                inner_body, (ls_arg, muts_arg),
                None, length=num_microbatches,
            )
          else:
            carry = (ls_arg, muts_arg)
            metrics_list = []
            for _ in range(num_microbatches):
              carry, step_met = inner_body(carry, None)
              metrics_list.append(step_met)
            final_ls, final_muts = carry
            met_arrays_local, _, _, _, _ = _flatten_nnx_state(met_arg)
            fallback_met = _arrays_to_linen_collection(met_arrays_local, met_keys)
            metrics = (
                jax.tree.map(lambda *xs: jnp.stack(xs), *metrics_list)
                if metrics_list else fallback_met
            )
          return (final_ls, final_muts), metrics

        scan_output, scan_vjp_fn = jax.vjp(
            scan_fn, loop_state_arg, bsw_arg,
        )
        (final_ls, final_muts), metrics = scan_output

        return ((final_ls, final_muts, bsw_arg), metrics), scan_vjp_fn

      def _run_microbatches_bwd(scan_vjp_fn, g_out):
        (g_ls_muts_bsw, _g_metrics) = g_out
        g_ls, g_muts, g_bsw = g_ls_muts_bsw
        d_ls, d_bsw = scan_vjp_fn(((g_ls, g_muts), _g_metrics))
        d_bsw = jax.tree.map(
            lambda d, g: d + g if hasattr(d, "shape") else d, d_bsw, g_bsw
        )
        return d_ls, d_bsw, None, None, None, None

      _run_microbatches.defvjp(_run_microbatches_fwd, _run_microbatches_bwd)

      # ---- Level 3: Weight prefetch + microbatch scan custom_vjp ----

      @jax.custom_vjp
      def _execute_stage(loop_state_arg, w_curr_arg, params_arg,
                         muts_arg, met_arg, pos_arg, seg_arg):
        return _execute_stage_fwd(loop_state_arg, w_curr_arg, params_arg,
                                   muts_arg, met_arg, pos_arg, seg_arg)[0]

      def _execute_stage_fwd(loop_state_arg, w_curr_arg, params_arg,
                              muts_arg, met_arg, pos_arg, seg_arg):
        iteration = loop_state_arg["loop_iteration"]
        w_next = ctx.weight_prefetching(
            params_arg, physical_partition_spec_full, iteration
        )
        bsw = (w_curr_arg, w_next)

        p_weight_prefetching = functools.partial(
            ctx.weight_prefetching,
            physical_partition_spec=physical_partition_spec_full,
            loop_iteration=iteration,
        )
        weight_prefetching_t = jax.linear_transpose(
            p_weight_prefetching, params_arg
        )

        ((final_ls, final_muts, _bsw_out), metrics), scan_microbatches_vjp = jax.vjp(
            lambda ls, b: _run_microbatches(ls, b, muts_arg, met_arg, pos_arg, seg_arg),
            loop_state_arg, bsw,
        )

        return ((final_ls, w_next, final_muts), metrics), (
            scan_microbatches_vjp, weight_prefetching_t,
        )

      def _execute_stage_bwd(residuals, g_out):
        scan_microbatches_vjp, weight_prefetching_t = residuals
        (g_ls_w_muts, g_metrics) = g_out
        g_ls, g_w_next, g_muts = g_ls_w_muts

        g_w_curr = jax.tree.map(jnp.zeros_like, g_w_next)
        g_bsw = (g_w_curr, g_w_next)
        g_level2_out = ((g_ls, g_muts, g_bsw), g_metrics)

        d_ls, d_bsw = scan_microbatches_vjp(g_level2_out)

        _, d_w_next = d_bsw
        (g_params,) = weight_prefetching_t(d_w_next)

        return d_ls, g_w_curr, g_params, None, None, None, None

      _execute_stage.defvjp(_execute_stage_fwd, _execute_stage_bwd)

      # ---- Execute the stage ----
      (new_loop_state, w_next, new_layer_mutables), inner_metrics = _execute_stage(
          loop_state_carry, w_curr, local_layers_params,
          current_layer_mutables, local_layers_metrics,
          local_positions, local_segment_ids,
      )

      # ---- Write updated mutables back to scope (carry_state only) ----
      new_mut_arrays, _, _, _, _ = _flatten_nnx_state(new_layer_mutables)
      for k, a in zip(mut_keys, new_mut_arrays):
        scope.put_variable('carry_state', k, a)

      # NOTE: We do NOT call scope.put_variable for any _scan_out_* collection.
      # They remain untouched -> identity pass-through -> KNOWN outputs in PE.

      return (new_loop_state, w_next), inner_metrics

    # ---- Build lift.scan(lift.checkpoint(stage_fn)) over REPEATS ----
    if self.config.set_remat_policy_on_pipeline_iterations:
      checkpointed_stage = flax_lift.checkpoint(
          _stage_fn_for_scope,
          variables=True,
          rngs=True,
          prevent_cse=not self.config.scan_pipeline_iterations,
          policy=remat_policy,
      )
    else:
      checkpointed_stage = _stage_fn_for_scope

    # V65e: variable_axes with 4 dummy identity pass-through collections
    # + variable_broadcast with _overwrite_with_gradient and non_trainable
    scanned_stage = flax_lift.scan(
        checkpointed_stage,
        variable_broadcast=[
            'params', 'broadcast_inputs', 'metrics_template',
            '_overwrite_with_gradient', 'non_trainable',
        ],
        variable_carry='carry_state',
        variable_axes={
            '_scan_out_0': 0,
            '_scan_out_1': 0,
            '_scan_out_2': 0,
            '_scan_out_3': 0,
        },
        split_rngs={},
        length=num_repeats,
    )

    # ---- Execute via flax.core.apply ----
    # Mutable list includes carry_state + all 4 dummy collections
    apply_fn = flax_scope.apply(
        lambda scope, init_carry: scanned_stage(scope, init_carry),
        mutable=[
            'carry_state',
            '_scan_out_0', '_scan_out_1', '_scan_out_2', '_scan_out_3',
        ],
    )

    # Prepare initial Linen variables
    initial_w_curr = jax.tree.map(lambda x: jnp.zeros(x.shape[1:], dtype=x.dtype), layers_params)
    init_mut_arrays, _, _, _, _ = _flatten_nnx_state(layers_mutables)
    linen_variables = {
        'params': _arrays_to_linen_collection(param_arrays, param_keys),
        'broadcast_inputs': _arrays_to_linen_collection(input_arrays, input_keys),
        'metrics_template': _arrays_to_linen_collection(met_arrays, met_keys),
        'carry_state': _arrays_to_linen_collection(init_mut_arrays, mut_keys),
        # V65e: Dummy variable_axes collections -- identity pass-through.
        # Shape: (num_repeats,) so lift.scan can index axis 0.
        # Each collection has a single dummy key with a scalar per repeat.
        # These are NEVER written inside the body.
        '_scan_out_0': {DUMMY_KEY: jnp.zeros(num_repeats, dtype=jnp.float32)},
        '_scan_out_1': {DUMMY_KEY: jnp.zeros(num_repeats, dtype=jnp.float32)},
        '_scan_out_2': {DUMMY_KEY: jnp.zeros(num_repeats, dtype=jnp.float32)},
        '_scan_out_3': {DUMMY_KEY: jnp.zeros(num_repeats, dtype=jnp.float32)},
        # V65e: Empty broadcast collections to match V46's broadcast list.
        # Even empty, their presence in the scope may affect PE behavior.
        '_overwrite_with_gradient': {},
        'non_trainable': {},
    }

    # Run the repeat scan
    scan_result, mutated_vars = apply_fn(
        linen_variables,
        (loop_state, initial_w_curr),
    )
    # scan_result = (final_carry, stacked_ys) from lift.scan
    (loop_state, final_w_curr), repeat_metrics_raw = scan_result

    # Extract final mutables from scope
    final_carry_state = mutated_vars['carry_state']
    final_mut_arrays = _linen_collection_to_arrays(final_carry_state, mut_keys)
    final_layer_mutables = _unflatten_nnx_state(
        final_mut_arrays, mut_treedef, mut_is_var, mut_var_types, mut_meta
    )

    # Extract stacked metrics (stacked over repeats by lift.scan)
    repeat_met_arrays = _linen_collection_to_arrays(repeat_metrics_raw, met_keys)
    repeat_metrics = _unflatten_nnx_state(
        repeat_met_arrays, met_treedef, met_is_var, met_var_types, met_meta
    )
    repeat_metrics = jax.tree.map(
        lambda x: x.reshape((num_repeats * num_microbatches,) + x.shape[2:]),
        repeat_metrics,
    )

    # ---- Bubble iterations (pipeline drain) ----
    if bubble_iterations > 0:
      bsw_ref[0] = (final_w_curr, final_w_curr)

      @jax.custom_vjp
      def _run_iter_bubble(loop_state_b, bsw_b, mutables_b, met_b, pos_b, seg_b):
        return _run_iter_bubble_fwd(loop_state_b, bsw_b, mutables_b, met_b, pos_b, seg_b)[0]

      def _run_iter_bubble_fwd(loop_state_b, bsw_b, mutables_b, met_b, pos_b, seg_b):
        def _run(ls, b):
          return ctx.run_one_iteration(
              ls, b, layers_graph, met_b,
              mutables_b, pos_b, seg_b,
              deterministic, model_mode, logical_partition_spec_stripped,
          )
        _run_remat = jax.remat(_run, policy=remat_policy)
        (new_ls, new_layer_state), vjp_fn = jax.vjp(_run_remat, loop_state_b, bsw_b)
        return (new_ls, new_layer_state), vjp_fn

      def _run_iter_bubble_bwd(vjp_fn, g_out):
        g_ls, g_layer_state = g_out
        d_ls, d_bsw = vjp_fn((g_ls, g_layer_state))
        return d_ls, d_bsw, None, None, None, None

      _run_iter_bubble.defvjp(_run_iter_bubble_fwd, _run_iter_bubble_bwd)

      def bubble_body(inner_carry, _):
        ls, muts = inner_carry
        muts = jax.lax.stop_gradient(muts)
        iteration_b = ls["loop_iteration"]
        advanced_muts = _advance_rng_state(muts, iteration_b)
        new_ls, new_layer_state = _run_iter_bubble(
            ls, bsw_ref[0], advanced_muts,
            layers_metrics, positions, segment_ids
        )
        _, _, new_metrics, new_muts = nnx.split(
            new_layer_state, _is_static_param, nnx.Intermediate, ...
        )
        new_met_arrays_b, _, _, _, _ = _flatten_nnx_state(new_metrics)
        return (new_ls, new_muts), _arrays_to_linen_collection(new_met_arrays_b, met_keys)

      if self.config.scan_pipeline_iterations:
        (loop_state, final_layer_mutables), bubble_metrics_raw = jax.lax.scan(
            bubble_body, (loop_state, final_layer_mutables),
            None, length=bubble_iterations,
        )
      else:
        bubble_carry = (loop_state, final_layer_mutables)
        bubble_metrics_list = []
        for _ in range(bubble_iterations):
          bubble_carry, bub_met = bubble_body(bubble_carry, None)
          bubble_metrics_list.append(bub_met)
        loop_state, final_layer_mutables = bubble_carry
        bubble_metrics_raw = (
            jax.tree.map(lambda *xs: jnp.stack(xs), *bubble_metrics_list)
            if bubble_metrics_list else _arrays_to_linen_collection(met_arrays, met_keys)
        )

      bubble_met_arrays = _linen_collection_to_arrays(bubble_metrics_raw, met_keys)
      bubble_metrics = _unflatten_nnx_state(
          bubble_met_arrays, met_treedef, met_is_var, met_var_types, met_meta
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


def create_nnx_pipeline(
    config: Config, stage_factory: Any, mesh: Mesh, remat_policy: Any = None, *, rngs: nnx.Rngs
) -> NNXPipeline | NNXCircularPipeline:
  """Factory function to instantiate the NNX Pipeline module."""
  if config.pipeline_fsdp_ag_per_repeat:
    return NNXCircularPipeline(
        config=config, stage_factory=stage_factory, mesh=mesh, remat_policy=remat_policy, rngs=rngs
    )
  return NNXPipeline(config=config, stage_factory=stage_factory, mesh=mesh, remat_policy=remat_policy, rngs=rngs)


Pipeline = to_linen_class(
    NNXPipeline,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
CircularPipeline = to_linen_class(
    NNXCircularPipeline,
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
