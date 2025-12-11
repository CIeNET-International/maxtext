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

"""Pipeline layer wrapping a decoder layer(s). Supports circular pipelining.
NNX Implementation.
"""
import functools
from typing import Any, Optional, Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from flax import nnx
from flax import linen as nn

from MaxText.common_types import Config, MODEL_MODE_TRAIN, EP_AS_CONTEXT
from MaxText.layers import nnx_wrappers
from MaxText.layers.initializers import variable_to_logically_partitioned

# ==============================================================================
# Shared Sharding Logic (Heuristics)
# ==============================================================================

def _infer_partition_names(value, repeats=1, base_names=None):
    """Helper to infer logical axis names (PartitionSpec) for pipeline weights."""
    ndim = value.ndim
    
    if repeats > 1:
        # Structure: [Repeats, Stage, ...]
        if base_names is not None:
             new_names = (None, 'stage') + base_names
        else:
            # Heuristics for [Repeats, Stage, ...]
            if ndim == 4: # [Repeats, Stage, Embed, MLP]
                new_names = (None, 'stage', 'fsdp', 'tensor')
            elif ndim == 3: # [Repeats, Stage, Bias]
                new_names = (None, 'stage', 'fsdp')
            else:
                new_names = (None, 'stage') + (None,) * (ndim - 2)
    else:
        # Structure: [Stage, ...]
        if base_names is not None:
            new_names = ('stage',) + base_names
        else:
             # Heuristics for [Stage, ...]
            if ndim == 3: # [Stage, Embed, MLP]
                new_names = ('stage', 'fsdp', 'tensor')
            elif ndim == 2: # [Stage, Bias]
                new_names = ('stage', 'fsdp')
            else:
                new_names = ('stage',) + (None,) * (ndim - 1)
    
    return new_names

# ==============================================================================
# Pipeline Class
# ==============================================================================

class Pipeline(nnx.Module):
    """NNX Implementation of the MaxText Pipeline."""

    def __init__(
        self,
        layer: nn.Module,
        config: Config,
        mesh: Mesh,
        rngs: nnx.Rngs,
        remat_policy: Any = None
    ):
        self.config = config
        self.mesh = mesh
        self.linen_module = layer
        self.remat_policy = remat_policy

        # --- Dimensions ---
        self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
        self.forwarding_delay = 2 if self.config.pipeline_delay_activation_forwarding else 1
        self.total_microbatches = self.config.num_pipeline_microbatches
        self.microbatch_size = self.config.micro_batch_size_to_train_on // self.total_microbatches
        self.microbatches_per_stage = self.total_microbatches // self.num_stages
        self.use_circ_storage = self._need_circ_storage()

        # Axis Naming
        if hasattr(self.config, 'expert_shard_attention_option') and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
            self.batch_axis_name = "activation_batch_no_exp"
            self.seq_len_axis_name = "activation_length"
        else:
            self.batch_axis_name = "activation_batch"
            self.seq_len_axis_name = "activation_length_no_exp"

        self.input_shape = (self.microbatch_size, self.config.max_target_length, self.config.emb_dim)

        # --- Eager Initialization with JIT + Out Shardings ---
        init_rng_key = rngs.params()
        repeats = self.config.num_pipeline_repeats

        # 1. Define the Generator (Pure JAX, no side effects)
        def generate_vars_fn(key):
            def init_single_stage(k):
                # Create dummy inputs locally to ensure shape inference works inside JIT
                dummy_in = jnp.zeros(self.input_shape, dtype=jnp.float32)
                dummy_pos_shape = (self.input_shape[0], self.input_shape[1])
                dummy_positions = jnp.zeros(dummy_pos_shape, dtype=jnp.int32)
                dummy_segments = jnp.zeros(dummy_pos_shape, dtype=jnp.int32)
                
                return self.linen_module.init(
                    {'params': k},
                    dummy_in,
                    dummy_segments,
                    dummy_positions,
                    True,
                    MODEL_MODE_TRAIN
                )

            if repeats > 1:
                repeat_keys = jax.random.split(key, repeats)
                # Outer vmap over repeats, Inner vmap over stages
                stage_rng_keys = jax.vmap(lambda k: jax.random.split(k, self.num_stages))(repeat_keys)
                return jax.vmap(jax.vmap(init_single_stage))(stage_rng_keys)
            else:
                stage_rng_keys = jax.random.split(key, self.num_stages)
                return jax.vmap(init_single_stage)(stage_rng_keys)

        # 2. Abstract Evaluation (Zero Memory)
        # Determine the shape and structure of the variables without allocating data
        abstract_variables = jax.eval_shape(generate_vars_fn, init_rng_key)

        # 3. Compute Target Sharding (Heuristics)
        # We assume the output structure matches what we expect for partitioning
        def get_target_sharding(abstract_leaf):
            # abstract_leaf is a ShapeDtypeStruct. We check its ndim.
            logical_names = _infer_partition_names(abstract_leaf, repeats=repeats)
            
            if logical_names:
                mesh_axes = nn.logical_to_mesh_axes(logical_names, self.config.logical_axis_rules)
                return NamedSharding(self.mesh, PartitionSpec(*mesh_axes))
            
            # Default fallback: Replicated
            return NamedSharding(self.mesh, PartitionSpec())

        sharding_tree = jax.tree.map(get_target_sharding, abstract_variables)

        # 4. JIT Compile with Out Shardings (Direct Allocation)
        # This tells XLA to compile a kernel that writes outputs directly to their 
        # final sharded destination, bypassing the single-device OOM bottleneck.
        sharded_variables = jax.jit(generate_vars_fn, out_shardings=sharding_tree)(init_rng_key)

        # --- Register with NNX ---
        self.layers = self._to_nnx_structure(sharded_variables)
        
        # Handle batch_stats if present
        if 'batch_stats' in sharded_variables:
            pass
    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def _to_nnx_structure(self, node):
        if isinstance(node, (dict, nn.FrozenDict)):
            return nnx.Dict({k: self._to_nnx_structure(v) for k, v in node.items()})
        elif isinstance(node, (list, tuple)):
            return nnx.List([self._to_nnx_structure(v) for v in node])
        else:
            return nnx.Param(node)

    def _to_pure_dict(self, node):
        if hasattr(node, 'items'): return {k: self._to_pure_dict(v) for k, v in node.items()}
        elif isinstance(node, (list, tuple)): return [self._to_pure_dict(v) for v in node]
        elif hasattr(node, 'value'): return node.value
        return node 

    def _with_logical_constraint(self, x, axis_names):
        if axis_names is None: return x
        mesh_axes = nn.logical_to_mesh_axes(axis_names, self.config.logical_axis_rules)
        sharding = NamedSharding(self.mesh, PartitionSpec(*mesh_axes))
        return jax.lax.with_sharding_constraint(x, sharding)

    def _need_circ_storage(self):
        return (self.config.num_pipeline_repeats > 1 and 
                self.config.num_pipeline_microbatches > self.num_stages * self.forwarding_delay)

    def iterations_to_complete_first_microbatch_one_repeat(self):
        return self.forwarding_delay * (self.num_stages - 1)

    # ==========================================================================
    # Buffer Management
    # ==========================================================================

    def init_states(self, inputs):
        shift = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
        shift = self._with_logical_constraint(shift, ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed"))
        
        if self.config.pipeline_delay_activation_forwarding:
            prev_outputs = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
            prev_outputs = self._with_logical_constraint(prev_outputs, ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed"))
        else:
            prev_outputs = None
            
        state_io = jnp.reshape(inputs, (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:])
        state_io = self._with_logical_constraint(state_io, ("activation_stage", None, self.batch_axis_name, self.seq_len_axis_name, "activation_embed"))
        
        if self.use_circ_storage:
            circ_storage = jnp.zeros((self.num_stages,) + inputs.shape, dtype=inputs.dtype)
            circ_storage_mover = shift
        else:
            circ_storage = None
            circ_storage_mover = None
            
        return {"state_io": state_io, "shift": shift, "circ_storage": circ_storage, "circ_storage_mover": circ_storage_mover, "loop_iteration": jnp.array(0, dtype=jnp.int32), "prev_outputs": prev_outputs, "rng_stream": jax.random.PRNGKey(0)}

    def get_iteration_inputs(self, loop_iteration, state_io, circ_storage, shift):
        state_io_batch_idx = loop_iteration % self.microbatches_per_stage
        state_io_slice = state_io[:, state_io_batch_idx]
        
        if self.use_circ_storage:
            circ_storage_batch_idx = loop_iteration % self.config.num_pipeline_microbatches
            circular_stage_in = circ_storage[:, circ_storage_batch_idx]
        else:
            circular_stage_in = shift
            
        first_stage_in = jnp.where(loop_iteration < self.config.num_pipeline_microbatches, state_io_slice, circular_stage_in)
        
        stages_in = jnp.where(jax.lax.broadcasted_iota("int32", shift.shape, 0) == 0, first_stage_in, shift)
        stages_in = self._with_logical_constraint(stages_in, ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed"))
        return stages_in

    def get_new_loop_state(self, output, loop_state):
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
            return _rotate_right(output_in)
        
        if self.config.pipeline_delay_activation_forwarding:
            new_shift = _update_shift(old_prev_outputs)
            new_prev_outputs = output
        else:
            new_shift = _update_shift(output)
            new_prev_outputs = None
            
        if self.use_circ_storage:
            rotated = _rotate_right(old_circ_storage_mover)
            rotated = jnp.expand_dims(rotated, 1)
            offset = (loop_iteration - self.iterations_to_complete_first_microbatch_one_repeat() - 1) % self.config.num_pipeline_microbatches
            new_circ_storage = jax.lax.dynamic_update_slice_in_dim(old_circ_storage, rotated, offset, axis=1)
            new_circ_storage_mover = output
        else:
            new_circ_storage = None
            new_circ_storage_mover = None
            
        stream_buf_idx = loop_iteration % self.microbatches_per_stage
        stream_slice = old_state_io[:, stream_buf_idx]
        padding = [[0, 1]] + [[0, 0]] * (stream_slice.ndim - 1)
        stream_slice = jax.lax.slice_in_dim(jnp.pad(stream_slice, padding), 1, stream_slice.shape[0] + 1, axis=0)
        stream_slice = jnp.where(jax.lax.broadcasted_iota("int32", stream_slice.shape, 0) == self.num_stages - 1, output, stream_slice)
        stream_slice = jnp.expand_dims(stream_slice, 1)
        new_state = jax.lax.dynamic_update_slice_in_dim(old_state_io, stream_slice, stream_buf_idx, axis=1)
        
        return {"state_io": new_state, "shift": new_shift, "circ_storage": new_circ_storage, "circ_storage_mover": new_circ_storage_mover, "loop_iteration": loop_iteration + 1, "prev_outputs": new_prev_outputs}

    # ==========================================================================
    # Logic for VMAP and Weight Gathering
    # ==========================================================================

    def get_microbatch_and_repeat_ids(self, loop_iteration):
        microbatches_processed = jnp.maximum(loop_iteration - self.forwarding_delay * jnp.arange(self.num_stages), 0)
        return microbatches_processed % self.total_microbatches, (microbatches_processed // self.total_microbatches).astype(jnp.int32)

    def shard_dim_by_stages(self, x, dim: int):
        dims_mapping = [PartitionSpec.UNCONSTRAINED] * x.ndim
        dims_mapping[dim] = "stage"
        sharding = NamedSharding(self.mesh, PartitionSpec(*dims_mapping))
        return jax.lax.with_sharding_constraint(x, sharding)

    def vmap_parallel_gather(self, weights, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights):
        def _gather_one(x, repeat_id):
            return jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, repeat_id, 1, repeat_dim_in_weights), repeat_dim_in_weights)
        gathered_weights_stage_dim = 0
        repeat_ids = self.shard_dim_by_stages(repeat_ids, 0)
        weights = self.shard_dim_by_stages(weights, stages_dim_in_weights)
        stage_weights = jax.vmap(_gather_one, in_axes=(stages_dim_in_weights, 0), out_axes=gathered_weights_stage_dim)(weights, repeat_ids)
        return self.shard_dim_by_stages(stage_weights, gathered_weights_stage_dim)

    def get_current_stage_weights(self, pipeline_weights, loop_iteration):
        if self.config.num_pipeline_repeats <= 1: return pipeline_weights
        _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)
        return jax.tree.map(lambda w: self.vmap_parallel_gather(w, repeat_ids, 0, 1), pipeline_weights)

    def permute_output_micro_per_stage_dim(self, output):
        microbatch_0_idx = self.iterations_to_complete_first_microbatch_one_repeat() % self.microbatches_per_stage
        permutation = (jnp.arange(self.microbatches_per_stage) + microbatch_0_idx) % self.microbatches_per_stage
        return output[:, permutation]

    # ==========================================================================
    # Integration Methods
    # ==========================================================================

    def get_pipeline_remat_policy(self):
        if self.config.remat_policy == "custom": return self.remat_policy
        save_input_policy = jax.checkpoint_policies.save_only_these_names("iteration_input", "decoder_layer_input")
        if self.remat_policy is not None: return jax.checkpoint_policies.save_from_both_policies(self.remat_policy, save_input_policy)
        return save_input_policy

    def get_weight_sharding(self, *init_args):
        variables = self.layers
        repeats = self.config.num_pipeline_repeats
        def _infer_partition_spec(node):
            if hasattr(node, 'value'):
                names = _infer_partition_names(node.value, repeats=repeats, base_names=None)
                return nn.LogicallyPartitioned(node.value, names).get_partition_spec()
            return None
        specs = jax.tree.map(_infer_partition_spec, variables, is_leaf=lambda x: hasattr(x, 'value'))
        return {'params': {'layers': specs['params']}}

    def _all_gather_over_fsdp(self, params, partition_spec):
        def _remove_fsdp_from_spec(spec):
            if isinstance(spec, PartitionSpec):
                new_spec = []
                for axis in spec:
                    if isinstance(axis, str) and axis in ("fsdp", "fsdp_transpose"): new_spec.append(None)
                    elif isinstance(axis, (list, tuple)): new_spec.append(tuple(a for a in axis if a not in ("fsdp", "fsdp_transpose")))
                    else: new_spec.append(axis)
                return PartitionSpec(*new_spec)
            return spec
        def _remove_fsdp_sharding(sharding_tree):
            return jax.tree.map(lambda x: NamedSharding(self.mesh, _remove_fsdp_from_spec(x.spec)) if isinstance(x, NamedSharding) else x, sharding_tree)
        physical = nn.logical_to_mesh_sharding(partition_spec, mesh=self.mesh, rules=self.config.logical_axis_rules)
        physical_no_fsdp = _remove_fsdp_sharding(physical)
        return jax.lax.with_sharding_constraint(params, physical_no_fsdp)

    # ==========================================================================
    # Main Call (__call__)
    # ==========================================================================

    def __call__(self, inputs: jax.Array, segment_ids: Optional[jax.Array] = None, positions: Optional[jax.Array] = None, deterministic: bool = True, model_mode: str = MODEL_MODE_TRAIN, partition_spec: Any = None) -> jax.Array:
        inputs = inputs.reshape((self.total_microbatches, self.microbatch_size, self.config.max_target_length, self.config.emb_dim))
        if positions is not None: positions = positions.reshape((self.total_microbatches, self.microbatch_size, self.config.max_target_length))
        if segment_ids is not None: segment_ids = segment_ids.reshape((self.total_microbatches, self.microbatch_size, self.config.max_target_length))
        
        loop_state = self.init_states(inputs)
        layer_variables = self._to_pure_dict(self.layers)
        
        compute_dtype = getattr(self.config, 'compute_dtype', jnp.bfloat16)
        if isinstance(compute_dtype, str):
            compute_dtype = {'bfloat16': jnp.bfloat16, 'float32': jnp.float32, 'float16': jnp.float16}.get(compute_dtype, jnp.bfloat16)
        layer_variables = jax.tree.map(lambda x: x.astype(compute_dtype), layer_variables)

        if self.config.pipeline_fsdp_ag_once and partition_spec is not None:
            try:
                if "params" in partition_spec and "layers" in partition_spec["params"]:
                     params_only = layer_variables['params']
                     params_spec = partition_spec['params']['layers']['params']
                     layer_variables['params'] = self._all_gather_over_fsdp(params_only, params_spec)
            except (KeyError, TypeError): pass

        def scan_body(carry, _):
            iteration = carry['loop_iteration']
            current_rng = carry['rng_stream']
            step_rng, next_rng = jax.random.split(current_rng)
            stage_rngs = jax.random.split(step_rng, self.num_stages)
            stages_inputs = self.get_iteration_inputs(iteration, carry['state_io'], carry['circ_storage'], carry['shift'])
            stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, "iteration_input")
            mb_ids, _ = self.get_microbatch_and_repeat_ids(iteration)
            stages_positions = jnp.take(positions, mb_ids, axis=0) if positions is not None else None
            stages_segment_ids = jnp.take(segment_ids, mb_ids, axis=0) if segment_ids is not None else None
            current_vars = self.get_current_stage_weights(layer_variables, iteration)

            def execution_logic(vars_in, inputs, rngs, pos, seg):
                def stage_fn(v, x, r, po, se):
                    rngs_dict = {'dropout': r} if not deterministic else {}
                    mutables = ['aux_loss', 'intermediates']
                    return self.linen_module.apply(v, x, se, po, deterministic, model_mode, rngs=rngs_dict, mutable=mutables)
                vmap_axes = [0, 0, 0]
                vmap_args = [vars_in, inputs, rngs]
                vmap_axes.append(0 if pos is not None else None); vmap_args.append(pos)
                vmap_axes.append(0 if seg is not None else None); vmap_args.append(seg)
                return jax.vmap(stage_fn, in_axes=tuple(vmap_axes))(*vmap_args)

            if self.config.set_remat_policy_on_pipeline_iterations:
                policy = self.get_pipeline_remat_policy()
                execution_logic = jax.checkpoint(execution_logic, policy=policy, prevent_cse=not self.config.scan_pipeline_iterations)

            stages_output, stages_mutables = execution_logic(current_vars, stages_inputs, stage_rngs, stages_positions, stages_segment_ids)
            if hasattr(self.config, 'scan_layers') and self.config.scan_layers:
                 if isinstance(stages_output, tuple): stages_output = stages_output[0]
            
            new_loop_state = self.get_new_loop_state(stages_output, carry)
            new_loop_state['rng_stream'] = next_rng
            return new_loop_state, stages_mutables

        bubble_iterations = self.forwarding_delay * (self.num_stages - 1)
        real_iterations = self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats
        total_ticks = real_iterations + bubble_iterations
        final_state, stacked_mutables = jax.lax.scan(scan_body, loop_state, None, length=total_ticks)
        output = final_state['state_io']
        output = self.permute_output_micro_per_stage_dim(output)
        output = output.reshape((self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim))
        return output

# ==============================================================================
# Factory
# ==============================================================================

def add_stage_axis_to_partitioning(variable, repeats=1):
    """Metadata helper for ToLinen."""
    partitioned_obj = variable_to_logically_partitioned(variable)
    if isinstance(partitioned_obj, nn.LogicallyPartitioned):
        base_names = partitioned_obj.names
        value = partitioned_obj.value
    else:
        value = partitioned_obj
        if not hasattr(value, 'ndim'): return value
        base_names = None
    
    new_names = _infer_partition_names(value, repeats=repeats, base_names=base_names)
    return nn.LogicallyPartitioned(value, new_names)

def create_pipeline(config: Config, layer: Callable | type, mesh: Mesh, remat_policy: Any = None) -> nnx_wrappers.ToLinen:
    repeats = getattr(config, 'num_pipeline_repeats', 1)
    metadata_fn = functools.partial(add_stage_axis_to_partitioning, repeats=repeats)
    return nnx_wrappers.to_linen(Pipeline, config=config, mesh=mesh, layer=layer, remat_policy=remat_policy, name="pipeline_module", abstract_init=False, metadata_fn=metadata_fn)