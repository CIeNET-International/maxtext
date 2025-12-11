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
from typing import Any, Optional, Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from flax import nnx
from flax import linen as nn
from MaxText.common_types import Config, MODEL_MODE_TRAIN, EP_AS_CONTEXT
from MaxText.layers import nnx_wrappers
from MaxText.layers.initializers import variable_to_logically_partitioned
from flax.linen import spmd

class Pipeline(nnx.Module):
    """
    NNX Implementation of the MaxText Pipeline.
    Wraps a Flax Linen Module and executes it using Pipeline Parallelism.
    """

    def __init__(
        self,
        layer: nn.Module,
        config: Any,
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

        # --- Axis Naming ---
        if hasattr(self.config, 'expert_shard_attention_option') and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
            self.batch_axis_name = "activation_batch_no_exp"
            self.seq_len_axis_name = "activation_length"
        else:
            self.batch_axis_name = "activation_batch"
            self.seq_len_axis_name = "activation_length_no_exp"

        self.input_shape = (
            self.microbatch_size, 
            self.config.max_target_length, 
            self.config.emb_dim
        )

        # --- Initialization ---
        init_rng_key = rngs.params()
        repeats = self.config.num_pipeline_repeats
        
        # Helper to init one stage
        def init_single_stage(key):
            dummy_in = jnp.zeros(self.input_shape, dtype=jnp.float32)
            dummy_pos_shape = (self.input_shape[0], self.input_shape[1])
            dummy_positions = jnp.zeros(dummy_pos_shape, dtype=jnp.int32)
            dummy_segments = jnp.zeros(dummy_pos_shape, dtype=jnp.int32)
            return self.linen_module.init(
                {'params': key}, 
                dummy_in, 
                decoder_segment_ids=dummy_segments, 
                decoder_positions=dummy_positions,  
                deterministic=True, 
                model_mode=MODEL_MODE_TRAIN
            )

        # 1. Run ONCE without vmap to capture correct Metadata (names)
        # This returns LogicallyPartitioned objects with rules like ('embed', 'mlp')
        ref_variables = init_single_stage(init_rng_key)

        # 2. Run WITH vmap to get the actual Data (values)
        if repeats > 1:
            repeat_keys = jax.random.split(init_rng_key, repeats)
            stage_rng_keys = jax.vmap(lambda k: jax.random.split(k, self.num_stages))(repeat_keys)
            raw_variables = jax.vmap(jax.vmap(init_single_stage))(stage_rng_keys)
        else:
            stage_rng_keys = jax.random.split(init_rng_key, self.num_stages)
            raw_variables = jax.vmap(init_single_stage)(stage_rng_keys)

        # 3. Register Parameters with Metadata Injection
        # We traverse the 'raw' (data) and 'ref' (metadata) trees together.
        def _create_param_with_metadata(raw_node, ref_node):
            if hasattr(raw_node, 'items'):
                return nnx.Dict({k: _create_param_with_metadata(v, ref_node[k]) for k, v in raw_node.items()})
            elif isinstance(raw_node, (list, tuple)):
                return nnx.List([_create_param_with_metadata(v, ref_node[i]) for i, v in enumerate(raw_node)])
            else:
                # Unwrap raw value if it came wrapped
                if isinstance(raw_node, nn.LogicallyPartitioned):
                    actual_value = raw_node.value
                else:
                    actual_value = raw_node

                # Check ref_node for metadata
                sharding_kwargs = {}
                if hasattr(ref_node, 'names'): 
                    original_names = ref_node.names
                    if repeats > 1:
                        new_names = (None, 'stage') + original_names
                    else:
                        new_names = ('stage',) + original_names
                    
                    # Pass as keyword argument to nnx.Param
                    sharding_kwargs['sharding_names'] = new_names

                return nnx.Param(actual_value, **sharding_kwargs)

        self.stage_params = _create_param_with_metadata(raw_variables['params'], ref_variables['params'])


    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def _need_circ_storage(self):
        return (self.config.num_pipeline_repeats > 1 and 
                self.config.num_pipeline_microbatches > self.num_stages * self.forwarding_delay)

    def iterations_to_complete_first_microbatch_one_repeat(self):
        return self.forwarding_delay * (self.num_stages - 1)


    def init_states(self, inputs):
        """Initialize pipeline buffers.
        Args:
            inputs: Rank 4 array [Total_Microbatches, Micro_Size, Seq, Emb]
        """
        # 1. Shift Buffer
        # Shape: [Num_Stages, Micro_Size, Seq, Emb]
        # (Derived from inputs.shape[1:])
        shift = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
        shift = nn.with_logical_constraint(
            shift,
            ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
            rules=self.config.logical_axis_rules,
            mesh=self.mesh,
        )

        # 2. Prev Outputs (for forwarding delay)
        if self.config.pipeline_delay_activation_forwarding:
            prev_outputs = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
            prev_outputs = nn.with_logical_constraint(
                prev_outputs,
                ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
                rules=self.config.logical_axis_rules,
                mesh=self.mesh,
            )
        else:
            prev_outputs = None

        # 3. State IO (The Main Buffer)
        # Reshape: [Total_Micro, ...] -> [Stages, Micro_Per_Stage, ...]
        state_io = jnp.reshape(
            inputs, 
            (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:]
        )
        state_io = nn.with_logical_constraint(
            state_io,
            ("activation_stage", None, self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
            rules=self.config.logical_axis_rules,
            mesh=self.mesh,
        )

        # 4. Circular Storage
        if self.use_circ_storage:
            # Shape: [Num_Stages, Total_Microbatches, Micro_Size, Seq, Emb]
            # (Derived from inputs.shape)
            circ_storage = jnp.zeros((self.num_stages,) + inputs.shape, dtype=inputs.dtype)
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
            "rng_stream": jax.random.PRNGKey(0) 
        }

    def get_iteration_inputs(self, loop_iteration, state_io, circ_storage, shift):
        state_io_batch_idx = loop_iteration % self.microbatches_per_stage
        state_io_slice = state_io[:, state_io_batch_idx]

        if self.use_circ_storage:
            circ_storage_batch_idx = loop_iteration % self.config.num_pipeline_microbatches
            circular_stage_in = circ_storage[:, circ_storage_batch_idx]
        else:
            circular_stage_in = shift

        first_stage_in = jnp.where(
            loop_iteration < self.config.num_pipeline_microbatches, 
            state_io_slice, 
            circular_stage_in
        )

        def select_state_or_input(first_stage_in, shift):
            return jnp.where(
                jax.lax.broadcasted_iota("int32", shift.shape, 0) == 0, 
                first_stage_in, 
                shift
            )

        stages_in = select_state_or_input(first_stage_in, shift)
        
        stages_in = nn.with_logical_constraint(
            stages_in,
            ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
            rules=self.config.logical_axis_rules,
            mesh=self.mesh,
        )
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
            def _rotate_right_and_update(circ_storage_mover_in, circ_storage_in):
                rotated = _rotate_right(circ_storage_mover_in)
                rotated = jnp.expand_dims(rotated, 1)
                offset = (
                    loop_iteration
                    - self.iterations_to_complete_first_microbatch_one_repeat()
                    - 1
                ) % self.config.num_pipeline_microbatches
                return jax.lax.dynamic_update_slice_in_dim(
                    circ_storage_in, rotated, offset, axis=1
                )

            new_circ_storage = _rotate_right_and_update(old_circ_storage_mover, old_circ_storage)
            new_circ_storage_mover = output
        else:
            new_circ_storage = None
            new_circ_storage_mover = None

        stream_buf_idx = loop_iteration % self.microbatches_per_stage
        stream_slice = old_state_io[:, stream_buf_idx]

        def _update_state_io(state_in, stream_slice, output):
            padding = [[0, 1]] + [[0, 0]] * (stream_slice.ndim - 1)
            stream_slice = jax.lax.slice_in_dim(
                jnp.pad(stream_slice, padding), 1, stream_slice.shape[0] + 1, axis=0
            )
            stream_slice = jnp.where(
                jax.lax.broadcasted_iota("int32", stream_slice.shape, 0) == self.num_stages - 1,
                output,
                stream_slice,
            )
            stream_slice = jnp.expand_dims(stream_slice, 1)
            return jax.lax.dynamic_update_slice_in_dim(
                state_in, stream_slice, stream_buf_idx, axis=1
            )

        new_state = _update_state_io(old_state_io, stream_slice, output)

        return {
            "state_io": new_state,
            "shift": new_shift,
            "circ_storage": new_circ_storage,
            "circ_storage_mover": new_circ_storage_mover,
            "loop_iteration": loop_iteration + 1,
            "prev_outputs": new_prev_outputs,
        }

    # ==========================================================================
    # FSDP Helpers
    # ==========================================================================

    def _all_gather_over_fsdp(self, params, partition_spec):
        """Helper to apply FSDP all-gather constraint."""
        def _remove_fsdp_from_spec(spec):
            if isinstance(spec, PartitionSpec):
                new_spec = []
                for axis in spec:
                    if isinstance(axis, str) and axis in ("fsdp", "fsdp_transpose"):
                        new_spec.append(None)
                    elif isinstance(axis, (list, tuple)):
                         new_spec.append(tuple(a for a in axis if a not in ("fsdp", "fsdp_transpose")))
                    else:
                        new_spec.append(axis)
                return PartitionSpec(*new_spec)
            return spec

        def _remove_fsdp_sharding(sharding_tree):
             return jax.tree.map(
                 lambda x: NamedSharding(self.mesh, _remove_fsdp_from_spec(x.spec)) 
                 if isinstance(x, NamedSharding) else x, 
                 sharding_tree
             )

        physical = nn.logical_to_mesh_sharding(partition_spec, mesh=self.mesh, rules=self.config.logical_axis_rules)
        physical_no_fsdp = _remove_fsdp_sharding(physical)
        return jax.lax.with_sharding_constraint(params, physical_no_fsdp)
    
    def _to_pure_dict(self, node):
        """Recursively converts NNX containers to standard Python dicts/lists."""
        if hasattr(node, 'items'):  # Handles nnx.Dict, dict, FrozenDict
            return {k: self._to_pure_dict(v) for k, v in node.items()}
        elif isinstance(node, (list, tuple)): # Handles nnx.List, list, tuple
            return [self._to_pure_dict(v) for v in node]
        elif hasattr(node, 'value'): # Handles nnx.Param, nnx.Variable
            return node.value
        return node # Leaf (e.g. JAX Array/Tracer)

    # ==========================================================================
    # Main Execution
    # ==========================================================================
    def get_microbatch_ids(self, loop_iteration):
        """Calculates the microbatch ID for each stage at the current tick."""
        # Calculate how many microbatches each stage has processed effectively
        # Stage 0 starts at 0. Stage 1 starts after 'forwarding_delay', etc.
        microbatches_processed = jnp.maximum(
            loop_iteration - self.forwarding_delay * jnp.arange(self.num_stages), 
            0
        )
        # Wrap around using modulo to get the ID within the circular buffer
        microbatch_ids = microbatches_processed % self.total_microbatches
        return microbatch_ids.astype(jnp.int32)

    # Handles logic when num_pipeline_repeats > 1
    def get_microbatch_and_repeat_ids(self, loop_iteration):
        """Gets the microbatch_ids and repeat_ids for all stages on this loop_iteration."""
        microbatches_processed = jnp.maximum(
            loop_iteration - self.forwarding_delay * jnp.arange(self.num_stages), 
            0
        )
        microbatch_ids = microbatches_processed % self.total_microbatches
        repeat_ids = microbatches_processed // self.total_microbatches
        return microbatch_ids.astype(jnp.int32), repeat_ids.astype(jnp.int32)

    # Helper for weight gathering
    def shard_dim_by_stages(self, x, dim: int):
        dims_mapping = [PartitionSpec.UNCONSTRAINED] * x.ndim
        dims_mapping[dim] = "stage"
        # We construct the NamedSharding manually to match Linen's logical_to_mesh
        # Note: In pure NNX/JAX, we can often just return x if sharding is handled by vmap, 
        # but we keep the constraint for strict parity.
        sharding = NamedSharding(self.mesh, PartitionSpec(*dims_mapping))
        return jax.lax.with_sharding_constraint(x, sharding)

    # Helper for circular weight selection
    def vmap_parallel_gather(self, weights, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights):
        """Use vmap to implement a sharded parallel gather for weights."""
        def _gather_one(x, repeat_id):
            return jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, repeat_id, 1, repeat_dim_in_weights), repeat_dim_in_weights)

        gathered_weights_stage_dim = 0
        repeat_ids = self.shard_dim_by_stages(repeat_ids, 0)
        weights = self.shard_dim_by_stages(weights, stages_dim_in_weights)
        
        stage_weights = jax.vmap(_gather_one, in_axes=(stages_dim_in_weights, 0), out_axes=gathered_weights_stage_dim)(weights, repeat_ids)
        stage_weights = self.shard_dim_by_stages(stage_weights, gathered_weights_stage_dim)
        return stage_weights

    # The main function missing from scan_body
    def get_current_stage_weights(self, pipeline_weights, loop_iteration):
        if self.config.num_pipeline_repeats <= 1:
            return pipeline_weights
        _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)
        
        # Helper to map over the PyTree of weights
        def gather_weights_for_stages_in(w):
            # Assumes weights are [Repeats, Stages, ...] -> Gather -> [Stages, ...]
            return self.vmap_parallel_gather(
                w, repeat_ids, repeat_dim_in_weights=0, stages_dim_in_weights=1
            )
        
        return jax.tree.map(gather_weights_for_stages_in, pipeline_weights)

    def get_pipeline_remat_policy(self):
        # Strict policy: Save ONLY the input tensor named 'iteration_input'.
        # Discard everything else (dots, gathers, etc).
        save_input_policy = jax.checkpoint_policies.save_only_these_names(
            "iteration_input", "decoder_layer_input"
        )
        
        if self.config.remat_policy == "custom":
            return self.remat_policy
            
        if self.remat_policy is not None:
            return jax.checkpoint_policies.save_from_both_policies(
                self.remat_policy, save_input_policy
            )
        return save_input_policy


    # Output sorting
    def permute_output_micro_per_stage_dim(self, output):
        microbatch_0_idx = self.iterations_to_complete_first_microbatch_one_repeat() % self.microbatches_per_stage
        permutation = (jnp.arange(self.microbatches_per_stage) + microbatch_0_idx) % self.microbatches_per_stage
        output = output[:, permutation]
        return output
    
    def __call__(
        self, 
        inputs: jax.Array, 
        segment_ids: Optional[jax.Array] = None, 
        positions: Optional[jax.Array] = None, 
        deterministic: bool = True,
        model_mode: str = MODEL_MODE_TRAIN,
        partition_spec: Any = None,
    ) -> jax.Array:
        
        # 1. Reshape Inputs
        inputs = inputs.reshape((
            self.total_microbatches, 
            self.microbatch_size, 
            self.config.max_target_length, 
            self.config.emb_dim
        ))
        
        if positions is not None:
            positions = positions.reshape((
                self.total_microbatches, 
                self.microbatch_size, 
                self.config.max_target_length
            ))
            
        if segment_ids is not None:
            segment_ids = segment_ids.reshape((
                self.total_microbatches, 
                self.microbatch_size, 
                self.config.max_target_length
            ))

        # 2. Initialize State
        loop_state = self.init_states(inputs)
        
        # 3. Prepare Weights
        param_values = self._to_pure_dict(self.stage_params)
        
        if self.config.pipeline_fsdp_ag_once and partition_spec is not None:
             try:
                param_values = self._all_gather_over_fsdp(param_values, partition_spec)
             except (ValueError, TypeError, KeyError):
                 pass

        # 4. Scan Loop
        def scan_body(carry, _):
            iteration = carry['loop_iteration']
            current_rng = carry['rng_stream']
            
            step_rng, next_rng = jax.random.split(current_rng)
            stage_rngs = jax.random.split(step_rng, self.num_stages)
            
            stages_inputs = self.get_iteration_inputs(
                iteration, 
                carry['state_io'], 
                carry['circ_storage'], 
                carry['shift']
            )
            # Checkpoint inputs: Crucial for policy detection
            stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, "iteration_input")
            
            # Gather Positions & Segments
            mb_ids, _ = self.get_microbatch_and_repeat_ids(iteration)
            stages_positions = jnp.take(positions, mb_ids, axis=0) if positions is not None else None
            stages_segment_ids = jnp.take(segment_ids, mb_ids, axis=0) if segment_ids is not None else None

            # Dynamic Weight Selection
            current_params = self.get_current_stage_weights(param_values, iteration)

            # --- A. Define the VMAP Logic (Pure Execution) ---
            def execution_logic(params, inputs, rngs, pos, seg):
                # Inner function applied to ONE stage
                def stage_fn(p, x, r, po, se):
                    variables = {'params': p}
                    rngs_dict = {'dropout': r} if not deterministic else {}
                    return self.linen_module.apply(
                        variables, x, 
                        decoder_segment_ids=se, decoder_positions=po,
                        deterministic=deterministic, rngs=rngs_dict,
                        model_mode=model_mode
                    )
                
                # Apply VMAP here
                # Map axes: Params(0), Inputs(0), Rngs(0), Pos(0/None), Seg(0/None)
                vmap_axes = [0, 0, 0]
                vmap_args = [params, inputs, rngs]
                
                vmap_axes.append(0 if pos is not None else None)
                vmap_args.append(pos)
                
                vmap_axes.append(0 if seg is not None else None)
                vmap_args.append(seg)
                
                return jax.vmap(stage_fn, in_axes=tuple(vmap_axes))(*vmap_args)

            # --- B. Apply Checkpoint to the VMAP (Parity with Linen) ---
            # We wrap the entire execution logic (which contains the vmap)
            
            if self.config.set_remat_policy_on_pipeline_iterations:
                policy = self.get_pipeline_remat_policy()
                
                # Checkpoint the function that DOES the vmap
                execution_logic = jax.checkpoint(
                    execution_logic, 
                    policy=policy,
                    prevent_cse=False
                )

            # --- C. Execute ---
            stages_output = execution_logic(
                current_params, stages_inputs, stage_rngs, stages_positions, stages_segment_ids
            )
            
            if hasattr(self.config, 'scan_layers') and self.config.scan_layers:
                 if isinstance(stages_output, tuple):
                     stages_output = stages_output[0]

            new_loop_state = self.get_new_loop_state(stages_output, carry)
            new_loop_state['rng_stream'] = next_rng
            
            return new_loop_state, None

        bubble_iterations = self.forwarding_delay * (self.num_stages - 1)
        real_iterations = self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats
        total_ticks = real_iterations + bubble_iterations

        final_state, _ = jax.lax.scan(scan_body, loop_state, None, length=total_ticks)

        output = final_state['state_io']
        output = self.permute_output_micro_per_stage_dim(output)
        output = output.reshape((self.config.micro_batch_size_to_train_on, 
                                 self.config.max_target_length, 
                                 self.config.emb_dim))
        
        return output

def add_stage_axis_to_partitioning(variable, repeats=1):
    """
    Metadata function for to_linen.
    """
    # 1. Try to recover metadata
    partitioned_obj = variable_to_logically_partitioned(variable)
    
    if isinstance(partitioned_obj, nn.LogicallyPartitioned):
        base_names = partitioned_obj.names
        value = partitioned_obj.value
    else:
        # 2. Fallback to Heuristics (Crucial for jax.vmap created variables)
        value = partitioned_obj
        if not hasattr(value, 'ndim'):
            return value
        base_names = None

    ndim = value.ndim
    
    if repeats > 1:
        # [Repeats, Stage, Inner...]
        if base_names is not None:
            new_names = (None, 'stage') + base_names
        else:
            # Heuristic for Rank 4 (Repeats, Stage, Embed, MLP)
            if ndim == 4:
                new_names = (None, 'stage', 'fsdp', 'tensor')
            elif ndim == 3:
                new_names = (None, 'stage', 'fsdp')
            else:
                new_names = (None, 'stage') + (None,) * (ndim - 2)
    else:
        # [Stage, Inner...]
        if base_names is not None:
             new_names = ('stage',) + base_names
        else:
            # Heuristic for Rank 3 (Stage, Embed, MLP)
            if ndim == 3: 
                new_names = ('stage', 'fsdp', 'tensor')
            elif ndim == 2: 
                new_names = ('stage', 'fsdp')
            else:
                new_names = ('stage',) + (None,) * (ndim - 1)

    return nn.LogicallyPartitioned(value, new_names)

def create_pipeline(
    config: Config,
    layer: Callable | type,
    mesh: Mesh,
    remat_policy: Any = None,
) -> nnx_wrappers.ToLinen:
    """Factory function to create a Pipeline wrapped as a Linen module."""

    repeats = getattr(config, "num_pipeline_repeat", 1)
    metadata_fn = functools.partial(add_stage_axis_to_partitioning,repeats=repeats)
    return nnx_wrappers.to_linen(
        Pipeline,
        config=config,
        mesh=mesh,
        layer=layer,
        remat_policy=remat_policy,
        name="pipeline_module",
        abstract_init=False,
        metadata_fn=metadata_fn,
    )
