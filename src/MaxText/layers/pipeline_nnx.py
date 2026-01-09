"""PipelineParallelismModule for MaxText using Flax NNX.
Native NNX Vectorized version for memory/speed efficiency.
Updated to match latest Linen Pipeline logic (SPMD rotations).
"""

import functools
from typing import Any, Optional, Dict, Type, Tuple, List
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import shard_map

from flax import nnx
from flax import linen as nn_linen # Still needed for logical_to_mesh helpers
from flax.linen.spmd import LogicallyPartitioned # Kept for compatibility checking

from MaxText.common_types import Config, MODEL_MODE_TRAIN, EP_AS_CONTEXT, ShardMode
from MaxText.sharding import (
    maybe_shard_with_logical,
    maybe_shard_with_name,
    create_sharding,
    logical_to_mesh_axes,
    logical_to_mesh,
)

# --- Helpers ---

def debug_pytree_stats(name, tree):
    """Prints the number of leaves and total memory footprint of a Pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    num_leaves = len(leaves)
    # Calculate size in GB (assuming most are BF16/FP32)
    total_bytes = sum(x.nbytes if hasattr(x, "nbytes") else 0 for x in leaves)
    total_gb = total_bytes / (1024**3)
    
    # Only print on Lead Host to avoid log spam
    if jax.process_index() == 0:
        print(f"--- [DEBUG] {name} ---")
        print(f"Count: {num_leaves} arrays")
        print(f"Size:  {total_gb:.4f} GB")

def cast_to_dtype(node, dtype):
    """Recursively casts all floating point arrays in a Pytree/State to the target dtype."""
    def _cast(leaf):
        if isinstance(leaf, (jax.Array, jnp.ndarray)) and jnp.issubdtype(leaf.dtype, jnp.floating):
            return leaf.astype(dtype)
        return leaf
    return jax.tree_util.tree_map(_cast, node)

def to_pure_dict(x):
    """Recursively converts any nnx.State or custom mapping into a plain Python dict."""
    if hasattr(x, "items") and not isinstance(x, (jnp.ndarray, jax.Array)):
        return {k: to_pure_dict(v) for k, v in x.items()}
    return x

# --- NNX Pipeline Module ---

class ScannedBlock(nnx.Module):
    """A container that allows a State/GraphDef pair to live inside NNX collections."""
    def __init__(self, state: nnx.Variable, graphdef: nnx.GraphDef):
        self.state = state
        self.graphdef = graphdef

class InternalMetrics(nnx.Variable):
    """Variable type for diagnostic activation metrics."""
    pass

class Pipeline(nnx.Module):
    def __init__(self, layers: nnx.Module, config: Config, mesh: Mesh, remat_policy: Any = None, rngs: nnx.Rngs | None = None):
        self.config = config
        self.mesh = mesh
        self.remat_policy = remat_policy
        self.rngs = rngs

        # 1. Pipeline Dimensions
        self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
        self.forwarding_delay = 2 if self.config.pipeline_delay_activation_forwarding else 1
        self.pipeline_microbatch_size = self.config.micro_batch_size_to_train_on // self.config.num_pipeline_microbatches
        self.microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages
        
        num_repeats = self.config.num_pipeline_repeats if self.config.num_pipeline_repeats > 1 else 1
        self.total_instances = num_repeats * self.num_stages

        # 2. Logical Axis Setup & Sharding Specs
        self.use_circ_storage = self.need_circ_storage()
        
        if self.config.expert_shard_attention_option == EP_AS_CONTEXT:
            self.batch_axis_name = "activation_batch_no_exp"
            self.seq_len_axis_name = "activation_length"
        else:
            self.batch_axis_name = "activation_batch"
            self.seq_len_axis_name = "activation_length_no_exp"
            
        self.spmd_axis_name = "stage" if self.config.shard_mode == ShardMode.AUTO else None

        # Pre-calculate Sharding Specs for shard_map (ported from Linen)
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

        # 3. Model Initialization (NNX Specific)
        if rngs is None:
            raise ValueError("Pipeline requires 'rngs' for initialization.")
        
        v_rngs = self.rngs.fork(split=self.total_instances)
        
        factory_kwargs = {
            "config": self.config,
            "mesh": self.mesh,
            "decoder_layer": getattr(layers, "decoder_layer", None),
            "num_decoder_layers": getattr(layers, "num_decoder_layers", 0),
            "model_mode": getattr(layers, "model_mode", MODEL_MODE_TRAIN),
            "quant": getattr(layers, "quant", None),
            "scan_layers": getattr(layers, "scan_layers", False),
            "dtype": self.config.dtype,
        }
        
        LayerCls = type(layers)

        def create_sharded_stage(r):
            m = LayerCls(rngs=r, **factory_kwargs)
            # Dummy pass to initialize
            m(
                jnp.zeros((1, 1, self.config.emb_dim), dtype=self.config.dtype),
                jnp.zeros((1, 1), dtype=jnp.int32),
                jnp.zeros((1, 1), dtype=jnp.int32),
                deterministic=False,
                model_mode=MODEL_MODE_TRAIN,
            )
            _, state = nnx.split(m)
            bf16_state = cast_to_dtype(state, self.config.dtype)
            nnx.update(m, bf16_state)
            nnx.pop(m, nnx.RngStream)
            return m

        # Initialize vectorized layers
        with self.mesh:
            self.layers = nnx.vmap(create_sharded_stage, in_axes=0, spmd_axis_name="stage")(v_rngs)

    # --- Wrappers for Sharding Helpers ---
    def _maybe_shard_with_logical(self, inputs, logical_axes):
        return maybe_shard_with_logical(
            inputs, logical_axes, shard_mode=self.config.shard_mode, mesh=self.mesh, rules=self.config.logical_axis_rules
        )

    def _maybe_shard_with_name(self, inputs, sharding_name):
        return maybe_shard_with_name(inputs, sharding_name, shard_mode=self.config.shard_mode)
    
    def shard_dim_by_stages(self, x, dim: int):
        """Helper to physically shard a dimension by 'stage'."""
        if self.mesh is None:
            return x
        # Create a placeholder Pspec
        dims = [PartitionSpec.UNCONSTRAINED] * x.ndim
        dims[dim] = "stage"
        # Convert to PSpec/Sharding
        sharding = NamedSharding(self.mesh, PartitionSpec(*dims))
        return jax.lax.with_sharding_constraint(x, sharding)

    # --- Pipeline Logic ---

    def need_circ_storage(self):
        return (self.config.num_pipeline_repeats > 1 and 
                self.config.num_pipeline_microbatches > self.num_stages * self.forwarding_delay)

    def iterations_to_complete_first_microbatch_one_repeat(self):
        return self.forwarding_delay * (self.num_stages - 1)

    def iterations_to_complete_first_microbatch(self):
        return (self.config.num_pipeline_microbatches * (self.config.num_pipeline_repeats - 1) + 
                self.iterations_to_complete_first_microbatch_one_repeat())

    def get_microbatch_and_repeat_ids(self, loop_iteration):
        """Determines which data and weights are active for the current step."""
        processed = jnp.maximum(loop_iteration - self.forwarding_delay * jnp.arange(self.num_stages), 0)
        microbatch_ids = processed % self.config.num_pipeline_microbatches
        repeat_ids = processed // self.config.num_pipeline_microbatches
        return microbatch_ids, repeat_ids

    def init_loop_state(self, inputs):
        """Initialize pipeline buffers with explicit sharding."""
        # Shape: [num_stages, micro_size, sequence, embed]
        shift = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
        shift = self._maybe_shard_with_logical(shift, self.stages_in_logical)

        if self.config.pipeline_delay_activation_forwarding:
            prev_outputs = jnp.zeros_like(shift)
            prev_outputs = self._maybe_shard_with_logical(prev_outputs, self.stages_in_logical)
        else:
            prev_outputs = None

        # State IO: [num_stages, microbatches/stage, micro_size, sequence, embed]
        state_io = jnp.reshape(inputs, (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:])
        # Important: Shard using output sharding logic
        state_io = maybe_shard_with_name(state_io, self.state_io_sharding, shard_mode=self.config.shard_mode)

        if self.use_circ_storage:
            circ_storage = jnp.zeros((self.num_stages,) + inputs.shape, dtype=inputs.dtype)
            # Use state_io sharding for circular storage as well
            circ_storage = maybe_shard_with_name(circ_storage, self.state_io_sharding, shard_mode=self.config.shard_mode)
            circ_mover = shift 
        else:
            circ_storage = None
            circ_mover = None

        return {
            "state_io": state_io,
            "shift": shift,
            "circ_storage": circ_storage,
            "circ_storage_mover": circ_mover,
            "loop_iteration": jnp.array(0, dtype=jnp.int32),
            "prev_outputs": prev_outputs,
        }

    def get_iteration_inputs(self, loop_iter, state_io, circ_storage, shift):
        state_io_batch_idx = loop_iter % self.microbatches_per_stage
        state_io_slice = state_io[:, state_io_batch_idx]
        
        # Ensure shift is sharded correctly before use
        shift = self._maybe_shard_with_logical(shift, self.stages_in_logical)

        if self.use_circ_storage:
            circ_storage_batch_idx = loop_iter % self.config.num_pipeline_microbatches
            circular_stage_in = circ_storage[:, circ_storage_batch_idx]
        else:
            circular_stage_in = shift

        first_stage_in = jnp.where(
            loop_iter < self.config.num_pipeline_microbatches, 
            state_io_slice, 
            circular_stage_in
        )
        first_stage_in = self._maybe_shard_with_logical(first_stage_in, self.stages_in_logical)

        # Broadcast iota to find stage 0
        is_stage_0 = (jax.lax.broadcasted_iota("int32", shift.shape, 0) == 0)
        
        # Select input: Stage 0 gets new data, others get shifted data
        stages_in = jnp.where(is_stage_0, first_stage_in, shift)
        return self._maybe_shard_with_logical(stages_in, self.stages_in_logical)

    def get_new_loop_state(self, output, loop_state):
        """
        Update buffers using SPMD rotations (ppermute) via shard_map.
        Matches Linen pipeline.py logic.
        """
        old_state_io = loop_state["state_io"]
        old_circ_storage = loop_state["circ_storage"]
        old_circ_storage_mover = loop_state["circ_storage_mover"]
        loop_iteration = loop_state["loop_iteration"]
        old_prev_outputs = loop_state["prev_outputs"]

        # --- SPMD Primitives (Defined inside to access self.mesh/specs) ---
        
        @shard_map(mesh=self.mesh, in_specs=self.stages_in_spec, out_specs=self.stages_in_spec, check_vma=True)
        def _rotate_right(arr):
            stage_size = jax.lax.axis_size("stage")
            perm = [(i, (i + 1) % stage_size) for i in range(stage_size)]
            arr = jax.lax.ppermute(arr, axis_name="stage", perm=perm)
            return arr

        @shard_map(mesh=self.mesh, in_specs=self.stages_in_spec, out_specs=self.stages_in_spec, check_vma=True)
        def _shift_right(arr):
            stage_idx = jax.lax.axis_index("stage")
            stage_size = jax.lax.axis_size("stage")
            perm = [(i, (i + 1) % stage_size) for i in range(stage_size)]
            arr = jax.lax.ppermute(arr, axis_name="stage", perm=perm)
            return jnp.where(stage_idx == 0, jnp.zeros_like(arr), arr)

        def _update_shift(output_in):
            if self.config.num_pipeline_repeats == 1 or self.use_circ_storage:
                return _shift_right(output_in) # Last stage doesn't send to first immediately
            else:
                return _rotate_right(output_in) # Circular: Last stage sends to first

        # Update Shift / Prev Outputs
        if self.config.pipeline_delay_activation_forwarding:
            new_shift = _update_shift(old_prev_outputs)
            new_prev_outputs = output
        else:
            new_shift = _update_shift(output)
            new_prev_outputs = None

        # Update Circular Storage
        if self.use_circ_storage:
            # Rotate the mover logic
            # We use a slightly modified rotation helper here usually, 
            # but _rotate_right defined above works on the [stages, ...] shape.
            
            def _rotate_right_and_update(circ_mover_in, circ_storage_in):
                rotated = _rotate_right(circ_mover_in)
                rotated = jnp.expand_dims(rotated, 1) # Add microbatch dim
                
                offset = (loop_iteration - self.iterations_to_complete_first_microbatch_one_repeat() - 1) % self.config.num_pipeline_microbatches
                return jax.lax.dynamic_update_slice_in_dim(circ_storage_in, rotated, offset, axis=1)

            new_circ_storage = _rotate_right_and_update(old_circ_storage_mover, old_circ_storage)
            new_circ_storage_mover = output
        else:
            new_circ_storage = None
            new_circ_storage_mover = None

        # Update State IO (Left Shift Logic)
        stream_buf_idx = loop_iteration % self.microbatches_per_stage
        stream_slice = old_state_io[:, stream_buf_idx]

        @shard_map(mesh=self.mesh, in_specs=self.stages_in_spec, out_specs=self.stages_in_spec, check_vma=True)
        def _rotate_left(arr):
            stage_size = jax.lax.axis_size("stage")
            perm = [(i, (i - 1) % stage_size) for i in range(stage_size)]
            return jax.lax.ppermute(arr, axis_name="stage", perm=perm)

        @shard_map(mesh=self.mesh, in_specs=(self.state_io_spec, self.stages_in_spec, self.stages_in_spec, PartitionSpec()), out_specs=self.state_io_spec)
        def _update_state_io(state_in, stream_slice_in, output_in, idx):
            stage_size = jax.lax.axis_size("stage")
            stage_idx = jax.lax.axis_index("stage")
            
            # Rotate current slice left
            shifted = _rotate_left(stream_slice_in)
            # Last stage gets the new output
            new_val = jnp.where(stage_idx == stage_size - 1, output_in, shifted)
            
            new_val = jnp.expand_dims(new_val, 1)
            return jax.lax.dynamic_update_slice_in_dim(state_in, new_val, idx, axis=1)

        new_state_io = _update_state_io(old_state_io, stream_slice, output, stream_buf_idx)

        return {
            "state_io": new_state_io,
            "shift": new_shift,
            "circ_storage": new_circ_storage,
            "circ_storage_mover": new_circ_storage_mover,
            "loop_iteration": loop_iteration + 1,
            "prev_outputs": new_prev_outputs,
        }

    def permute_output_micro_per_stage_dim(self, output):
        idx0 = self.iterations_to_complete_first_microbatch() % self.microbatches_per_stage
        perm = (np.arange(self.microbatches_per_stage) + idx0) % self.microbatches_per_stage
        return output[:, perm]

    def get_pipeline_remat_policy(self):
        """Returns the JAX rematerialization policy."""
        if self.config.remat_policy == "custom":
            return self.remat_policy
        
        save_input_policy = jax.checkpoint_policies.save_only_these_names("iteration_input", "decoder_layer_input")
        
        if self.remat_policy is not None:
            return jax.checkpoint_policies.save_from_both_policies(self.remat_policy, save_input_policy)
        return save_input_policy

    def __call__(self, inputs, segment_ids=None, positions=None, deterministic=False, model_mode=MODEL_MODE_TRAIN):
        # 1. Input Reshaping
        inputs = jnp.asarray(inputs).reshape((
            self.config.num_pipeline_microbatches,
            self.pipeline_microbatch_size,
            self.config.max_target_length,
            self.config.emb_dim,
        ))
        
        # Shard initial input inputs
        if self.input_sharding:
             inputs = self._maybe_shard_with_name(inputs, self.input_sharding)

        if segment_ids is not None:
            segment_ids = jnp.asarray(segment_ids).reshape((
                self.config.num_pipeline_microbatches,
                self.pipeline_microbatch_size,
                self.config.max_target_length
            ))
        
        if positions is not None:
            positions = jnp.asarray(positions).reshape((
                self.config.num_pipeline_microbatches,
                self.pipeline_microbatch_size,
                self.config.max_target_length
            ))

        # 2. Split State
        layers_def, layers_state = nnx.split(self.layers)
        params_state, metrics_state, remainder_state = layers_state.split(nnx.Param, InternalMetrics, ...)
        
        rng_def, rng_state = nnx.split(self.rngs)

        # 3. Prepare Scan Carry
        scan_carry = {
            "loop_state": self.init_loop_state(inputs),
            "metrics_state": to_pure_dict(metrics_state),
            "rng_state": to_pure_dict(rng_state),
        }

        # Convert static params to pure dicts for closure/vectorization
        params_pure_dict = to_pure_dict(params_state)
        remainder_pure_dict = to_pure_dict(remainder_state)

        # 4. Define Scan Function
        def scan_fn(carry, _):
            l_state = carry["loop_state"]
            loop_iter = l_state["loop_iteration"]
            
            micro_ids, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iter)
            
            # Input Gathering
            it_inputs = self.get_iteration_inputs(loop_iter, l_state["state_io"], l_state["circ_storage"], l_state["shift"])
            it_inputs = jax.ad_checkpoint.checkpoint_name(it_inputs, "iteration_input")
            
            it_pos = jnp.take(positions, micro_ids, axis=0) if positions is not None else None
            it_seg = jnp.take(segment_ids, micro_ids, axis=0) if segment_ids is not None else None
            
            # Explicitly shard gathered metadata inputs by stage
            if it_pos is not None: it_pos = self.shard_dim_by_stages(it_pos, 0)
            if it_seg is not None: it_seg = self.shard_dim_by_stages(it_seg, 0)

            # RNG Handling
            it_rngs = nnx.merge(rng_def, nnx.State(carry["rng_state"]))
            vmap_rngs_obj = it_rngs.fork(split=self.num_stages)
            _, next_rng_state = nnx.split(it_rngs)
            _, vmap_rng_state = nnx.split(vmap_rngs_obj)

            # Stage Index Selection
            stage_indices = jnp.arange(self.num_stages)
            target_indices = (stage_indices if self.config.num_pipeline_repeats <= 1 
                              else (repeat_ids * self.num_stages + stage_indices))

            # Gather Active Weights & Metrics
            active_params = jax.tree_util.tree_map(lambda x: x[target_indices], params_pure_dict)
            active_metrics = jax.tree_util.tree_map(lambda x: x[target_indices], carry["metrics_state"])
            active_remainder = jax.tree_util.tree_map(lambda x: x[target_indices], remainder_pure_dict)

            # Run Stage
            def run_stage(p_raw, m_raw, r_raw, x, seg, pos, r_keys):
                # Reconstruct module
                m = nnx.merge(layers_def, nnx.State(p_raw), nnx.State(m_raw), nnx.State(r_raw))
                nnx.update(m, nnx.State(r_keys))
                
                # Execute
                out, _ = m(x, decoder_segment_ids=seg, decoder_positions=pos, deterministic=deterministic, model_mode=model_mode)
                
                # Split updated metrics
                _, _, final_metrics, _ = nnx.split(m, nnx.Param, InternalMetrics, ...)
                return out, to_pure_dict(final_metrics)

            # VMAP Execution
            stages_out, updated_metrics = nnx.vmap(run_stage)(
                active_params, active_metrics, active_remainder, it_inputs, it_seg, it_pos, to_pure_dict(vmap_rng_state)
            )

            # Update Metrics State
            new_metrics_state = jax.tree_util.tree_map(
                lambda full, sub: full.at[target_indices].set(sub), 
                carry["metrics_state"], 
                updated_metrics
            )

            new_carry = {
                "loop_state": self.get_new_loop_state(stages_out, l_state),
                "metrics_state": new_metrics_state,
                "rng_state": to_pure_dict(next_rng_state),
            }
            return new_carry, None

        # 5. Execute Scan
        policy = self.get_pipeline_remat_policy()
        scannable_fn = (jax.checkpoint(scan_fn, policy=policy) 
                        if self.config.set_remat_policy_on_pipeline_iterations else scan_fn)
        
        total_steps = (self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats) + \
                      self.forwarding_delay * (self.num_stages - 1)
        
        final_carry, _ = jax.lax.scan(scannable_fn, scan_carry, None, length=total_steps)

        # 6. Reconstruct & Return
        nnx.update(self.layers, nnx.State(final_carry["metrics_state"]))
        nnx.update(self.rngs, nnx.State(final_carry["rng_state"]))
        
        out = self.permute_output_micro_per_stage_dim(final_carry["loop_state"]["state_io"])
        
        # Final reshape & sharding
        final_output = jnp.reshape(out, (
            self.config.micro_batch_size_to_train_on, 
            self.config.max_target_length, 
            self.config.emb_dim
        ))
        
        if self.output_sharding:
             final_output = self._maybe_shard_with_name(final_output, self.output_sharding)
             
        return final_output