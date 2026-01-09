"""PipelineParallelismModule for MaxText using Flax NNX.
Native NNX Vectorized version for memory/speed efficiency.
"""

import functools
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.shard_map import shard_map

from flax import nnx
from flax import linen as nn_linen 

from MaxText.common_types import Config, MODEL_MODE_TRAIN, EP_AS_CONTEXT, ShardMode
from MaxText.sharding import (
    maybe_shard_with_logical,
    maybe_shard_with_name,
    create_sharding,
    logical_to_mesh_axes,
    logical_to_mesh,
)

# --- Helpers ---

class PipelineVar(nnx.Param):
    """Container for stacked pipeline weights/states."""
    pass

class InternalMetrics(nnx.Variable):
    """Variable type for diagnostic activation metrics."""
    pass

def cast_to_dtype(node, dtype):
    """Recursively casts all floating point arrays in a Pytree/State to the target dtype."""
    def _cast(leaf):
        if isinstance(leaf, (jax.Array, jnp.ndarray)) and jnp.issubdtype(leaf.dtype, jnp.floating):
            return leaf.astype(dtype)
        return leaf
    return jax.tree_util.tree_map(_cast, node)

def to_pure_dict(x):
    """Recursively converts nnx.State into a plain Python dict."""
    if hasattr(x, "items") and not isinstance(x, (jnp.ndarray, jax.Array)):
        return {k: to_pure_dict(v) for k, v in x.items()}
    return x

# --- NNX Pipeline Module ---

class Pipeline(nnx.Module):
    def __init__(
        self, 
        layers: nnx.Module, 
        config: Config, 
        mesh: Mesh, 
        remat_policy: Any = None, 
        rngs: nnx.Rngs | None = None
    ):
        self.config = config
        self.mesh = mesh
        self.remat_policy = remat_policy
        
        # Dimensions
        self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
        self.forwarding_delay = 2 if self.config.pipeline_delay_activation_forwarding else 1
        self.pipeline_microbatch_size = self.config.micro_batch_size_to_train_on // self.config.num_pipeline_microbatches
        self.microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages

        num_repeats = self.config.num_pipeline_repeats if self.config.num_pipeline_repeats > 1 else 1
        self.total_instances = num_repeats * self.num_stages
        self.use_circ_storage = self.need_circ_storage()

        # Logical Axis Setup
        if self.config.expert_shard_attention_option == EP_AS_CONTEXT:
            self.batch_axis_name = "activation_batch_no_exp"
            self.seq_len_axis_name = "activation_length"
        else:
            self.batch_axis_name = "activation_batch"
            self.seq_len_axis_name = "activation_length_no_exp"

        self.spmd_axis_name = "stage" if self.config.shard_mode == ShardMode.AUTO else None

        # Pre-calculate Sharding Specs
        self.stages_in_logical = ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed")
        self.stages_in_spec = logical_to_mesh_axes(self.stages_in_logical, self.mesh, rules=self.config.logical_axis_rules)
        self.state_io_logical = ("activation_stage", None, self.batch_axis_name, self.seq_len_axis_name, "activation_embed")
        self.state_io_spec = logical_to_mesh_axes(self.state_io_logical, self.mesh, rules=self.config.logical_axis_rules)
        
        self.state_io_sharding = NamedSharding(self.mesh, self.state_io_spec) if self.config.shard_mode == ShardMode.EXPLICIT else None
        self.input_sharding = create_sharding(self.mesh, (None, self.batch_axis_name, self.seq_len_axis_name, "activation_embed"), rules=self.config.logical_axis_rules) if self.config.shard_mode == ShardMode.EXPLICIT else None
        self.output_sharding = create_sharding(self.mesh, (self.batch_axis_name, self.seq_len_axis_name, "activation_embed"), rules=self.config.logical_axis_rules) if self.config.shard_mode == ShardMode.EXPLICIT else None

        if rngs is None:
            raise ValueError("Pipeline requires 'rngs' to initialize stage parameters.")

        # --- INITIALIZATION ---
        LayerCls = type(layers)
        kwargs = {}
        for attr in ['decoder_layer', 'num_decoder_layers', 'quant', 'model_mode', 'scan_layers']:
            if hasattr(layers, attr):
                kwargs[attr] = getattr(layers, attr)

        # 1. Rules Setup
        if self.mesh is not None:
            mesh_identity_rules = [(name, name) for name in self.mesh.axis_names]
            aug_rules = list(self.config.logical_axis_rules) + mesh_identity_rules
        else:
            aug_rules = []

        # 2. Key Generation (Sharded)
        root_key = rngs.params()
        keys = jax.random.split(root_key, self.total_instances)
        
        # 3. Template & Specs (Abstract Evaluation)
        # We create a lightweight template to get GraphDef and Specs
        dummy_rng = nnx.Rngs(params=keys[0])
        template_module = LayerCls(config=self.config, mesh=self.mesh, rngs=dummy_rng, **kwargs)
        StateFilter = (nnx.Variable,)
        self.graphdef, _ = nnx.split(template_module, StateFilter)
        
        # Get logical partition specs from the template
        if self.mesh is not None:
            logical_specs = nnx.get_partition_spec(template_module)
            # Create a PyTree of PartitionSpecs matching the state structure
            # We filter specs to only match Params (RNGs/Metrics don't get specs usually)
            state_specs = nnx.state(logical_specs, nnx.Param)
        else:
            state_specs = None

        # 4. Prepare for Shard Map
        # We need to know the output specs for shard_map.
        # We construct them by resolving the logical specs to physical specs.
        def unwrap_spec_axis(axis):
            if isinstance(axis, (list, tuple)): return axis[0]
            return axis

        def resolve_output_spec(val, logical_spec):
            if logical_spec is None:
                # If no spec, it's just sharded on 'stage' (which shard_map handles implicitly)
                # Inner spec is unconstrained (P())
                return PartitionSpec() 
            
            # Unwrap tuple specs
            clean_spec = PartitionSpec(*[unwrap_spec_axis(ax) for ax in logical_spec])
            
            # Map logical -> mesh axes (e.g. 'fsdp' -> 'fsdp')
            # logical_to_mesh_axes returns a PartitionSpec
            mesh_spec = logical_to_mesh_axes(clean_spec, self.mesh, aug_rules)
            return mesh_spec

        # Use eval_shape to get the abstract state structure without allocating
        abstract_state = jax.eval_shape(lambda: nnx.state(template_module, StateFilter))
        
        # Only Params have specs; fill others with None
        full_specs_tree = abstract_state.filter(nnx.Param).merge(state_specs) if state_specs else abstract_state
        
        # Construct the output specs tree for shard_map
        # If mesh is None, everything is unconstrained P()
        if self.mesh is not None:
            # We map over the abstract state (values) and the state_specs (logical specs)
            # Use 'state_specs' where available, else None
            
            # Align structures: abstract_state has everything. state_specs has Params.
            # We iterate over abstract_state. If key in state_specs, use it.
            
            def map_spec(path, val):
                # Retrieve spec from state_specs if it exists
                # This is tricky with PyTrees. Simpler to assume strict alignment if we filter.
                pass 
            
            # Better approach: Map over the abstract_state and look up in state_specs
            # Since state_specs is a State object, we can use it as a dict lookup if needed,
            # or tree_map with keeping None for missing.
            
            # Let's trust tree_map to handle structure mismatch by treating missing as None? No.
            # We'll use a wrapper that defaults to None.
            
            # Create a full tree of specs matching abstract_state
            full_logical_specs = jax.tree.map(lambda x: None, abstract_state)
            if state_specs:
                full_logical_specs = full_logical_specs.merge(state_specs)
            
            out_specs = jax.tree.map(resolve_output_spec, abstract_state, full_logical_specs)
        else:
            out_specs = jax.tree.map(lambda x: PartitionSpec(), abstract_state)

        # 5. SHARD MAP INITIALIZATION (The OOM Solution)
        # This function runs on each device individually.
        def create_layer_shard(shard_keys):
            # shard_keys has shape [num_repeats_per_shard, ...] (likely just 1 or num_repeats)
            # We vmap over the local chunk of keys
            
            def create_single(k):
                stage_rngs = nnx.Rngs(params=k)
                m = LayerCls(config=self.config, mesh=self.mesh, rngs=stage_rngs, **kwargs)
                
                # Initialize Params (Dummy Pass)
                m(
                    jnp.zeros((1, 1, self.config.emb_dim), dtype=self.config.dtype),
                    jnp.zeros((1, 1), dtype=jnp.int32),
                    jnp.zeros((1, 1), dtype=jnp.int32),
                    deterministic=False,
                    model_mode=MODEL_MODE_TRAIN,
                )
                
                _, state = nnx.split(m)
                casted_state = cast_to_dtype(state, self.config.dtype)
                return casted_state

            return jax.vmap(create_single)(shard_keys)

        # Prepare inputs for shard_map
        # keys must be reshaped to [num_stages, num_repeats, ...] so we can shard on axis 0
        keys_reshaped = keys.reshape((self.num_stages, num_repeats) + keys.shape[1:])
        
        # Run shard_map
        # Input specs: Shard 0-th dim on 'stage'
        in_specs = PartitionSpec('stage', None)
        # Output specs: Shard 0-th dim on 'stage', inner dims per out_specs
        # We need to construct P('stage', *inner) for every leaf
        
        def prepend_stage(spec):
            return PartitionSpec('stage', *spec)
        
        shard_map_out_specs = jax.tree.map(prepend_stage, out_specs)

        # EXECUTE
        if self.mesh is not None:
            full_stacked_state = shard_map(
                create_layer_shard,
                mesh=self.mesh,
                in_specs=in_specs,
                out_specs=shard_map_out_specs,
                check_rep=True
            )(keys_reshaped)
        else:
            # Fallback if no mesh (CPU/Test)
            full_stacked_state = jax.vmap(create_layer_shard)(keys_reshaped)

        # Flatten the [Stage, Repeats] dims back to [Total_Instances]
        # Current shape: [Stages, Repeats, ...] -> [Stages*Repeats, ...]
        def flatten_leading_dims(x):
            return x.reshape((self.total_instances,) + x.shape[2:])
            
        full_stacked_state = jax.tree.map(flatten_leading_dims, full_stacked_state)

        # 6. DECOMPOSE AND STORE
        flat_values, tree_def = jax.tree.flatten(full_stacked_state)
        self.state_treedef = tree_def
        self.storage_vars = nnx.List([PipelineVar(v) for v in flat_values])

    # --- Helper Methods ---

    def need_circ_storage(self):
        return (self.config.num_pipeline_repeats > 1 and 
                self.config.num_pipeline_microbatches > self.num_stages * self.forwarding_delay)

    def iterations_to_complete_first_microbatch_one_repeat(self):
        return self.forwarding_delay * (self.num_stages - 1)

    def iterations_to_complete_first_microbatch(self):
        return (self.config.num_pipeline_microbatches * (self.config.num_pipeline_repeats - 1) + 
                self.iterations_to_complete_first_microbatch_one_repeat())

    def get_pipeline_remat_policy(self):
        if self.config.remat_policy == "custom": return self.remat_policy
        save_input = jax.checkpoint_policies.save_only_these_names("iteration_input", "decoder_layer_input")
        if self.remat_policy is not None:
            return jax.checkpoint_policies.save_from_both_policies(self.remat_policy, save_input)
        return save_input

    def get_microbatch_and_repeat_ids(self, loop_iteration):
        processed = jnp.maximum(loop_iteration - self.forwarding_delay * jnp.arange(self.num_stages), 0)
        return processed % self.config.num_pipeline_microbatches, processed // self.config.num_pipeline_microbatches

    def _maybe_shard_with_logical(self, inputs, logical_axes):
        return maybe_shard_with_logical(
            inputs, logical_axes, shard_mode=self.config.shard_mode, mesh=self.mesh, rules=self.config.logical_axis_rules
        )

    def _maybe_shard_with_name(self, inputs, sharding_name):
        return maybe_shard_with_name(inputs, sharding_name, shard_mode=self.config.shard_mode)

    def shard_dim_by_stages(self, x, dim: int):
        if self.mesh is None: return x
        dims = [PartitionSpec.UNCONSTRAINED] * x.ndim
        dims[dim] = "stage"
        sharding = NamedSharding(self.mesh, PartitionSpec(*dims))
        return jax.lax.with_sharding_constraint(x, sharding)

    def get_weight_sharding(self, *args, **kwargs):
        def get_spec(var):
            leaf = var.value
            if hasattr(leaf, 'sharding') and isinstance(leaf.sharding, NamedSharding):
                return leaf.sharding.spec
            return None
        
        current_arrays = [v.value for v in self.storage_vars]
        temp_state = jax.tree.unflatten(self.state_treedef, current_arrays)
        return jax.tree.map(get_spec, temp_state)

    def all_gather_over_fsdp_list(self):
        gathered_arrays = []
        def _strip_spec(spec):
            return PartitionSpec(*[x for x in spec if x not in ('fsdp', 'fsdp_transpose')])

        for var in self.storage_vars:
            leaf = var.value
            if hasattr(leaf, 'sharding') and isinstance(leaf.sharding, NamedSharding):
                new_spec = _strip_spec(leaf.sharding.spec)
                target = NamedSharding(leaf.sharding.mesh, new_spec)
                gathered_arrays.append(jax.lax.with_sharding_constraint(leaf, target))
            else:
                gathered_arrays.append(leaf)
        return gathered_arrays

    def init_loop_state(self, inputs):
        shift = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
        shift = self._maybe_shard_with_logical(shift, self.stages_in_logical)

        prev_outputs = jnp.zeros_like(shift) if self.config.pipeline_delay_activation_forwarding else None
        if prev_outputs is not None:
            prev_outputs = self._maybe_shard_with_logical(prev_outputs, self.stages_in_logical)

        state_io = jnp.reshape(inputs, (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:])
        if self.state_io_sharding is not None:
            state_io = maybe_shard_with_name(state_io, self.state_io_sharding, shard_mode=self.config.shard_mode)

        if self.use_circ_storage:
            circ_storage = jnp.zeros((self.num_stages,) + inputs.shape, dtype=inputs.dtype)
            if self.state_io_sharding is not None:
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
        shift = self._maybe_shard_with_logical(shift, self.stages_in_logical)

        if self.use_circ_storage:
            circ_storage_batch_idx = loop_iter % self.config.num_pipeline_microbatches
            circular_stage_in = circ_storage[:, circ_storage_batch_idx]
        else:
            circular_stage_in = shift

        first_stage_in = jnp.where(loop_iter < self.config.num_pipeline_microbatches, state_io_slice, circular_stage_in)
        first_stage_in = self._maybe_shard_with_logical(first_stage_in, self.stages_in_logical)
        
        stages_in = jnp.where(jax.lax.broadcasted_iota("int32", shift.shape, 0) == 0, first_stage_in, shift)
        return self._maybe_shard_with_logical(stages_in, self.stages_in_logical)
    
    def get_new_loop_state(self, output, loop_state):
        old_state_io = loop_state["state_io"]
        old_circ_storage = loop_state["circ_storage"]
        old_circ_storage_mover = loop_state["circ_storage_mover"]
        loop_iteration = loop_state["loop_iteration"]
        old_prev_outputs = loop_state["prev_outputs"]

        @functools.partial(shard_map, mesh=self.mesh, in_specs=self.stages_in_spec, out_specs=self.stages_in_spec, check_rep=True)
        def _rotate_right(arr):
            stage_size = jax.lax.axis_size("stage")
            perm = [(i, (i + 1) % stage_size) for i in range(stage_size)]
            return jax.lax.ppermute(arr, axis_name="stage", perm=perm)

        @functools.partial(shard_map, mesh=self.mesh, in_specs=self.stages_in_spec, out_specs=self.stages_in_spec, check_rep=True)
        def _shift_right(arr):
            stage_idx = jax.lax.axis_index("stage")
            stage_size = jax.lax.axis_size("stage")
            perm = [(i, (i + 1) % stage_size) for i in range(stage_size)]
            arr = jax.lax.ppermute(arr, axis_name="stage", perm=perm)
            return jnp.where(stage_idx == 0, jnp.zeros_like(arr), arr)

        def _update_shift(output_in):
            if self.config.num_pipeline_repeats == 1 or self.use_circ_storage:
                return _shift_right(output_in)
            return _rotate_right(output_in)

        new_shift = _update_shift(old_prev_outputs) if self.config.pipeline_delay_activation_forwarding else _update_shift(output)
        new_prev_outputs = output if self.config.pipeline_delay_activation_forwarding else None

        if self.use_circ_storage:
            def _rotate_right_and_update(circ_mover_in, circ_storage_in):
                rotated = _rotate_right(circ_mover_in)
                rotated = jnp.expand_dims(rotated, 1)
                offset = (loop_iteration - self.iterations_to_complete_first_microbatch_one_repeat() - 1) % self.config.num_pipeline_microbatches
                return jax.lax.dynamic_update_slice_in_dim(circ_storage_in, rotated, offset, axis=1)

            new_circ_storage = _rotate_right_and_update(old_circ_storage_mover, old_circ_storage)
            new_circ_storage_mover = output
        else:
            new_circ_storage = None
            new_circ_storage_mover = None

        stream_buf_idx = loop_iteration % self.microbatches_per_stage
        stream_slice = old_state_io[:, stream_buf_idx]

        @functools.partial(shard_map, mesh=self.mesh, in_specs=self.stages_in_spec, out_specs=self.stages_in_spec, check_rep=True)
        def _rotate_left(arr):
            stage_size = jax.lax.axis_size("stage")
            perm = [(i, (i - 1) % stage_size) for i in range(stage_size)]
            return jax.lax.ppermute(arr, axis_name="stage", perm=perm)

        @functools.partial(shard_map, 
                           mesh=self.mesh, 
                           in_specs=(self.state_io_spec, self.stages_in_spec, self.stages_in_spec, PartitionSpec()), 
                           out_specs=self.state_io_spec,
                           check_rep=True)
        def _update_state_io(state_in, stream_slice_in, output_in, idx):
            stage_size = jax.lax.axis_size("stage")
            stage_idx = jax.lax.axis_index("stage")
            perm = [(i, (i - 1) % stage_size) for i in range(stage_size)]
            shifted = jax.lax.ppermute(stream_slice_in, axis_name="stage", perm=perm)
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

    def __call__(self, inputs, segment_ids=None, positions=None, deterministic=False, model_mode=MODEL_MODE_TRAIN, partition_spec=None):
        # 1. Input Setup
        inputs = jnp.asarray(inputs).reshape((
            self.config.num_pipeline_microbatches, 
            self.pipeline_microbatch_size, 
            self.config.max_target_length, 
            self.config.emb_dim
        ))
        
        if self.input_sharding:
            inputs = self._maybe_shard_with_name(inputs, self.input_sharding)
        
        ag_sharding = NamedSharding(self.mesh, PartitionSpec(None, None))
        if positions is not None:
            positions = jax.lax.with_sharding_constraint(jnp.asarray(positions), ag_sharding).reshape((self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length))
        if segment_ids is not None:
            segment_ids = jax.lax.with_sharding_constraint(jnp.asarray(segment_ids), ag_sharding).reshape((self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length))

        # 2. RECONSTRUCT STATE
        if self.config.pipeline_fsdp_ag_once: 
            current_arrays = self.all_gather_over_fsdp_list()
        else:
            current_arrays = [v.value for v in self.storage_vars]
        
        current_stacked_state = jax.tree.unflatten(self.state_treedef, current_arrays)

        loop_state = self.init_loop_state(inputs)

        # 3. SCAN FN
        def scan_fn(carry, _):
            curr_loop, curr_stacked = carry
            loop_iter = curr_loop["loop_iteration"]
            
            stages_inputs = self.get_iteration_inputs(loop_iter, curr_loop["state_io"], curr_loop["circ_storage"], curr_loop["shift"])
            stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, "iteration_input")
            
            micro_ids, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iter)
            
            s_pos = positions[micro_ids] if positions is not None else None
            s_seg = segment_ids[micro_ids] if segment_ids is not None else None
            if s_pos is not None: s_pos = self.shard_dim_by_stages(s_pos, 0)
            if s_seg is not None: s_seg = self.shard_dim_by_stages(s_seg, 0)

            stage_indices = jnp.arange(self.num_stages)
            target_indices = stage_indices 
            if self.config.num_pipeline_repeats > 1:
                target_indices = repeat_ids * self.num_stages + stage_indices
            
            # Select active weights
            current_states = jax.tree.map(lambda leaf: leaf[target_indices], curr_stacked)

            def run_layer(state, x, seg, pos):
                # Merge logic
                model = nnx.merge(self.graphdef, state)
                out = model(x, decoder_segment_ids=seg, decoder_positions=pos, deterministic=deterministic, model_mode=model_mode)
                
                # Capture Updates
                _, new_state = nnx.split(model)
                return out, new_state

            in_axes_seg = 0 if s_seg is not None else None
            in_axes_pos = 0 if s_pos is not None else None
            
            # Run VMAP
            stages_out, updated_active_states = jax.vmap(run_layer, in_axes=(0, 0, in_axes_seg, in_axes_pos))(
                current_states, stages_inputs, s_seg, s_pos
            )

            # Scatter Updates
            def update_slice(full_tensor, updates):
                return full_tensor.at[target_indices].set(updates)
            
            next_stacked = jax.tree.map(update_slice, curr_stacked, updated_active_states)

            if self.config.scan_layers and isinstance(stages_out, tuple):
                stages_out = stages_out[0]

            next_loop = self.get_new_loop_state(stages_out, curr_loop)
            return (next_loop, next_stacked), None

        # 4. EXECUTE SCAN
        total_steps = (self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats) + self.forwarding_delay * (self.num_stages - 1)
        policy = self.get_pipeline_remat_policy() if self.config.set_remat_policy_on_pipeline_iterations else None
        
        init_carry = (loop_state, current_stacked_state)

        if self.config.scan_pipeline_iterations:
             scan_fn = jax.checkpoint(scan_fn, policy=policy, prevent_cse=not self.config.scan_pipeline_iterations)
             (final_loop, final_stacked), _ = jax.lax.scan(scan_fn, init_carry, None, length=total_steps)
        else:
             curr = init_carry
             scan_fn = jax.checkpoint(scan_fn, policy=policy) if policy else scan_fn
             for _ in range(total_steps): curr, _ = scan_fn(curr, None)
             (final_loop, final_stacked) = curr
        
        # 5. PERSIST UPDATES
        if not self.config.pipeline_fsdp_ag_once:
             final_flat_values, _ = jax.tree.flatten(final_stacked)
             for var, new_val in zip(self.storage_vars, final_flat_values):
                 var.value = new_val

        out = self.permute_output_micro_per_stage_dim(final_loop["state_io"])
        final_output = jnp.reshape(out, (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim))
        
        if self.output_sharding:
            final_output = self._maybe_shard_with_name(final_output, self.output_sharding)

        return final_output