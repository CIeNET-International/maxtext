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

pipeline_n3_debug.py -- HEAVILY INSTRUMENTED copy of pipeline_n3.py.

Computation is IDENTICAL to pipeline_n3.py. Only logging/prints/asserts are
added so the execution flow (trace-time shapes + runtime values) can be
understood from logs alone when running on a TPU VM.

Logging conventions:
  - print()            -> trace-time info (shapes/dtypes/tree structure),
                          visible in compile logs as the jaxpr is built.
  - jax.debug.print()  -> runtime values (actual numbers), visible during
                          execution.
  - _n3log             -> the "PIPELINE_N3" logger (mirrors print() messages).

pipeline_n3.py -- NNXCircularPipeline with 2-level custom_vjp + linear_transpose,
matching the Linen golden pattern from pipeline_utils.py.

Differences from pipeline.py:
  - NNXCircularPipeline.__call__ rewritten with:
    * Outer custom_vjp (execute_stage): manages BSW + linear_transpose for
      reduce-scatter dual of weight_prefetching all-gather.
    * Inner custom_vjp (run_microbatch): per-microbatch remat with d+g gradient
      accumulation on BSW.
  - `import functools` added.
  - No nn.scan, nn.remat, nn.Module, flax.core.lift used.
"""

from typing import Any

import functools
import numpy as np

from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import jax
import jax.ad_checkpoint

from aqt.jax.v2 import aqt_tensor
from flax import linen as nn
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

# ---------------------------------------------------------------------------
# PIPELINE_N3 DEBUG INSTRUMENTATION
# ---------------------------------------------------------------------------
import logging as _logging

_n3log = _logging.getLogger("PIPELINE_N3")
_n3log.setLevel(_logging.DEBUG)
if not _n3log.handlers:
  _h = _logging.StreamHandler()
  _h.setFormatter(_logging.Formatter("[PIPELINE_N3] %(message)s"))
  _n3log.addHandler(_h)


def _n3_log(msg):
  """Emit a trace-time message via both print() and the PIPELINE_N3 logger.

  print() guarantees the line shows up on stdout/compile logs even if logging
  is reconfigured by the host; the logger keeps a consistent prefix.
  """
  print(f"[PIPELINE_N3] {msg}")
  _n3log.debug(msg)


def _byte_size(tree):
  """Total bytes of all array leaves in a pytree (size * dtype.itemsize)."""
  leaves = jax.tree.leaves(tree)
  total = 0
  for l in leaves:
    if hasattr(l, "size") and hasattr(l, "dtype"):
      try:
        total += l.size * l.dtype.itemsize
      except Exception:  # pylint: disable=broad-except
        pass
  return total


def _leaf_count(tree, is_leaf=None):
  """Number of leaves in a pytree (optionally with a custom is_leaf)."""
  if is_leaf is None:
    return len(jax.tree.leaves(tree))
  return len(jax.tree.leaves(tree, is_leaf=is_leaf))


def _shapes(tree, limit=5, is_leaf=None):
  """Return a short list of (shape, dtype) strings for the first `limit` leaves."""
  if is_leaf is None:
    leaves = jax.tree.leaves(tree)
  else:
    leaves = jax.tree.leaves(tree, is_leaf=is_leaf)
  out = []
  for l in leaves[:limit]:
    shp = getattr(l, "shape", None)
    dt = getattr(l, "dtype", None)
    out.append(f"{tuple(shp) if shp is not None else l}:{dt}")
  return out


def _tree_keys(d):
  """Best-effort list of top-level keys for a dict-like loop_state."""
  try:
    return list(d.keys())
  except Exception:  # pylint: disable=broad-except
    return f"<non-dict {type(d).__name__}>"


def _is_static_param(path, v):
  """Predicate matching nnx.Param and FP8 _overwrite_with_gradient variables.

  Used throughout the pipeline to split state into trainable params vs other state.
  Must be consistent everywhere to prevent tree structure mismatches.
  """
  return isinstance(v, nnx.Param) or type(v).__name__ == "_overwrite_with_gradient"


def _advance_rng_state(state, iteration):
  """Fold loop_iteration into all RNG keys to produce unique dropout masks per scan step.

  jax.lax.scan has no split_rngs mechanism (unlike Linen's nn.scan), so every
  iteration would otherwise see the same dropout mask. This mirrors the effect
  of ``nn.scan(split_rngs={"random": True})`` from the Linen pipeline.

  Only typed PRNG key variables (``RngKey``) are folded. RNG counters
  (``RngCount``) are uint32 arrays and must be left untouched -- calling
  ``jax.random.fold_in`` on raw uint32 data triggers a PRNG-impl shape
  mismatch (e.g. shape ``(N, 2)`` vs ``unsafe_rbg`` expecting ``(4,)``).

  Args:
    state: An ``nnx.State`` (or partition thereof) that may contain
        ``nnx.RngState`` variable entries whose ``.value`` is a JAX PRNG key.
    iteration: A scalar integer (the loop counter) folded into each key via
        ``jax.random.fold_in``.

  Returns:
    A new state with the same tree structure, where every typed PRNG key
    entry has a unique key derived from the original key and *iteration*.
  """

  def _fold_if_rng(x):
    if isinstance(x, nnx.Variable) and issubclass(x.type, nnx.RngState):
      val = x.value
      # Only fold typed PRNG keys (RngKey). Skip uint32 RNG counters
      # (RngCount) -- fold_in would try to wrap them with the default PRNG
      # impl and fail on shape mismatch after vmap batching.
      if jax.dtypes.issubdtype(val.dtype, jax.dtypes.prng_key):
        # fold_in requires a scalar key (shape ()). After nnx.vmap over
        # stages and repeats, keys are batched arrays of shape e.g.
        # (num_repeats, num_stages). Nest jax.vmap over each batch
        # dimension so fold_in sees individual scalar keys.
        def folded(k):
          return jax.random.fold_in(k, iteration)

        for _ in range(val.ndim):
          folded = jax.vmap(folded)
        return x.replace(value=folded(val))
    return x

  return jax.tree.map(_fold_if_rng, state, is_leaf=lambda x: isinstance(x, nnx.Variable))


def is_spec_leaf(x):
  """Predicate matching leaves in the bsw_pps treedef, which can be either P or None (if no sharding)."""
  return isinstance(x, P) or x is None


class NNXPipelineBase(nnx.Module):
  """
  Base module that implements shared pipelining logic across stages.
  Contains pure JAX and mathematical utilities.
  """

  def _setup_pipeline_attributes(self):
    """Initializes the configuration, calculating num_stages, delay, axes, and partition specs."""
    self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
    self.forwarding_delay = 2 if self.config.pipeline_delay_activation_forwarding else 1
    self.pipeline_microbatch_size = self.config.micro_batch_size_to_train_on // self.config.num_pipeline_microbatches
    self.microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages
    self.use_circ_storage = self.need_circ_storage()

    self.batch_axis_name = "activation_batch"
    self.seq_len_axis_name = "activation_length"
    self.spmd_axis_name = "stage" if self.config.shard_mode == ShardMode.AUTO else None

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

  def need_circ_storage(self):
    return (
        self.config.num_pipeline_repeats > 1
        and self.config.num_pipeline_microbatches > self.num_stages * self.forwarding_delay
    )

  def iterations_to_complete_first_microbatch_one_repeat(self):
    # Return the number of iterations it takes for microbatch 0 to finish a repeat
    return self.forwarding_delay * (self.num_stages - 1)

  def iterations_to_complete_first_microbatch(self):
    # Return the number of iterations it takes for microbatch 0 to finish the last stage of the last repeat
    return (
        self.config.num_pipeline_microbatches * (self.config.num_pipeline_repeats - 1)
        + self.iterations_to_complete_first_microbatch_one_repeat()
    )

  def _maybe_shard_with_logical(self, inputs, logical_axes):
    """Wrapper of maybe_shard_with_logical"""
    return maybe_shard_with_logical(
        inputs,
        logical_axes,
        shard_mode=self.config.shard_mode,
        mesh=self.mesh,
        rules=self.config.logical_axis_rules,
        debug_sharding=self.config.debug_sharding,
        extra_stack_level=1,
    )

  def _maybe_shard_with_name(self, inputs, sharding_name):
    """Wrapper of maybe_shard_with_name"""
    return maybe_shard_with_name(
        inputs,
        sharding_name,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
    )

  def get_iteration_inputs(self, loop_iteration, state_io, circ_storage, shift):
    """
    Construct stages_in: the global array that is operated on for this iteration, shape same as
    shift=[stages, micro_size, sequence, embed]
    This is almost a rotated version of the last outputs, except for the first stage which must grab a new batch from
    state_io or an old one from circ_storage
    """
    # Setup potential input from state_io, which has a rotating microbatch index (size of microbatches_per_stage)
    state_io_batch_idx = loop_iteration % self.microbatches_per_stage
    state_io_slice = state_io[:, state_io_batch_idx]
    shift = self._maybe_shard_with_logical(shift, self.stages_in_logical)

    if self.use_circ_storage:
      # Setup potential input from circ_storage, which also has a rotating index for microbatch,
      # size of num_microbatches
      circ_storage_batch_idx = loop_iteration % self.config.num_pipeline_microbatches
      circular_stage_in = circ_storage[:, circ_storage_batch_idx]
    else:
      # The last stage immediately flows into the first stage, use this rotated shift instead of circular storage
      circular_stage_in = shift

    # For early loop iterations we grab a new input for stage 0 from the state_io. Once each microbatch has left
    # state_io we instead grab from the last stage's output (possibly buffered when num_microbatches > num_stages, e.g.
    # from circ_storage).
    first_stage_in = jnp.where(loop_iteration < self.config.num_pipeline_microbatches, state_io_slice, circular_stage_in)
    first_stage_in = self._maybe_shard_with_logical(first_stage_in, self.stages_in_logical)

    # Note that first_stage_in may correspond to bubble computation during the last few iterations.
    # However, these bubble computation results remain in the shift buffer (do not make it back to state_io) and are
    # thus discarded / not returned.
    # The final returned output is stored in the state_io, which has the appropriate total size of num_microbatches. The
    # state_io will not contain bubble results at the end of the last iteration.

    def select_state_or_input(first_stage_in, shift):
      # Selects input for stage 0, shift for other stages
      return jnp.where(
          jax.lax.broadcasted_iota("int32", shift.shape, 0, out_sharding=self.stages_in_sharding) == 0,
          first_stage_in,
          shift,
      )

    # Selects input (from stream_io) for stage 0, other stages get from shift (the rotated previous output)
    stages_in = select_state_or_input(first_stage_in, shift)
    return self._maybe_shard_with_logical(stages_in, self.stages_in_logical)

  def get_microbatch_and_repeat_ids(self, loop_iteration):
    """Gets the microbatch_ids and repeat_ids for all stages on this loop_iteration. Works for both circular and
    non-circular"""
    # Stage 0 has processed one microbatch every loop_iter, but Stage 1 is 1 behind due to bubble, etc for other stages
    microbatches_processed = jnp.maximum(loop_iteration - self.forwarding_delay * jnp.arange(self.num_stages), 0)
    microbatches_processed = self._maybe_shard_with_name(microbatches_processed, NamedSharding(self.mesh, P("stage")))
    microbatch_ids = microbatches_processed % self.config.num_pipeline_microbatches
    repeat_ids = microbatches_processed // self.config.num_pipeline_microbatches
    return microbatch_ids, repeat_ids

  def get_pipeline_remat_policy(self):
    """Returns the pipeline remat policy for this pipeline.

    Saves two named tensors during jax.checkpoint recomputation:
      - "iteration_input": routed microbatch data entering the decoder
      - "decoder_layer_input": input to the decoder layer itself
    Everything else is recomputed during backward to save memory.
    """
    if self.config.remat_policy == "custom":
      return self.remat_policy
    save_input_policy = jax.checkpoint_policies.save_only_these_names("iteration_input", "decoder_layer_input")
    if self.remat_policy is not None:
      return jax.checkpoint_policies.save_from_both_policies(self.remat_policy, save_input_policy)
    return save_input_policy

  @staticmethod
  def _remove_fsdp_from_physical_partition_spec(pps):
    """Removes 'fsdp' and 'fsdp_transpose' from physical partition spec."""
    if isinstance(pps, P):
      new_spec = []
      # Iterate through each axis in the original PartitionSpec.
      for axis in pps:
        if axis is None:
          new_spec.append(None)
        elif isinstance(axis, str):
          # If the axis is 'fsdp', replace it with None to signify replication.
          if axis not in ("fsdp", "fsdp_transpose"):
            new_spec.append(axis)
          else:
            new_spec.append(None)
        elif isinstance(axis, (list, tuple)):
          # If the axis is a collection, filter out 'fsdp'.
          new_axis = [a for a in axis if a not in ("fsdp", "fsdp_transpose")]
          new_spec.append(tuple(new_axis))
        else:
          raise ValueError(f"Unsupported_axis_type: {type(axis)}")
        # Return a new sharding object with the modified spec.
      return P(*new_spec)
    return pps

  def init_states(self, inputs):
    """Initialize components of state: state_io, shift, circular_storage and circular_storage_mover
    Assumes input has already been reshaped into microbatches: [num_micro_batches, micro_batch_size, sequence, embed]

    Returns a dictionary with properties
      shift: zeros shape [num_stages, micro_size, sequence, embed]
      prev_outputs: same shape as shift, only used when pipeline_delay_activation_forwarding is set to true, else None
      state_io: reshaped inputs [num_stages, microbatches/stages, micro_size, sequence, embed]
      circ_storage: zeros [num_stages, microbatches, micro_size, sequence, embed] when needed, else None
      circ_storage_mover: zeros[num_stages, micro_size, sequence, embed] when needed, else None
      loop_iteration: scalar set initially to 0
      bsw: pytree of identical structure as weights with leaf arrays leading dimension of num_repeats replaced by 2, e.g.
        a leaf of shape [num_repeats, stages, mlp, embed] is mapped to [2, num_stages, mlp, embed].
    """
    # Shift is used to rotate the output of each pipeline into the input of the next
    # shift has shape [num_stages, micro_size, sequence, embed]
    shift = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
    shift = self._maybe_shard_with_logical(shift, self.stages_in_logical)

    # Prev outputs has the same shape of the output (and shift)
    if self.config.pipeline_delay_activation_forwarding:
      prev_outputs = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
      prev_outputs = self._maybe_shard_with_logical(prev_outputs, self.stages_in_logical)
    else:
      prev_outputs = None

    # state_io (state input output) at first holds all of the input batches, but also will hold the outputs
    #   as the pipeline runs/finishes
    # state_io has shape [num_stages, microbatches/stages, micro_size, sequence, embed]
    state_io = jnp.reshape(
        inputs, (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:], out_sharding=self.state_io_sharding
    )

    # We shard the pipeline_microbatch_size axis by data/fsdp, not num_microbatches since those are looped over.
    state_io = self._maybe_shard_with_logical(state_io, self.state_io_logical)

    # circ_storage is used to hold the final pipeline stage outputs before it is used for the next repeat. It is only
    # needed when num_microbatches > num_stages, else instead the final stage will immediately pass to the first without
    # additional storage.
    # circ_storage has shape [num_stages, microbatches, micro_size, sequence, embed].
    # Note that this shape is a factor of num_stages larger than necessary - each stage holds the global batch, but only
    # stage 0 holds the real activations (since it will use them), the rest hold dummy ones. This amount of storage
    # [global_batch, sequence, embed] is fine as long as there is some amount of additional sharding axes, e.g. FSDP,
    # TP, DP (e.g. there are many devices that shard stage 0)
    # We may look into alternatives using less storage if this becomes an issue (ideas in b/347603101).
    # circ_storage_mover is used to push the microbatches from the pipeline into circ_storage with one buffer iteration
    # of delay circ_storage_mover shape is same as shift: [num_stages, micro_size, sequence, embed]
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
        "loop_iteration": 0,
        "prev_outputs": prev_outputs,
    }

  def shard_dim_by_stages(self, x, dim: int, physical_partition_spec: P | None, is_stage_weight: bool = False):
    """Shards x using the provided partition_spec, but adds the "stage" mesh axis to the existing sharding at
    the specified dimension."""
    placeholder = None if self.config.shard_mode == ShardMode.EXPLICIT else P.UNCONSTRAINED
    if physical_partition_spec is None:
      dims_mapping = [placeholder] * x.ndim
    else:
      physical_partition_spec = self._remove_fsdp_from_physical_partition_spec(physical_partition_spec)
      dims_mapping = list(physical_partition_spec)
      # If not a stage weight, we handle the repeat dimension offset
      if not is_stage_weight:
        dims_mapping = [placeholder] * (dim + 1) + dims_mapping[dim:]  # inflat one dimension for num_repeats
    dims_mapping[dim] = "stage"
    dims_mapping = tuple(dims_mapping)
    # We add reduced rule only when pspec is given for a stage weight
    if physical_partition_spec and is_stage_weight and self.config.shard_mode == ShardMode.EXPLICIT:
      batch_mesh_axis = ["data", "fsdp"]
      reduced_mark = [mesh_axis for mesh_axis in batch_mesh_axis if self.mesh.shape[mesh_axis] > 1]
      pspec = P(*dims_mapping, reduced=set(reduced_mark))
    else:
      pspec = P(*dims_mapping)
    sharding = jax.sharding.NamedSharding(self.mesh, pspec)
    return self._maybe_shard_with_name(x, sharding)

  def vmap_parallel_gather(
      self, weights, physical_partition_spec, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights
  ):
    """Use vmap to implement a sharded parallel gather.
    Parallel gather means each stage has its own weights, and gets one slice from it.
    Args:
      weights: Per-stage data to be gathered from.
      repeat_ids: Integer tensor of shape [num_stages], the repeats of the stages.
      repeat_dim_in_weights: The dimension in weights where repeat_ids are applied. The output will not
        have this dimension.
      stages_dim_in_weights: The dimension in weights that represents parallel stages.

    Returns:
      The per-stage gathered values. The shape is weights.shape but with repeat_dim_in_weights
        removed.
    """

    def _gather_one(x, repeat_id):
      return jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, repeat_id, 1, repeat_dim_in_weights), repeat_dim_in_weights)

    gathered_weights_stage_dim = 0
    repeat_ids = self.shard_dim_by_stages(repeat_ids, 0, physical_partition_spec=None)
    weights = self.shard_dim_by_stages(
        weights, stages_dim_in_weights, physical_partition_spec=physical_partition_spec, is_stage_weight=False
    )
    stage_weights = jax.vmap(_gather_one, in_axes=(stages_dim_in_weights, 0), out_axes=gathered_weights_stage_dim)(
        weights, repeat_ids
    )
    return self.shard_dim_by_stages(
        stage_weights, gathered_weights_stage_dim, physical_partition_spec=physical_partition_spec, is_stage_weight=True
    )

  def vmap_gather(self, xs, ids, ids_dim):
    """Use vmap to implement a stage-wise sharded gather.

    The stages share the same input, but they have different offsets.

    Args:
      xs: Data shared by all stages, to be gathered from.
      ids: Integer tensor of shape [num_stages], the offsets of the stages.
      ids_dim: The dimension in xs where ids are applied. In the output, this
        dimension will be [num_stages], since each stage gets one slice.

    Returns:
      The per-stage gathered values. The shape is xs.shape but with ids_dim size
        replaced with [num_stages].
    """
    xs = jnp.asarray(xs)
    ndim = xs.ndim

    def _gather_one(x, i):
      idx = tuple(i if d == ids_dim else slice(None) for d in range(ndim))
      replicated_sharding = NamedSharding(self.mesh, P())
      return x.at[idx].get(out_sharding=replicated_sharding)

    ids = self.shard_dim_by_stages(ids, 0, physical_partition_spec=None)
    outs = jax.vmap(_gather_one, in_axes=(None, 0), out_axes=ids_dim)(xs, ids)
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

    @jax.shard_map(mesh=self.mesh, in_specs=self.stages_in_spec, out_specs=self.stages_in_spec, check_vma=True)
    def _rotate_right(arr):
      # we use +1 for right shifting
      stage_size = jax.lax.axis_size("stage")
      perm = [(i, (i + 1) % stage_size) for i in range(stage_size)]
      return jax.lax.ppermute(arr, axis_name="stage", perm=perm)

    @jax.shard_map(mesh=self.mesh, in_specs=self.stages_in_spec, out_specs=self.stages_in_spec, check_vma=True)
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
        return _shift_right(output_in)  # last stage does not have to send to first immediately
      else:
        return _rotate_right(output_in)  # last stage must immediately send to first

    if self.config.pipeline_delay_activation_forwarding:
      new_shift = _update_shift(old_prev_outputs)
      new_prev_outputs = output
    else:
      new_shift = _update_shift(output)
      new_prev_outputs = None

    if self.use_circ_storage:
      # Insert the circ_storage_mover into new_circ_storage at a microbatch-rotating index.
      # circ_storage_mover still points to the output of PREVIOUS iteration, which should aid in allowing overlapped
      # compute/async transfers
      def _rotate_right_and_update(circ_storage_mover_in, circ_storage_in):
        rotated = jnp.expand_dims(_rotate_right(circ_storage_mover_in), 1)
        # We rotate the pushing index into circ storage, and ensure that microbatch 0 lands in index 0
        offset = (
            loop_iteration - self.iterations_to_complete_first_microbatch_one_repeat() - 1
        ) % self.config.num_pipeline_microbatches
        # previous output - using circ_storage_mover before it is updated
        return jax.lax.dynamic_update_slice_in_dim(circ_storage_in, rotated, offset, axis=1)

      new_circ_storage = _rotate_right_and_update(old_circ_storage_mover, old_circ_storage)
      new_circ_storage_mover = output
    else:
      new_circ_storage = None
      new_circ_storage_mover = None

    # Rotate stream_io left/up by 1 on rotating micro/stage index (stream_buf_idx), replacing the last/bottom with the
    # last stage output
    stream_buf_idx = loop_iteration % self.microbatches_per_stage
    stream_slice = old_state_io[:, stream_buf_idx]

    def _rotate_left(arr, stage_size):
      # we use -1 for left shifting
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
      # Shift the current slice to the left, then fill the last stage with the final output.
      stage_size = jax.lax.axis_size("stage")
      stream_slice = _shift_left(stream_slice, stage_size, output)
      stream_slice = jnp.expand_dims(stream_slice, 1)
      return jax.lax.dynamic_update_slice_in_dim(state_in, stream_slice, stream_buf_idx, axis=1)

    new_state = _update_state_io(old_state_io, stream_slice, output, stream_buf_idx)

    return {
        "state_io": new_state,
        "shift": new_shift,
        "circ_storage": new_circ_storage,
        "circ_storage_mover": new_circ_storage_mover,
        "loop_iteration": loop_iteration + 1,
        "prev_outputs": new_prev_outputs,
    }

  def permute_output_micro_per_stage_dim(self, output):
    """
    Permutes the output microbatches to match the input order.

    The pipeline execution introduces a delay (bubble) for each stage.
    Consequently, the first microbatch (index 0) finishes after a certain number of iterations
    and lands at a shifted position in the output buffer (`state_io`).
    This function calculates the offset (`microbatch_0_idx`) and permutes the output
    along the microbatch dimension so that microbatch 0 is at index 0, microbatch 1 at index 1, etc.
    """
    # The first real output (microbatch 0) takes a certain amount of loop iterations to finish and be pushed to
    # state_io - it will land on a different index of state_io depending on the number of iterations.
    microbatch_0_idx = self.iterations_to_complete_first_microbatch() % self.microbatches_per_stage
    permutation = (np.arange(self.microbatches_per_stage) + microbatch_0_idx) % self.microbatches_per_stage
    return output[:, permutation]

  def realign_output_microbatches(self, output):
    """Reorders the output tensor to reverse the circular shifts applied during execution.

    Because the pipeline operates circularly, the output microbatches are shifted
    out of order by the time the final stage is completed. This rolls them back
    into their original sequential layout.
    """
    microbatch_0_idx = self.iterations_to_complete_first_microbatch() % self.microbatches_per_stage
    output = jnp.roll(output, shift=-microbatch_0_idx, axis=1)
    return self._maybe_shard_with_logical(output, self.state_io_logical)

  def get_weight_sharding(self, *init_args):
    """Returns a pytree of logical-name PartitionSpecs mirroring the params state."""

    state = nnx.state(self.layers, _is_static_param)

    def get_spec(x):
      if not isinstance(x, nnx.Variable):
        # Non-VariableState leaf (e.g., nnx.Empty): treat as replicated.
        return P()
      # _overwrite_with_gradient variables (FP8 amax history / scales) carry no
      # partition metadata; return replicated to keep the tree aligned.
      if x.type.__name__ == "_overwrite_with_gradient":
        return P()
      # AQT QTensor values are a pytree wrapping quantized data; mirror the
      # skip-list in variable_to_logically_partitioned (initializers.py:81-83).
      if isinstance(x.value, aqt_tensor.QTensor):
        return P()
      if isinstance(x.value, nn.spmd.LogicallyPartitioned):
        # Dead in the NNX-first flow; retained as a forward-compat guard in
        # case a Linen-wrapped param is ever merged into this module.
        return x.value.partitions
      metadata = x.get_metadata()
      # Try each known metadata key in order; first hit wins.
      sharding = metadata.get("out_sharding")
      if sharding is None:
        sharding = metadata.get("sharding_names")
      if sharding is None:
        sharding = metadata.get("sharding")
      # Already a PartitionSpec - pass through.
      if isinstance(sharding, P):
        return sharding
      # Happy path: tuple/list of logical axis names from nnx.Param(sharding=...).
      if isinstance(sharding, (tuple, list)):
        return P(*sharding)
      # Non-PartitionSpec wrapper with an explicit ``.spec`` attribute (kept
      # for forward compatibility with future Flax wrapper types).
      if sharding is not None and hasattr(sharding, "spec"):
        return sharding.spec
      # Fallback: replicated sharding (valid for shard_map, unlike None).
      return P()

    return jax.tree.map(get_spec, state, is_leaf=lambda x: isinstance(x, nnx.Variable))

  def get_main_vmap_func_for_iterations(self):
    """Returns vmapped function that runs one pipeline iteration across stages."""

    def func_to_vmap(graph, state, stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode):
      module = nnx.merge(graph, state)
      out = module(stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode)
      return out, nnx.state(module)

    # Use jax.vmap instead of nnx.vmap to avoid nnx.State threading overhead
    # that produces 6.3x more compile-time temp memory. nnx.vmap adds extra
    # dynamic-slice/dynamic-update-slice ops for State management that inflate
    # gradient buffers. jax.vmap produces identical HLO to nn.vmap.
    #
    # spmd_axis_name is handled manually: in EXPLICIT mode, sharding is applied
    # by with_sharding_constraint calls in the pipeline. In AUTO mode, we apply
    # the axis name via jax.vmap's spmd_axis_name if available (JAX 0.4.31+).
    vmap_kwargs = dict(
        in_axes=(None, 0, 0, 0, 0, None, None),
        out_axes=(0, 0),
    )
    if self.spmd_axis_name is not None:
      vmap_kwargs["spmd_axis_name"] = self.spmd_axis_name
    return jax.vmap(func_to_vmap, **vmap_kwargs)

  @staticmethod
  def _stamp_at_current_trace(weights):
    """Pass each leaf through a no-op dynamic_slice so JAX creates new arrays
    at the *current* trace level.  This prevents trace-level mismatches when
    outer-trace values (e.g. closed-over by ``jax.lax.scan``) are later fed
    into ``nnx.merge`` inside the scan body.

    The operation is semantically an identity: ``x[0 : x.shape[0]]`` along
    axis 0, which XLA will optimise away.
    """

    def _identity_slice(x):
      if hasattr(x, "shape") and len(x.shape) > 0:
        return jax.lax.dynamic_slice_in_dim(x, 0, x.shape[0], axis=0)
      return x  # scalars / non-array leaves pass through unchanged

    return jax.tree.map(_identity_slice, weights)

  def get_current_stage_weights(self, pipeline_weights, loop_iteration, physical_partition_spec=None):
    """
    Gets the current weights used for one iteration. Outputs a pytree whose arrays have leading dimension of stages, e.g.
    {'mlp': 'wo': [stages, mlp, embed]}. Stage 0 will use the 0th index of this pytree, Stage 1 the 1st index, etc.
    For non-circular pipelines, this simply returns all weights - every weight is used in every iteraiton. However
    for circular pipelines each stage grabs only the weights corresponding to the current repeat.
    """
    if self.config.num_pipeline_repeats > 1:
      return self.get_current_repeat_from_stages(
          pipeline_weights, loop_iteration, physical_partition_spec=physical_partition_spec
      )
    # Stamp weights at the current trace level so that nnx.merge inside
    # func_to_vmap does not hit a trace-level mismatch when running under
    # jax.lax.scan (the weights may originate from an outer trace).
    return self._stamp_at_current_trace(pipeline_weights)

  def all_gather_over_fsdp(self, variables, logical_partition_spec):
    """
    all-gathers the variables over fsdp if fsdp is in the logical partition spec.
    """
    if logical_partition_spec is None:
      return variables

    def _gather_leaf(var, spec):
      if spec is None:
        return var
      physical = logical_to_mesh_axes(spec, self.mesh, rules=self.config.logical_axis_rules)
      no_fsdp = self._remove_fsdp_from_physical_partition_spec(physical)
      sharding = NamedSharding(self.mesh, no_fsdp)
      if isinstance(var, nnx.Variable):
        var.value = self._maybe_shard_with_name(var.value, sharding)
        return var
      return self._maybe_shard_with_name(var, sharding)

    # nnx.Variable and PartitionSpec are JAX pytree nodes -- treat them as leaves
    # so the two trees align at the dict level. None must also be a leaf to avoid
    # being treated as an empty container (0 children) vs the Variable's 1 child.
    def is_leaf(x):
      return isinstance(x, (nnx.Variable, P)) or x is None

    return jax.tree.map(_gather_leaf, variables, logical_partition_spec, is_leaf=is_leaf)

  def get_logical_spec_repeats_removed(self, full_logical):
    """Returns a new logical spec with 'circular_repeats' removed."""
    if full_logical is None or self.config.num_pipeline_repeats == 1:
      return full_logical

    def _remove_from_spec(spec):
      if not isinstance(spec, P):
        return spec
      if spec and (spec[0] == "circular_repeats" or spec[0] is None):
        return jax.sharding.PartitionSpec(*spec[1:])
      return jax.sharding.PartitionSpec(*[dim for dim in spec if dim != "circular_repeats"])

    return jax.tree.map(_remove_from_spec, full_logical, is_leaf=lambda x: isinstance(x, P))

  def __init__(
      self,
      config: Config,
      stage_factory: Any,
      mesh: Mesh,
      remat_policy: Any = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.remat_policy = remat_policy
    self._setup_pipeline_attributes()

    def build_batched_rngs(shape):
      kwargs = {}
      rng_state = nnx.state(rngs, nnx.RngState)
      leaves, _ = jax.tree_util.tree_flatten_with_path(rng_state)
      for path, key in leaves:
        stream_name = getattr(path[0], "key", str(path[0]))
        if not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
          key = jax.random.key(key)
        num_splits = int(np.prod(shape))
        flat_keys = jax.random.split(key, num_splits)
        kwargs[stream_name] = flat_keys.reshape(shape + key.shape)
      return nnx.Rngs(**kwargs)

    def create_stage_fn(r):
      stage = stage_factory(r)
      # Split into (GraphDef, Param State, Rest of State)
      return nnx.split(stage, nnx.Param, ...)

    vmap_stages = nnx.vmap(
        create_stage_fn,
        in_axes=0,
        out_axes=(None, 0, 0),
        spmd_axis_name=self.spmd_axis_name,
        transform_metadata={nnx.PARTITION_NAME: "layers"},
    )

    if self.config.num_pipeline_repeats > 1:
      vmap_repeats = nnx.vmap(
          vmap_stages,
          in_axes=0,
          out_axes=(None, 0, 0),
          transform_metadata={nnx.PARTITION_NAME: "circular_repeats"},
      )
      batched_rngs = build_batched_rngs((self.config.num_pipeline_repeats, self.num_stages))
      graphdef, params, rest = vmap_repeats(batched_rngs)
    else:
      batched_rngs = build_batched_rngs((self.num_stages,))
      graphdef, params, rest = vmap_stages(batched_rngs)

    # Merge the batched states back into the module
    self.layers = nnx.merge(graphdef, params, rest)


class NNXPipeline(NNXPipelineBase):
  """Original Pipeline implementation adapted for NNX."""

  def get_current_stage_weights(self, pipeline_weights, loop_iteration, physical_partition_spec=None):
    if self.config.num_pipeline_repeats > 1:
      return self.get_current_repeat_from_stages(
          pipeline_weights, loop_iteration, physical_partition_spec=physical_partition_spec
      )
    return self._stamp_at_current_trace(pipeline_weights)

  def get_current_repeat_from_stages(self, weights, loop_iteration, physical_partition_spec=None):
    """Fetches the weights for the current repeat from the stages."""
    _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)

    def gather_weights_for_stages_in(w, spec=None):
      if w is None:
        return None
      return self.vmap_parallel_gather(
          w, repeat_ids=repeat_ids, repeat_dim_in_weights=0, stages_dim_in_weights=1, physical_partition_spec=spec
      )

    if physical_partition_spec is None:
      return jax.tree.map(gather_weights_for_stages_in, weights)

    _, weights_params, weights_rest = nnx.split(weights, _is_static_param, ...)

    spec_leaves = jax.tree_util.tree_leaves(physical_partition_spec, is_leaf=is_spec_leaf)
    assert len(spec_leaves) == len(jax.tree_util.tree_leaves(weights_params)), (
        f"Spec tree leaf count ({len(spec_leaves)}) != weights tree leaf count "
        f"({len(jax.tree_util.tree_leaves(weights_params))}). "
        "The _is_static_param predicate may have diverged between get_weight_sharding and __call__."
    )
    spec_iter = iter(spec_leaves)
    gathered_params = jax.tree.map(
        lambda w: gather_weights_for_stages_in(w, next(spec_iter)),
        weights_params,
    )

    # Non-params gathered without sharding hints.
    gathered_rest = jax.tree.map(gather_weights_for_stages_in, weights_rest)

    return nnx.State.merge(gathered_params, gathered_rest)

  def run_one_iteration(
      self,
      loop_state,
      pipeline_weights_graph,
      pipeline_weights_state,
      positions,
      segment_ids,
      deterministic,
      model_mode,
      logical_partition_spec=None,
  ):
    """Executes the logic for a single microbatch iteration, including routing inputs and weights, and advancing buffers."""
    state_io = loop_state["state_io"]
    shift = loop_state["shift"]
    circ_storage = loop_state["circ_storage"]
    loop_iteration = loop_state["loop_iteration"]

    microbatch_ids, _ = self.get_microbatch_and_repeat_ids(loop_iteration)
    physical_partition_spec = logical_to_mesh(logical_partition_spec, self.mesh, rules=self.config.logical_axis_rules)

    stages_inputs = self.get_iteration_inputs(loop_iteration, state_io, circ_storage, shift)
    stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, "iteration_input")
    stages_positions = self.vmap_gather(positions, microbatch_ids, 0) if positions is not None else None
    stages_segment_ids = self.vmap_gather(segment_ids, microbatch_ids, 0) if segment_ids is not None else None

    vmap_func = self.get_main_vmap_func_for_iterations()

    stage_weights_state = self.get_current_stage_weights(
        pipeline_weights_state, loop_iteration, physical_partition_spec=physical_partition_spec
    )

    # Strip nnx.Variable wrappers to raw arrays before nnx.vmap.
    # When called inside jax.lax.scan, outer-scope Variables have
    # _can_update=False, causing check_consistent_aliasing to reject them.
    # nnx.merge inside func_to_vmap creates fresh Variables from raw values.
    stage_weights_state = jax.tree.map(
        lambda x: x.value if isinstance(x, nnx.Variable) else x,
        stage_weights_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

    stages_output, updated_stage_weights_state = vmap_func(
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

    if self.config.num_pipeline_repeats > 1:
      _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)

      def _scatter_update(fw, uw, spec=None):
        if fw is None or uw is None:
          return fw

        def _update_one_stage(f_s, u_s, r_id):
          return jax.lax.dynamic_update_slice_in_dim(f_s, jnp.expand_dims(u_s, 0), r_id, axis=0)

        r_ids = self.shard_dim_by_stages(repeat_ids, 0, physical_partition_spec=None)
        updated_fw = jax.vmap(_update_one_stage, in_axes=(1, 0, 0), out_axes=1)(fw, uw, r_ids)
        return self.shard_dim_by_stages(updated_fw, 1, physical_partition_spec=spec, is_stage_weight=False)

      pipeline_weights_state = jax.tree.map(_scatter_update, pipeline_weights_state, updated_stage_weights_state)
    else:
      pipeline_weights_state = updated_stage_weights_state

    new_state = self.get_new_loop_state(stages_output, loop_state)
    return new_state, pipeline_weights_state

  def __call__(
      self,
      inputs: jnp.ndarray,
      segment_ids: jnp.ndarray,
      positions: jnp.ndarray,
      deterministic: bool,
      model_mode=MODEL_MODE_TRAIN,
      logical_partition_spec=None,  # Pytree of sharding specifications of the weights (aka self.layers.variables)
  ) -> jnp.ndarray:
    """The main method that maps the series of decoder layer inputs to final layer outputs.
    Has the same signature of a single decoder layer, and expects the same shapes, e.g. the inputs should have shape
    [global_batch], and internally this will be reshapped into microbatches.
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

    bubble_iterations = self.forwarding_delay * (self.num_stages - 1)
    real_iterations = self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats
    total_iterations = real_iterations + bubble_iterations

    logical_partition_spec = self.get_logical_spec_repeats_removed(logical_partition_spec)

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

    if self.config.pipeline_fsdp_ag_once:
      layers_params = self.all_gather_over_fsdp(layers_params, logical_partition_spec)

    def scan_body(carry, _):
      current_loop_state, current_layer_mutables = carry
      iteration = current_loop_state["loop_iteration"]
      advanced_mutables = _advance_rng_state(current_layer_mutables, iteration)
      current_layer_state = nnx.State.merge(layers_params, layers_metrics, advanced_mutables)

      new_loop_state, new_layer_state = self.run_one_iteration(
          current_loop_state,
          layers_graph,
          current_layer_state,
          positions,
          segment_ids,
          deterministic,
          model_mode,
          logical_partition_spec,
      )

      _, _, new_layer_metrics, new_layer_mutables = nnx.split(new_layer_state, _is_static_param, nnx.Intermediate, ...)
      return (new_loop_state, new_layer_mutables), new_layer_metrics

    if self.config.set_remat_policy_on_pipeline_iterations:
      scan_body = jax.checkpoint(
          scan_body, policy=self.get_pipeline_remat_policy(), prevent_cse=not self.config.scan_pipeline_iterations
      )

    if self.config.scan_pipeline_iterations:
      (loop_state, final_layer_mutables), stacked_metrics = jax.lax.scan(
          scan_body, (loop_state, layers_mutables), None, length=total_iterations
      )
    else:
      current_carry = (loop_state, layers_mutables)
      metrics_history = []
      for _ in range(total_iterations):
        current_carry, step_metrics = scan_body(current_carry, None)
        metrics_history.append(step_metrics)
      loop_state, final_layer_mutables = current_carry
      stacked_metrics = jax.tree.map(lambda *xs: jnp.stack(xs), *metrics_history) if metrics_history else layers_metrics

    final_layer_state = nnx.State.merge(layers_params, stacked_metrics, final_layer_mutables)
    nnx.update(self.layers, final_layer_state)

    final_output = self.permute_output_micro_per_stage_dim(loop_state["state_io"])
    return jnp.reshape(
        final_output,
        (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim),
        out_sharding=self.output_sharding,
    )


class NNXCircularPipeline(NNXPipelineBase):
  """NNX circular pipeline with 2-level custom_vjp + linear_transpose.

  Matches the Linen golden pattern from pipeline_utils.py:
    - Outer custom_vjp (execute_stage): manages BSW + linear_transpose for
      reduce-scatter dual of weight_prefetching all-gather.
    - Inner custom_vjp (run_microbatch): per-microbatch d+g gradient
      accumulation on BSW with remat.

  Key design:
    - layers_params is CLOSURE of outer jax.lax.scan (stored once, not per-iteration)
    - layers_params is POSITIONAL ARG of execute_stage custom_vjp (_bwd returns d_params)
    - BSW enters inner scan as closure; inner _bwd returns None for bsw grad
    - Outer _bwd handles BSW gradients via d+g pattern + linear_transpose
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
      # TODO (chengnuojin) set use_shardmap=true after JAX >= 0.10.0 and use all_gather(..., to='invarying')
      use_shardmap=False,  # using shardmap produces additional reduce-scatter in backward pass
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

      # Dynamically gather the index pytrees for all specified axes
      axis_indices_dict = {
          axis: pipeline_utils.get_mesh_axis_dim_indices(physical_partition_spec, axis) for axis in axes_to_gather
      }

      axis_names = list(axis_indices_dict.keys())
      axis_pytrees = list(axis_indices_dict.values())

      def should_skip_gather(axis_name, path_keys):
        """Defines specific rule-based exceptions for gathering certain axes."""
        if axis_name == "expert" and "MoeBlock_0" in path_keys:
          return True
        # Add more exclusion rules for other axes here if needed in the future
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
    """Prefetch next repeat's weights for the Buffer Sliding Window.

    Only gathers weights for `loop_iteration + 1`. The current iteration's
    weights are carried forward from the previous scan step's prefetch,
    matching the Linen sliding-window pattern and halving the number of
    FSDP all-gathers per iteration.
    """
    nxt_repeat_weights = self.from_all_variables_to_repeat_weights(weights_state, loop_iteration + 1)
    return self.from_repeat_weights_to_bsw(nxt_repeat_weights, physical_partition_spec)

  def fetch_active_stage_weights(self, bsw, loop_iteration, physical_partition_spec=None):
    """The module fetches the actively prefetched weights
    from the Buffer Sliding Window to avoid mid-iteration FSDP all-gathers.
    """
    return self.get_current_weights_from_bsw(bsw, loop_iteration, physical_partition_spec)

  def get_current_weights_from_bsw(self, bsw, loop_iteration, physical_partition_spec):
    """Pulls the fully gathered parameters for the current repeat from the BSW dual-buffer."""
    # Fast path: both BSW slots are the same object -- skip shard_map select.
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
    """Executes the forward/backward logic for a single microbatch inside the circular pipeline.

    Fetches params from BSW (params-only), gathers metrics/mutables directly for the current
    repeat, merges into full state for the forward pass, then scatter-updates only non-params
    back (params are static in scan and handled by AD/gradient).
    """
    state_io = loop_state["state_io"]
    shift = loop_state["shift"]
    circ_storage = loop_state["circ_storage"]
    loop_iteration = loop_state["loop_iteration"]

    _n3_log(
      f"    . run_one_iteration: bsw="
      f"{'None' if bsw is None else f'leaves={_leaf_count(bsw)} bytes={_byte_size(bsw)/1e9:.3f} GB'} "
      f"state_io_shape={tuple(state_io.shape)} shift_shape={tuple(shift.shape)} "
      f"circ_storage={'None' if circ_storage is None else tuple(circ_storage.shape)}"
    )

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
    _n3_log(
      f"    . run_one_iteration: stages_inputs_shape={tuple(stages_inputs.shape)} "
      f"stage_params(from BSW) leaves={_leaf_count(stage_params)} "
      f"bytes={_byte_size(stage_params)/1e9:.3f} GB first_shape={_shapes(stage_params, limit=1)}"
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
      # Stamp at current trace level to avoid nnx.merge trace-level mismatch
      # (layers_metrics is closed over from outer scope in scan).
      stage_metrics = self._stamp_at_current_trace(layers_metrics)
      stage_mutables = current_layer_mutables  # already at scan trace level (from carry)

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

    # Scatter-back: only update mutables (params handled by AD/gradient, metrics returned directly)
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
    """NNXCircularPipeline forward pass with 2-level custom_vjp + linear_transpose.

    Architecture (matches Linen golden pattern from pipeline_utils.py):

    outer_scan(repeats):
      carry = (loop_state, layers_mutables, w_curr)
      layers_params as CLOSURE (scan constant, 1 copy)

      scan_body -> execute_stage(loop_state, w_curr, layers_params, mutables)
        [Outer custom_vjp: BSW + linear_transpose]
        fwd: weight_prefetching -> BSW -> inner_scan -> save (scan_vjp, weight_prefetching_t)
        bwd: scan_vjp(g) -> d+g accumulation -> linear_transpose(d_w_next) -> d_params

        inner_scan(microbatches):
          run_microbatch(loop_state, mutables, bsw_closure)
            [Inner custom_vjp: per-microbatch remat + d+g on BSW]
            fwd: vjp(remat(forward)) -> save vjp_fn
            bwd: vjp_fn(g) -> (d_loop_state, d_bsw) -> d+g accumulation
    """
    # === #1: __call__ ENTER -- config ===
    _n3_log("=" * 70)
    _n3_log("NNXCircularPipeline.__call__ ENTER")
    _n3_log(
        f"  config: num_stages={self.num_stages} "
        f"num_microbatches={self.config.num_pipeline_microbatches} "
        f"num_repeats={self.config.num_pipeline_repeats}"
    )
    _n3_log(
        f"  config: scan_pipeline_iterations={self.config.scan_pipeline_iterations} "
        f"scan_pipeline_repeats={getattr(self.config, 'scan_pipeline_repeats', None)} "
        f"pipeline_fsdp_ag_per_repeat={self.config.pipeline_fsdp_ag_per_repeat}"
    )
    _n3_log(
        f"  config: forwarding_delay={self.forwarding_delay} "
        f"use_circ_storage={self.use_circ_storage} "
        f"microbatches_per_stage={self.microbatches_per_stage} "
        f"set_remat_policy_on_pipeline_iterations={self.config.set_remat_policy_on_pipeline_iterations}"
    )
    _n3_log(
        f"  inputs in: shape={tuple(inputs.shape)} dtype={inputs.dtype} "
        f"segment_ids={'None' if segment_ids is None else tuple(segment_ids.shape)} "
        f"positions={'None' if positions is None else tuple(positions.shape)} "
        f"deterministic={deterministic} model_mode={model_mode}"
    )

    inputs = inputs.reshape(
        (
            self.config.num_pipeline_microbatches,
            self.pipeline_microbatch_size,
            self.config.max_target_length,
            self.config.emb_dim,
        ),
        out_sharding=self.input_sharding,
    )
    _n3_log(f"  inputs reshaped to microbatches: shape={tuple(inputs.shape)} dtype={inputs.dtype}")

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
    # === #3 (partial): initial loop_state structure ===
    _n3_log(f"  init loop_state keys={_tree_keys(loop_state)}")
    for _k in _tree_keys(loop_state) if isinstance(loop_state, dict) else []:
      _v = loop_state[_k]
      _shp = getattr(_v, "shape", None)
      _n3_log(f"    loop_state['{_k}']: {tuple(_shp) if _shp is not None else _v}")

    if is_linen_initializing():
      _n3_log("  is_linen_initializing()==True -> returning zeros (init path, NO compute)")
      return jnp.zeros(
          (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim),
          dtype=inputs.dtype,
      )

    # Two spec variants needed:
    # - Full spec (with circular_repeats axis) -> BSW creation inside scan_body via
    #   from_all_variables_to_repeat_weights + from_repeat_weights_to_bsw.
    # - Stripped logical spec (circular_repeats removed) -> BSW consumption via
    #   run_one_iteration.
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

    # layers_mutables catch-all should contain ONLY RngState variables (RngKey/RngCount).
    assert all(
        isinstance(v, nnx.RngState)
        for v in jax.tree.leaves(layers_mutables, is_leaf=lambda x: isinstance(x, nnx.Variable))
        if isinstance(v, nnx.Variable)
    ), (
        "Non-RngState variable found in layers_mutables catch-all partition. "
        "Only RngState variables (RngKey/RngCount) should be present."
    )

    # === #2: after nnx.split -- param/metric/mutable tree leaf counts + shapes ===
    _n3_log("after nnx.split(layers_state):")
    _n3_log(
        f"  layers_params : leaves={_leaf_count(layers_params)} "
        f"bytes={_byte_size(layers_params)/1e9:.3f} GB first_shapes={_shapes(layers_params)}"
    )
    _n3_log(
        f"  layers_metrics: leaves={_leaf_count(layers_metrics)} "
        f"bytes={_byte_size(layers_metrics)/1e9:.3f} GB first_shapes={_shapes(layers_metrics)}"
    )
    _n3_log(
        f"  layers_mutables: leaves={_leaf_count(layers_mutables)} "
        f"bytes={_byte_size(layers_mutables)/1e9:.3f} GB first_shapes={_shapes(layers_mutables)}"
    )
    _n3_log(
        f"  physical_partition_spec_full leaves="
        f"{_leaf_count(physical_partition_spec_full, is_leaf=is_spec_leaf)} "
        f"bubble_iterations={bubble_iterations}"
    )

    # ---- 2-level custom_vjp nested scan structure ----
    #
    # outer scan (repeats):
    #   layers_params as CLOSURE (scan constant -- 1 copy, not per-iteration)
    #   carry = (loop_state, layers_mutables, w_curr)
    #
    #   outer_body -> execute_stage custom_vjp:
    #     fwd: weight_prefetching -> BSW -> inner_scan_with_vjp
    #          saves (scan_vjp_fn, weight_prefetching_t) as residuals
    #     bwd: scan_vjp_fn backprop -> d+g on BSW -> linear_transpose -> d_params
    #
    #   inner scan (microbatches) -> run_microbatch custom_vjp:
    #     fwd: vjp(remat(run_one_iteration)) -> saves vjp_fn
    #     bwd: vjp_fn(g) -> d_loop_state, d_bsw (returned), None for mutables
    #
    # bubble iterations: same inner_body, no custom_vjp needed

    num_microbatches = self.config.num_pipeline_microbatches
    remat_policy = self.get_pipeline_remat_policy()
    scan_pipeline_iters = self.config.scan_pipeline_iterations
    # === #10: jax.checkpoint remat policy name ===
    _n3_log(
        f"remat policy resolved: config.remat_policy={self.config.remat_policy!r} "
        f"resolved_type={type(remat_policy).__name__} scan_pipeline_iters={scan_pipeline_iters} "
        f"num_microbatches={num_microbatches}"
    )

    # ---- Inner custom_vjp: per-microbatch remat with d+g on BSW ----
    #
    # Matches Linen's run_single_microbatch_custom from pipeline_utils.py.
    # BSW enters as positional arg so gradient flows through it.
    # Mutables have stop_gradient applied.

    @jax.custom_vjp
    def run_microbatch(loop_state_arg, bsw_arg, mutables_arg):
      """Forward pass for one microbatch iteration."""
      return run_microbatch_fwd(loop_state_arg, bsw_arg, mutables_arg)[0]

    def run_microbatch_fwd(loop_state_arg, bsw_arg, mutables_arg):
      """Forward + save VJP closure for backward remat."""
      # === #9: run_microbatch_fwd ENTER ===
      _n3_log(
        f"  >> run_microbatch_fwd ENTER loop_state keys={_tree_keys(loop_state_arg)} "
        f"bsw_leaves={_leaf_count(bsw_arg)} bsw_first_shape={_shapes(bsw_arg, limit=1)} "
        f"bsw_bytes={_byte_size(bsw_arg)/1e9:.3f} GB mutables_leaves={_leaf_count(mutables_arg)}"
      )
      jax.debug.print(
          "[PIPELINE_N3] run_microbatch_fwd RUNTIME loop_iteration={i}",
          i=loop_state_arg["loop_iteration"],
      )
      # === #10: jax.checkpoint application ===
      _n3_log(f"  >> run_microbatch_fwd applying remat policy: {type(remat_policy).__name__}")
      def _run(ls, b):
        iteration = ls["loop_iteration"]
        advanced_muts = _advance_rng_state(mutables_arg, iteration)
        new_ls, new_layer_state = self.run_one_iteration(
            ls, b, layers_graph, layers_metrics, advanced_muts,
            positions, segment_ids, deterministic, model_mode,
            logical_partition_spec_stripped,
        )
        _, _, new_metrics, new_muts = nnx.split(
            new_layer_state, _is_static_param, nnx.Intermediate, ...
        )
        return (new_ls, new_muts), new_metrics

      _run_remat = jax.remat(_run, policy=remat_policy)
      out, vjp_fn = jax.vjp(_run_remat, loop_state_arg, bsw_arg)
      # === #Residual tracking: run_microbatch_fwd ===
      _n3_log(
        "  >> run_microbatch_fwd RESIDUALS: vjp_fn(closure over remat(run_one_iteration), "
        "loop_state_arg, bsw_arg). bsw is differentiable input -> gathered weights become "
        "VJP residuals unless remat-recomputed."
      )
      return out, vjp_fn

    def run_microbatch_bwd(vjp_fn, g_out):
      """Backward: backprop through remat'd forward."""
      # === #9: run_microbatch_bwd ENTER -- incoming grad shapes ===
      _n3_log(
        f"  << run_microbatch_bwd ENTER g_out leaves={_leaf_count(g_out)} "
        f"first_shapes={_shapes(g_out)}"
      )
      d_ls, d_bsw = vjp_fn(g_out)
      # === #9: run_microbatch_bwd -- outgoing grad shapes ===
      _n3_log(
        f"  << run_microbatch_bwd OUT d_ls leaves={_leaf_count(d_ls)} "
        f"d_bsw leaves={_leaf_count(d_bsw)} d_bsw_bytes={_byte_size(d_bsw)/1e9:.3f} GB "
        f"d_bsw_first_shape={_shapes(d_bsw, limit=1)} (mutables grad=None)"
      )
      # No gradient for mutables (stop_gradient applied in fwd)
      return d_ls, d_bsw, None

    run_microbatch.defvjp(run_microbatch_fwd, run_microbatch_bwd)

    # ---- Outer custom_vjp wrapper: microbatch scan with d+g accumulation ----
    #
    # Matches Linen's run_pipeline_microbatches_custom from pipeline_utils.py.
    # Wraps the inner scan and applies d+g gradient accumulation on BSW.

    @jax.custom_vjp
    def run_microbatch_scan(loop_state_arg, bsw_arg, mutables_arg):
      """Scan over microbatches with d+g gradient accumulation."""
      return run_microbatch_scan_fwd(loop_state_arg, bsw_arg, mutables_arg)[0]

    def run_microbatch_scan_fwd(loop_state_arg, bsw_arg, mutables_arg):
      """Forward: scan over microbatches, capture VJP for backward."""
      # === #8: inner scan ENTER -- microbatch count + inner carry shapes ===
      _n3_log(
        f" >>> run_microbatch_scan_fwd ENTER (INNER SCAN) length={num_microbatches} "
        f"scan_pipeline_iters={scan_pipeline_iters}"
      )
      _n3_log(
        f"     inner carry: loop_state keys={_tree_keys(loop_state_arg)} "
        f"bsw_leaves={_leaf_count(bsw_arg)} bsw_first_shape={_shapes(bsw_arg, limit=1)} "
        f"mutables_leaves={_leaf_count(mutables_arg)}"
      )
      def scan_fn(ls_arg, b_arg):
        def inner_body(inner_carry, _):
          ls, muts = inner_carry
          muts = jax.lax.stop_gradient(muts)
          (new_ls, new_muts), new_metrics = run_microbatch(ls, b_arg, muts)
          return (new_ls, new_muts), new_metrics

        if scan_pipeline_iters:
          (final_ls, final_muts), metrics = jax.lax.scan(
              inner_body, (ls_arg, mutables_arg), None, length=num_microbatches,
          )
        else:
          carry = (ls_arg, mutables_arg)
          metrics_list = []
          for _ in range(num_microbatches):
            carry, step_met = inner_body(carry, None)
            metrics_list.append(step_met)
          final_ls, final_muts = carry
          metrics = (
              jax.tree.map(lambda *xs: jnp.stack(xs), *metrics_list)
              if metrics_list else layers_metrics
          )
        return (final_ls, final_muts), metrics

      scan_output, scan_vjp_fn = jax.vjp(scan_fn, loop_state_arg, bsw_arg)
      (final_ls, final_muts), metrics = scan_output
      # === #Residual tracking: run_microbatch_scan_fwd ===
      _n3_log(
        " >>> run_microbatch_scan_fwd RESIDUALS: scan_vjp_fn(closure over jax.lax.scan of "
        f"inner_body). bsw_arg passed through to output (g_bsw routing). "
        f"metrics_leaves={_leaf_count(metrics)} metrics_first_shapes={_shapes(metrics, limit=2)}"
      )

      # Return bsw_arg as part of output so outer custom_vjp can get g_bsw
      return ((final_ls, final_muts, bsw_arg), metrics), scan_vjp_fn

    def run_microbatch_scan_bwd(scan_vjp_fn, g_out):
      """Backward: d+g gradient accumulation on BSW."""
      (g_ls_muts_bsw, g_metrics) = g_out
      g_ls, g_muts, g_bsw = g_ls_muts_bsw
      # === #8: inner scan BWD ENTER ===
      _n3_log(
        f" <<< run_microbatch_scan_bwd ENTER g_ls leaves={_leaf_count(g_ls)} "
        f"g_muts leaves={_leaf_count(g_muts)} g_bsw leaves={_leaf_count(g_bsw)} "
        f"g_bsw_first_shape={_shapes(g_bsw, limit=1)} g_metrics leaves={_leaf_count(g_metrics)}"
      )

      # Backprop through scan
      d_ls, d_bsw = scan_vjp_fn(((g_ls, g_muts), g_metrics))

      # === #7: d+g accumulation -- d_bsw and g_bsw shapes BEFORE ===
      _n3_log(
        f" <<< run_microbatch_scan_bwd d+g BEFORE: d_bsw leaves={_leaf_count(d_bsw)} "
        f"d_bsw_bytes={_byte_size(d_bsw)/1e9:.3f} GB d_bsw_first_shape={_shapes(d_bsw, limit=1)} "
        f"| g_bsw leaves={_leaf_count(g_bsw)} g_bsw_bytes={_byte_size(g_bsw)/1e9:.3f} GB "
        f"g_bsw_first_shape={_shapes(g_bsw, limit=1)}"
      )
      # Tree-structure assertion before the add (catch mismatch before TPU crash).
      assert jax.tree.structure(d_bsw) == jax.tree.structure(g_bsw), (
          "[PIPELINE_N3] TREE MISMATCH d_bsw vs g_bsw in run_microbatch_scan_bwd: "
          f"{jax.tree.structure(d_bsw)} != {jax.tree.structure(g_bsw)}"
      )

      # d+g accumulation: add direct BSW gradient to scan-accumulated gradient
      d_bsw = jax.tree.map(
          lambda d, g: d + g if hasattr(d, "shape") else d, d_bsw, g_bsw
      )

      # === #7: d+g accumulation -- accumulated shape AFTER ===
      _n3_log(
        f" <<< run_microbatch_scan_bwd d+g AFTER: d_bsw leaves={_leaf_count(d_bsw)} "
        f"d_bsw_bytes={_byte_size(d_bsw)/1e9:.3f} GB d_bsw_first_shape={_shapes(d_bsw, limit=1)} "
        f"d_ls leaves={_leaf_count(d_ls)} (mutables grad=None)"
      )

      return d_ls, d_bsw, None

    run_microbatch_scan.defvjp(run_microbatch_scan_fwd, run_microbatch_scan_bwd)

    # ---- Outermost custom_vjp: execute_stage with linear_transpose ----
    #
    # Matches Linen's execute_pipeline_stage_pure from pipeline_utils.py.
    # Manages BSW construction via weight_prefetching + linear_transpose
    # for reduce-scatter dual in backward.
    #
    # layers_params is POSITIONAL ARG here so _bwd can return d_params.
    # But layers_params is CLOSURE of the outer jax.lax.scan (scan constant).

    @jax.custom_vjp
    def execute_stage(loop_state_arg, w_curr_arg, pipeline_weights_arg, mutables_arg):
      """One pipeline repeat: prefetch weights, run microbatch scan."""
      return execute_stage_fwd(loop_state_arg, w_curr_arg, pipeline_weights_arg, mutables_arg)[0]

    def execute_stage_fwd(loop_state_arg, w_curr_arg, pipeline_weights_arg, mutables_arg):
      """Forward: weight prefetching + microbatch scan, save linear_transpose."""
      iteration = loop_state_arg["loop_iteration"]
      # === #4: execute_stage_fwd ENTER -- arg shapes ===
      _n3_log("-" * 60)
      _n3_log("ENTER execute_stage_fwd (OUTER custom_vjp / one repeat)")
      _n3_log(
        f"  arg loop_state keys={_tree_keys(loop_state_arg)} "
        f"w_curr leaves={_leaf_count(w_curr_arg)} w_curr_bytes={_byte_size(w_curr_arg)/1e9:.3f} GB "
        f"w_curr_first_shape={_shapes(w_curr_arg, limit=1)}"
      )
      _n3_log(
        f"  arg pipeline_weights leaves={_leaf_count(pipeline_weights_arg)} "
        f"pipeline_weights_bytes={_byte_size(pipeline_weights_arg)/1e9:.3f} GB "
        f"first_shapes={_shapes(pipeline_weights_arg)} mutables leaves={_leaf_count(mutables_arg)}"
      )
      jax.debug.print("[PIPELINE_N3] execute_stage_fwd RUNTIME loop_iteration={i}", i=iteration)

      # === #5: weight_prefetching call -- input shapes ===
      _n3_log(
        f"  >> weight_prefetching INPUT pipeline_weights leaves={_leaf_count(pipeline_weights_arg)} "
        f"bytes={_byte_size(pipeline_weights_arg)/1e9:.3f} GB"
      )
      # Prefetch next repeat's weights (all-gather)
      w_next = self.weight_prefetching(
          pipeline_weights_arg, physical_partition_spec_full, iteration
      )
      # === #5: weight_prefetching -- output shapes ===
      _n3_log(
        f"  << weight_prefetching OUTPUT w_next leaves={_leaf_count(w_next)} "
        f"bytes={_byte_size(w_next)/1e9:.3f} GB first_shapes={_shapes(w_next, limit=3)} "
        "(fully-gathered next-repeat weights)"
      )

      # Construct Buffer Sliding Window
      bsw = (w_curr_arg, w_next)
      _n3_log(
        f"  BSW constructed = (w_curr, w_next): total_leaves={_leaf_count(bsw)} "
        f"total_bytes={_byte_size(bsw)/1e9:.3f} GB"
      )

      # Build linear_transpose of weight_prefetching for backward reduce-scatter.
      # Partial out physical_partition_spec and loop_iteration so only
      # pipeline_weights is the differentiable input.
      # === #6: linear_transpose -- creating weight_prefetching_t ===
      _n3_log("  >> Creating weight_prefetching_t via jax.linear_transpose(weight_prefetching)")
      p_weight_prefetching = functools.partial(
          self.weight_prefetching,
          physical_partition_spec=physical_partition_spec_full,
          loop_iteration=iteration,
      )
      weight_prefetching_t = jax.linear_transpose(
          p_weight_prefetching,
          pipeline_weights_arg,
      )
      _n3_log(
        "  << weight_prefetching_t CREATED (reduce-scatter dual of all-gather). "
        "It is a CLOSURE residual -- does NOT itself store gathered weights; computes "
        "d_pipeline_weights analytically from d_w_next in bwd."
      )

      # Run microbatch scan with VJP capture
      ((final_ls, final_muts, _bsw_out), metrics), scan_vjp_fn = jax.vjp(
          lambda ls, b: run_microbatch_scan(ls, b, mutables_arg),
          loop_state_arg,
          bsw,
      )

      # === #Residual tracking (CRITICAL): execute_stage_fwd ===
      _n3_log(
        "  execute_stage_fwd RESIDUALS: scan_vjp(closure), weight_prefetching_t(closure)"
      )
      _n3_log(
        f"    NOTE: w_next (bytes={_byte_size(w_next)/1e9:.3f} GB) is RETURNED as output and "
        "becomes the next-iteration w_curr CARRY of the outer scan -> fully-gathered weights "
        "live in the scan carry across repeats (primary suspected memory cost)."
      )

      # Output: (loop_state, w_next, mutables), metrics
      # Residuals: (scan_vjp_fn, weight_prefetching_t)
      return ((final_ls, w_next, final_muts), metrics), (scan_vjp_fn, weight_prefetching_t)

    def execute_stage_bwd(residuals, g_out):
      """Backward: scan VJP backprop + linear_transpose for weight gradients."""
      scan_vjp_fn, weight_prefetching_t = residuals
      (g_result, g_metrics) = g_out
      g_ls, g_w_next, g_muts = g_result
      # === #4: execute_stage_bwd ENTER -- incoming grad shapes ===
      _n3_log("-" * 60)
      _n3_log("ENTER execute_stage_bwd (OUTER custom_vjp bwd / one repeat)")
      _n3_log(
        f"  incoming g_ls leaves={_leaf_count(g_ls)} g_w_next leaves={_leaf_count(g_w_next)} "
        f"g_w_next_bytes={_byte_size(g_w_next)/1e9:.3f} GB g_w_next_first_shape={_shapes(g_w_next, limit=1)} "
        f"g_muts leaves={_leaf_count(g_muts)} g_metrics leaves={_leaf_count(g_metrics)}"
      )

      # Initialize zero cotangents for w_curr (consumed in forward)
      g_w_curr = jax.tree.map(jnp.zeros_like, g_w_next)
      g_bsw = (g_w_curr, g_w_next)

      # Backprop through microbatch scan
      # scan_vjp_fn expects gradients matching scan_fn outputs:
      #   scan_fn returns ((final_ls, final_muts, bsw), metrics)
      #   so g_out for scan_vjp = ((g_ls, g_muts, g_bsw), g_metrics)
      d_ls, d_bsw = scan_vjp_fn(((g_ls, g_muts, g_bsw), g_metrics))

      # The d+g accumulation was already done inside run_microbatch_scan_bwd.
      # d_bsw now contains the fully accumulated BSW gradients.
      d_w_curr, d_w_next = d_bsw
      _n3_log(
        f"  after scan_vjp_fn: d_ls leaves={_leaf_count(d_ls)} "
        f"d_w_curr leaves={_leaf_count(d_w_curr)} d_w_curr_bytes={_byte_size(d_w_curr)/1e9:.3f} GB "
        f"d_w_next leaves={_leaf_count(d_w_next)} d_w_next_bytes={_byte_size(d_w_next)/1e9:.3f} GB"
      )

      # Apply linear_transpose of weight_prefetching (reduce-scatter dual
      # of all-gather) to map gathered-weight gradients back to FSDP-sharded
      # parameter space.
      # === #6: linear_transpose CALL in bwd ===
      _n3_log(
        f"  >> calling weight_prefetching_t(d_w_next): IN d_w_next_bytes="
        f"{_byte_size(d_w_next)/1e9:.3f} GB (gathered grads -> reduce-scatter to FSDP-sharded)"
      )
      (d_pipeline_weights,) = weight_prefetching_t(d_w_next)
      _n3_log(
        f"  << weight_prefetching_t OUT d_pipeline_weights leaves={_leaf_count(d_pipeline_weights)} "
        f"bytes={_byte_size(d_pipeline_weights)/1e9:.3f} GB first_shapes={_shapes(d_pipeline_weights)}"
      )
      _n3_log(
        f"  execute_stage_bwd RETURN: d_ls, d_w_curr({_byte_size(d_w_curr)/1e9:.3f} GB), "
        f"d_pipeline_weights({_byte_size(d_pipeline_weights)/1e9:.3f} GB), None(mutables)"
      )

      # Return gradients for: loop_state, w_curr, pipeline_weights, mutables
      return d_ls, d_w_curr, d_pipeline_weights, None

    execute_stage.defvjp(execute_stage_fwd, execute_stage_bwd)

    # ---- Outer scan body (repeats) ----

    def outer_body(carry, _):
      """One repeat: execute_stage custom_vjp with layers_params as positional arg."""
      current_loop_state, current_layer_mutables, w_curr = carry
      _n3_log(
        f"### outer_body (REPEAT) ENTER: carry=(loop_state[{_tree_keys(current_loop_state)}], "
        f"mutables[{_leaf_count(current_layer_mutables)} leaves], "
        f"w_curr[{_leaf_count(w_curr)} leaves, {_byte_size(w_curr)/1e9:.3f} GB])"
      )

      # execute_stage takes layers_params as positional arg for gradient flow.
      # layers_params is also closed over by the outer scan (scan constant).
      (new_loop_state, w_next, new_layer_mutables), inner_metrics = execute_stage(
          current_loop_state, w_curr, layers_params, current_layer_mutables
      )

      # Carry must keep identical tree structure across scan iterations.
      assert jax.tree.structure(w_next) == jax.tree.structure(w_curr), (
          "[PIPELINE_N3] TREE MISMATCH outer_body w_next(new carry) vs w_curr(old carry): "
          f"{jax.tree.structure(w_next)} != {jax.tree.structure(w_curr)}"
      )
      _n3_log(
        f"### outer_body EXIT: w_next(new w_curr carry)[{_leaf_count(w_next)} leaves, "
        f"{_byte_size(w_next)/1e9:.3f} GB] inner_metrics[{_leaf_count(inner_metrics)} leaves]"
      )

      return (new_loop_state, new_layer_mutables, w_next), inner_metrics

    # ---- Bubble inner_body (no custom_vjp needed) ----
    # During bubble iterations, BSW is fixed (final_w_curr, final_w_curr).
    # No weight prefetching or gradient accumulation needed.
    bsw_ref = [None]

    def bubble_inner_body(carry, _):
      """One bubble iteration: simple forward, no custom_vjp."""
      current_loop_state, current_layer_mutables = carry
      # === #11: ENTER bubble iteration -- BSW shapes ===
      _n3_log(
        f"@@@ bubble_inner_body ENTER loop_state keys={_tree_keys(current_loop_state)} "
        f"bsw={'None' if bsw_ref[0] is None else f'leaves={_leaf_count(bsw_ref[0])} bytes={_byte_size(bsw_ref[0])/1e9:.3f} GB first_shape={_shapes(bsw_ref[0], limit=1)}'}"
      )
      current_layer_mutables = jax.lax.stop_gradient(current_layer_mutables)
      iteration = current_loop_state["loop_iteration"]
      advanced_mutables = _advance_rng_state(current_layer_mutables, iteration)
      new_loop_state, new_layer_state = self.run_one_iteration(
          current_loop_state,
          bsw_ref[0],
          layers_graph,
          layers_metrics,
          advanced_mutables,
          positions,
          segment_ids,
          deterministic,
          model_mode,
          logical_partition_spec_stripped,
      )
      _, _, new_layer_metrics, new_layer_mutables = nnx.split(new_layer_state, _is_static_param, nnx.Intermediate, ...)
      return (new_loop_state, new_layer_mutables), new_layer_metrics

    if self.config.set_remat_policy_on_pipeline_iterations:
      # === #10: jax.checkpoint application on bubble body ===
      _n3_log(f"Applying remat policy on bubble_inner_body: {type(remat_policy).__name__}")
      bubble_inner_body = jax.checkpoint(
          bubble_inner_body,
          policy=remat_policy,
          prevent_cse=not scan_pipeline_iters,
      )

    # ---- Execute: outer scan (repeats) + bubble scan ----
    num_repeats = self.config.num_pipeline_repeats
    # Initial w_curr: zeros with BSW shape (repeat dim removed from layers_params)
    initial_w_curr = jax.tree.map(lambda x: jnp.zeros(x.shape[1:], dtype=x.dtype), layers_params)

    # === #3: before outer scan -- carry structure + each carry element shape ===
    _n3_log("=" * 60)
    _n3_log(f"BEFORE OUTER SCAN (repeats): length={num_repeats} scan_pipeline_iters={scan_pipeline_iters}")
    _n3_log(f"  carry[0] loop_state keys={_tree_keys(loop_state)}")
    _n3_log(
        f"  carry[1] layers_mutables leaves={_leaf_count(layers_mutables)} "
        f"first_shapes={_shapes(layers_mutables)}"
    )
    _n3_log(
        f"  carry[2] initial_w_curr leaves={_leaf_count(initial_w_curr)} "
        f"bytes={_byte_size(initial_w_curr)/1e9:.3f} GB first_shapes={_shapes(initial_w_curr, limit=3)} "
        "(zeros, repeat-dim removed from layers_params -- this is the BSW-shaped carry)"
    )

    if scan_pipeline_iters:
      unroll_repeats = num_repeats if not self.config.scan_pipeline_repeats else 1
      (loop_state, final_layer_mutables, final_w_curr), repeat_metrics = jax.lax.scan(
          outer_body,
          (loop_state, layers_mutables, initial_w_curr),
          None,
          length=num_repeats,
          unroll=unroll_repeats,
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
          if repeat_metrics_list
          else layers_metrics
      )

    # ---- Bubble iterations (pipeline drain) ----
    if bubble_iterations > 0:
      bsw_ref[0] = (final_w_curr, final_w_curr)
      # === #11: ENTER bubble -- length, BSW shapes ===
      _n3_log("=" * 60)
      _n3_log(
        f"ENTER bubble scan: length(bubble_iterations)={bubble_iterations} "
        f"bsw=(final_w_curr, final_w_curr) leaves={_leaf_count(bsw_ref[0])} "
        f"bytes={_byte_size(bsw_ref[0])/1e9:.3f} GB first_shape={_shapes(bsw_ref[0], limit=1)}"
      )

      if scan_pipeline_iters:
        (loop_state, final_layer_mutables), bubble_metrics = jax.lax.scan(
            bubble_inner_body, (loop_state, final_layer_mutables), None, length=bubble_iterations
        )
      else:
        bubble_carry = (loop_state, final_layer_mutables)
        bubble_metrics_list = []
        for _ in range(bubble_iterations):
          bubble_carry, bub_metrics = bubble_inner_body(bubble_carry, None)
          bubble_metrics_list.append(bub_metrics)
        loop_state, final_layer_mutables = bubble_carry
        bubble_metrics = (
            jax.tree.map(lambda *xs: jnp.stack(xs), *bubble_metrics_list) if bubble_metrics_list else layers_metrics
        )

      stacked_metrics = jax.tree.map(lambda r, b: jnp.concatenate([r, b], axis=0), repeat_metrics, bubble_metrics)
    else:
      stacked_metrics = repeat_metrics

    # === #12: before nnx.update -- final state structure ===
    _n3_log("=" * 60)
    _n3_log("BEFORE nnx.update(self.layers, final_layer_state):")
    _n3_log(
        f"  final_layer_mutables leaves={_leaf_count(final_layer_mutables)} "
        f"first_shapes={_shapes(final_layer_mutables)}"
    )
    _n3_log(
        f"  stacked_metrics leaves={_leaf_count(stacked_metrics)} "
        f"first_shapes={_shapes(stacked_metrics, limit=3)}"
    )
    final_layer_state = nnx.State.merge(layers_params, stacked_metrics, final_layer_mutables)
    _n3_log(f"  merged final_layer_state leaves={_leaf_count(final_layer_state)}")
    nnx.update(self.layers, final_layer_state)

    final_output = self.realign_output_microbatches(loop_state["state_io"])
    _n3_log(
        f"  final_output (pre-reshape) shape={tuple(final_output.shape)} dtype={final_output.dtype}"
    )
    _n3_log("NNXCircularPipeline.__call__ EXIT")
    _n3_log("=" * 70)
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
