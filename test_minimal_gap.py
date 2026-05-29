"""Minimal reproducer: isolate NNX vs Linen memory gap via progressive complexity.

Each level adds one dimension of pipeline complexity. By comparing
memory_analysis().temp_size_in_bytes at each level we can pinpoint
EXACTLY where the gap opens up.

Levels:
  0  Pure functional scan (baseline)
  1  NNX merge/split in scan body
  2  Linen model.apply in scan body
  3a NNX + vmap over stages
  3b Linen + vmap over stages
  4a NNX nested scans (outer repeats + inner microbatches)
  4b Linen nested scans
  5a NNX + BSW dual-buffer
  5b Linen + BSW dual-buffer
  6a NNX + scan ys (return metrics per iteration)
  6b Linen + scan ys
  7a NNX full pipeline complexity
  7b Linen full pipeline complexity

Usage:
  python test_minimal_gap.py                     # all levels
  python test_minimal_gap.py 0 1 2               # specific levels
  python test_minimal_gap.py --worker 0           # subprocess (internal)
  python test_minimal_gap.py --scale 2            # 2x dimensions

IMPORTANT: Must run on TPU for accurate memory_analysis(). Do NOT set
JAX_PLATFORMS=cpu.
"""

import gc
import json
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Parse args early (before JAX import)
# ---------------------------------------------------------------------------
def _parse_scale():
    for i, arg in enumerate(sys.argv):
        if arg == "--scale" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return 1

SCALE = _parse_scale()

# ---------------------------------------------------------------------------
# Constants — Llama2-7B dimensions, scaled
# ---------------------------------------------------------------------------
DIM = 4096 * SCALE
FF_DIM = 11008 * SCALE
BATCH = 8
SEQ = 32
NUM_STAGES = 2
NUM_REPEATS = 2
NUM_MICROBATCHES = 4
OUTER_N = NUM_REPEATS
INNER_N = NUM_MICROBATCHES
DTYPE_STR = "bfloat16"

# ---------------------------------------------------------------------------
# JAX setup
# ---------------------------------------------------------------------------
def setup_jax():
    """Import and configure JAX. Returns (jax, jnp, nnx, nn, mesh, DTYPE)."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from flax import linen as nn
    from jax.sharding import Mesh

    DTYPE = jnp.bfloat16

    devices = jax.devices()
    if len(devices) < NUM_STAGES:
        print(f"WARNING: need {NUM_STAGES} devices, got {len(devices)}. "
              f"Will proceed but sharding tests may differ.")

    mesh = Mesh(
        np.array(devices[:max(NUM_STAGES, len(devices))]).reshape(-1),
        axis_names=("stage",),
    )
    return jax, jnp, nnx, nn, mesh, DTYPE


def get_device_memory(jax):
    """Get HBM memory stats from first device."""
    for d in jax.local_devices():
        stats = d.memory_stats()
        if stats:
            return {
                "peak_gb": stats.get("peak_bytes_in_use", 0) / 1e9,
                "current_gb": stats.get("bytes_in_use", 0) / 1e9,
                "limit_gb": stats.get("bytes_limit", 0) / 1e9,
            }
    return None


# ============================================================================
# Model classes
# ============================================================================
def make_ffn_nnx(nnx, jnp, DTYPE):
    """Create an NNX FFN module."""
    from flax import nnx as nnx_mod

    class FFN_NNX(nnx_mod.Module):
        def __init__(self, dim, ff_dim, rngs):
            self.w1 = nnx_mod.Param(
                jax_import().random.normal(rngs.params(), (dim, ff_dim), dtype=DTYPE) * 0.01
            )
            self.w2 = nnx_mod.Param(
                jax_import().random.normal(rngs.params(), (ff_dim, dim), dtype=DTYPE) * 0.01
            )

        def __call__(self, x):
            import jax.numpy as _jnp
            return _jnp.dot(
                jax_import().nn.relu(_jnp.dot(x, self.w1.value)),
                self.w2.value,
            )

    return FFN_NNX


def jax_import():
    import jax
    return jax


def make_ffn_linen(nn, DTYPE):
    """Create a Linen FFN module class."""
    from flax import linen as nn_mod

    class FFN_Linen(nn_mod.Module):
        dim: int
        ff_dim: int
        dtype: type = DTYPE

        @nn_mod.compact
        def __call__(self, x):
            import jax
            import jax.numpy as _jnp
            w1 = self.param(
                "w1", nn_mod.initializers.normal(0.01), (self.dim, self.ff_dim),
            )
            w2 = self.param(
                "w2", nn_mod.initializers.normal(0.01), (self.ff_dim, self.dim),
            )
            w1 = w1.astype(self.dtype)
            w2 = w2.astype(self.dtype)
            return _jnp.dot(jax.nn.relu(_jnp.dot(x, w1)), w2)

    return FFN_Linen


# ============================================================================
# Level builders
# ============================================================================

def build_level_0(jax, jnp, nnx, nn, mesh, DTYPE):
    """Level 0: Pure functional scan (baseline).

    Weight captured by closure, scan + checkpoint.
    """
    w1 = jax.random.normal(jax.random.PRNGKey(0), (DIM, FF_DIM), dtype=DTYPE) * 0.01
    w2 = jax.random.normal(jax.random.PRNGKey(1), (FF_DIM, DIM), dtype=DTYPE) * 0.01
    x = jax.random.normal(jax.random.PRNGKey(2), (BATCH, SEQ, DIM), dtype=DTYPE)

    def loss_pure(w1, w2, x):
        def body(carry, _):
            x = carry
            x = jnp.dot(jax.nn.relu(jnp.dot(x, w1)), w2)
            return x, None

        body = jax.checkpoint(body)
        final, _ = jax.lax.scan(body, x, None, length=INNER_N)
        return jnp.sum(final)

    grad_fn = jax.value_and_grad(loss_pure, argnums=(0, 1))
    args = (w1, w2, x)
    return grad_fn, args, "Level 0 (Pure scan)"


def build_level_1(jax, jnp, nnx, nn, mesh, DTYPE):
    """Level 1: NNX merge/split in scan body.

    Same computation as Level 0, but with nnx.merge -> call -> nnx.split cycle.
    """
    FFN_NNX = make_ffn_nnx(nnx, jnp, DTYPE)
    module = FFN_NNX(DIM, FF_DIM, nnx.Rngs(0))
    graph, state = nnx.split(module)

    x = jax.random.normal(jax.random.PRNGKey(2), (BATCH, SEQ, DIM), dtype=DTYPE)

    def loss_nnx(state, x):
        def body(carry, _):
            x = carry
            m = nnx.merge(graph, state)
            x = m(x)
            return x, None

        body = jax.checkpoint(body)
        final, _ = jax.lax.scan(body, x, None, length=INNER_N)
        return jnp.sum(final)

    grad_fn = jax.value_and_grad(loss_nnx)
    args = (state, x)
    return grad_fn, args, "Level 1 (NNX merge/split)"


def build_level_2(jax, jnp, nnx, nn, mesh, DTYPE):
    """Level 2: Linen model.apply in scan body.

    Same computation as Level 0, but with model.apply.
    """
    FFN_Linen = make_ffn_linen(nn, DTYPE)
    model = FFN_Linen(dim=DIM, ff_dim=FF_DIM, dtype=DTYPE)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((BATCH, SEQ, DIM), dtype=DTYPE))

    x = jax.random.normal(jax.random.PRNGKey(2), (BATCH, SEQ, DIM), dtype=DTYPE)

    def loss_linen(params, x):
        def body(carry, _):
            x = carry
            x = model.apply(params, x)
            return x, None

        body = jax.checkpoint(body)
        final, _ = jax.lax.scan(body, x, None, length=INNER_N)
        return jnp.sum(final)

    grad_fn = jax.value_and_grad(loss_linen)
    args = (params, x)
    return grad_fn, args, "Level 2 (Linen apply)"


def build_level_3a(jax, jnp, nnx, nn, mesh, DTYPE):
    """Level 3a: NNX + vmap over stages.

    Stacked params [num_stages, dim, ff_dim], vmap over stage axis.
    """
    # Create one module per stage, stack their states
    modules = []
    for i in range(NUM_STAGES):
        m = make_ffn_nnx(nnx, jnp, DTYPE)(DIM, FF_DIM, nnx.Rngs(i))
        modules.append(m)

    graph0, state0 = nnx.split(modules[0])
    # Stack states across stages
    states = [nnx.split(m)[1] for m in modules]
    stacked_state = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *states)

    x_per_stage = jax.random.normal(
        jax.random.PRNGKey(2), (NUM_STAGES, BATCH, SEQ, DIM), dtype=DTYPE
    )

    def loss_nnx_vmap(stacked_state, x_per_stage):
        def body(carry, _):
            x = carry

            def per_stage(state_flat_arrays, x_i):
                # Reconstruct per-stage state from the flat arrays
                st = jax.tree.map(lambda s, a: a, state0, state_flat_arrays)
                m = nnx.merge(graph0, st)
                return m(x_i)

            x = jax.vmap(per_stage)(stacked_state, x)
            return x, None

        body = jax.checkpoint(body)
        final, _ = jax.lax.scan(body, x_per_stage, None, length=INNER_N)
        return jnp.sum(final)

    grad_fn = jax.value_and_grad(loss_nnx_vmap)
    args = (stacked_state, x_per_stage)
    return grad_fn, args, "Level 3a (NNX + vmap)"


def build_level_3b(jax, jnp, nnx, nn, mesh, DTYPE):
    """Level 3b: Linen + vmap over stages.

    Stacked params [num_stages, dim, ff_dim], vmap over stage axis.
    """
    FFN_Linen = make_ffn_linen(nn, DTYPE)
    model = FFN_Linen(dim=DIM, ff_dim=FF_DIM, dtype=DTYPE)
    single_params = model.init(
        jax.random.PRNGKey(0), jnp.ones((BATCH, SEQ, DIM), dtype=DTYPE)
    )
    # Stack params across stages
    stacked_params = jax.tree.map(
        lambda p: jnp.stack([p] * NUM_STAGES, axis=0), single_params
    )

    x_per_stage = jax.random.normal(
        jax.random.PRNGKey(2), (NUM_STAGES, BATCH, SEQ, DIM), dtype=DTYPE
    )

    def loss_linen_vmap(stacked_params, x_per_stage):
        def body(carry, _):
            x = carry

            def per_stage(params_i, x_i):
                return model.apply(params_i, x_i)

            x = jax.vmap(per_stage)(stacked_params, x)
            return x, None

        body = jax.checkpoint(body)
        final, _ = jax.lax.scan(body, x_per_stage, None, length=INNER_N)
        return jnp.sum(final)

    grad_fn = jax.value_and_grad(loss_linen_vmap)
    args = (stacked_params, x_per_stage)
    return grad_fn, args, "Level 3b (Linen + vmap)"


def build_level_4a(jax, jnp, nnx, nn, mesh, DTYPE):
    """Level 4a: NNX nested scans (outer repeats + inner microbatches).

    outer scan (repeats) -> inner scan (microbatches), like pipeline.
    """
    FFN_NNX = make_ffn_nnx(nnx, jnp, DTYPE)
    modules = [FFN_NNX(DIM, FF_DIM, nnx.Rngs(i)) for i in range(NUM_STAGES)]
    graph0, state0 = nnx.split(modules[0])
    states = [nnx.split(m)[1] for m in modules]
    stacked_state = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *states)

    x_per_stage = jax.random.normal(
        jax.random.PRNGKey(2), (NUM_STAGES, BATCH, SEQ, DIM), dtype=DTYPE
    )

    def loss_nested_nnx(stacked_state, x_per_stage):
        def inner_body(carry, _):
            x = carry

            def per_stage(state_i, x_i):
                m = nnx.merge(graph0, jax.tree.map(lambda s, a: a, state0, state_i))
                return m(x_i)

            x = jax.vmap(per_stage)(stacked_state, x)
            return x, None

        inner_body_ckpt = jax.checkpoint(inner_body)

        def outer_body(carry, _):
            x = carry
            final_x, _ = jax.lax.scan(inner_body_ckpt, x, None, length=INNER_N)
            return final_x, None

        final, _ = jax.lax.scan(outer_body, x_per_stage, None, length=OUTER_N)
        return jnp.sum(final)

    grad_fn = jax.value_and_grad(loss_nested_nnx)
    args = (stacked_state, x_per_stage)
    return grad_fn, args, "Level 4a (NNX nested scan)"


def build_level_4b(jax, jnp, nnx, nn, mesh, DTYPE):
    """Level 4b: Linen nested scans (outer repeats + inner microbatches)."""
    FFN_Linen = make_ffn_linen(nn, DTYPE)
    model = FFN_Linen(dim=DIM, ff_dim=FF_DIM, dtype=DTYPE)
    single_params = model.init(
        jax.random.PRNGKey(0), jnp.ones((BATCH, SEQ, DIM), dtype=DTYPE)
    )
    stacked_params = jax.tree.map(
        lambda p: jnp.stack([p] * NUM_STAGES, axis=0), single_params
    )

    x_per_stage = jax.random.normal(
        jax.random.PRNGKey(2), (NUM_STAGES, BATCH, SEQ, DIM), dtype=DTYPE
    )

    def loss_nested_linen(stacked_params, x_per_stage):
        def inner_body(carry, _):
            x = carry

            def per_stage(params_i, x_i):
                return model.apply(params_i, x_i)

            x = jax.vmap(per_stage)(stacked_params, x)
            return x, None

        inner_body_ckpt = jax.checkpoint(inner_body)

        def outer_body(carry, _):
            x = carry
            final_x, _ = jax.lax.scan(inner_body_ckpt, x, None, length=INNER_N)
            return final_x, None

        final, _ = jax.lax.scan(outer_body, x_per_stage, None, length=OUTER_N)
        return jnp.sum(final)

    grad_fn = jax.value_and_grad(loss_nested_linen)
    args = (stacked_params, x_per_stage)
    return grad_fn, args, "Level 4b (Linen nested scan)"


def build_level_5a(jax, jnp, nnx, nn, mesh, DTYPE):
    """Level 5a: NNX + BSW dual-buffer.

    Outer carry includes gathered weights (w_curr, w_next), like pipeline BSW.
    """
    FFN_NNX = make_ffn_nnx(nnx, jnp, DTYPE)
    # Create stacked weights: [num_repeats, num_stages, dim, ff_dim]
    modules = []
    for r in range(NUM_REPEATS):
        stage_mods = []
        for s in range(NUM_STAGES):
            m = FFN_NNX(DIM, FF_DIM, nnx.Rngs(r * NUM_STAGES + s))
            stage_mods.append(m)
        modules.append(stage_mods)

    graph0, state0 = nnx.split(modules[0][0])
    # Stack all states: [repeats, stages, ...]
    all_states = []
    for r in range(NUM_REPEATS):
        stage_states = [nnx.split(m)[1] for m in modules[r]]
        stacked_stage = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *stage_states)
        all_states.append(stacked_stage)
    full_state = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *all_states)
    # full_state leaves have shape [repeats, stages, ...]

    x_per_stage = jax.random.normal(
        jax.random.PRNGKey(2), (NUM_STAGES, BATCH, SEQ, DIM), dtype=DTYPE
    )

    def gather_weights(full_st, repeat_id):
        """Slice weights for a single repeat: [stages, ...]"""
        return jax.tree.map(
            lambda w: jax.lax.dynamic_slice_in_dim(w, repeat_id, 1, axis=0)[0],
            full_st,
        )

    def loss_bsw_nnx(full_state, x_per_stage):
        w_init = gather_weights(full_state, 0)

        def outer_body(carry, _):
            x, w_curr, repeat_id = carry
            # Prefetch next repeat's weights (dual buffer)
            next_id = jnp.minimum(repeat_id + 1, NUM_REPEATS - 1)
            w_next = gather_weights(full_state, next_id)

            def inner_body(inner_carry, _):
                x = inner_carry

                def per_stage(state_i, x_i):
                    m = nnx.merge(graph0, jax.tree.map(lambda s, a: a, state0, state_i))
                    return m(x_i)

                x = jax.vmap(per_stage)(w_curr, x)
                return x, None

            inner_body_ckpt = jax.checkpoint(inner_body)
            final_x, _ = jax.lax.scan(inner_body_ckpt, x, None, length=INNER_N)
            return (final_x, w_next, repeat_id + 1), None

        (final_x, _, _), _ = jax.lax.scan(
            outer_body,
            (x_per_stage, w_init, jnp.int32(0)),
            None,
            length=OUTER_N,
        )
        return jnp.sum(final_x)

    grad_fn = jax.value_and_grad(loss_bsw_nnx)
    args = (full_state, x_per_stage)
    return grad_fn, args, "Level 5a (NNX + BSW dual-buf)"


def build_level_5b(jax, jnp, nnx, nn, mesh, DTYPE):
    """Level 5b: Linen + BSW dual-buffer."""
    FFN_Linen = make_ffn_linen(nn, DTYPE)
    model = FFN_Linen(dim=DIM, ff_dim=FF_DIM, dtype=DTYPE)
    single_params = model.init(
        jax.random.PRNGKey(0), jnp.ones((BATCH, SEQ, DIM), dtype=DTYPE)
    )
    # Stack: [repeats, stages, ...]
    stacked_stage = jax.tree.map(
        lambda p: jnp.stack([p] * NUM_STAGES, axis=0), single_params
    )
    full_params = jax.tree.map(
        lambda p: jnp.stack([p] * NUM_REPEATS, axis=0), stacked_stage
    )

    x_per_stage = jax.random.normal(
        jax.random.PRNGKey(2), (NUM_STAGES, BATCH, SEQ, DIM), dtype=DTYPE
    )

    def gather_weights(full_p, repeat_id):
        return jax.tree.map(
            lambda w: jax.lax.dynamic_slice_in_dim(w, repeat_id, 1, axis=0)[0],
            full_p,
        )

    def loss_bsw_linen(full_params, x_per_stage):
        w_init = gather_weights(full_params, 0)

        def outer_body(carry, _):
            x, w_curr, repeat_id = carry
            next_id = jnp.minimum(repeat_id + 1, NUM_REPEATS - 1)
            w_next = gather_weights(full_params, next_id)

            def inner_body(inner_carry, _):
                x = inner_carry

                def per_stage(params_i, x_i):
                    return model.apply(params_i, x_i)

                x = jax.vmap(per_stage)(w_curr, x)
                return x, None

            inner_body_ckpt = jax.checkpoint(inner_body)
            final_x, _ = jax.lax.scan(inner_body_ckpt, x, None, length=INNER_N)
            return (final_x, w_next, repeat_id + 1), None

        (final_x, _, _), _ = jax.lax.scan(
            outer_body,
            (x_per_stage, w_init, jnp.int32(0)),
            None,
            length=OUTER_N,
        )
        return jnp.sum(final_x)

    grad_fn = jax.value_and_grad(loss_bsw_linen)
    args = (full_params, x_per_stage)
    return grad_fn, args, "Level 5b (Linen + BSW dual-buf)"


def build_level_6a(jax, jnp, nnx, nn, mesh, DTYPE):
    """Level 6a: NNX + BSW + scan ys (return metrics per iteration).

    Same as Level 5a but inner scan returns per-iteration metrics as ys,
    like pipeline returning per-microbatch loss contributions.
    """
    FFN_NNX = make_ffn_nnx(nnx, jnp, DTYPE)
    modules = []
    for r in range(NUM_REPEATS):
        stage_mods = []
        for s in range(NUM_STAGES):
            m = FFN_NNX(DIM, FF_DIM, nnx.Rngs(r * NUM_STAGES + s))
            stage_mods.append(m)
        modules.append(stage_mods)

    graph0, state0 = nnx.split(modules[0][0])
    all_states = []
    for r in range(NUM_REPEATS):
        stage_states = [nnx.split(m)[1] for m in modules[r]]
        stacked_stage = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *stage_states)
        all_states.append(stacked_stage)
    full_state = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *all_states)

    x_per_stage = jax.random.normal(
        jax.random.PRNGKey(2), (NUM_STAGES, BATCH, SEQ, DIM), dtype=DTYPE
    )

    def gather_weights(full_st, repeat_id):
        return jax.tree.map(
            lambda w: jax.lax.dynamic_slice_in_dim(w, repeat_id, 1, axis=0)[0],
            full_st,
        )

    def loss_bsw_ys_nnx(full_state, x_per_stage):
        w_init = gather_weights(full_state, 0)

        def outer_body(carry, _):
            x, w_curr, repeat_id = carry
            next_id = jnp.minimum(repeat_id + 1, NUM_REPEATS - 1)
            w_next = gather_weights(full_state, next_id)

            def inner_body(inner_carry, _):
                x = inner_carry

                def per_stage(state_i, x_i):
                    m = nnx.merge(graph0, jax.tree.map(lambda s, a: a, state0, state_i))
                    return m(x_i)

                x = jax.vmap(per_stage)(w_curr, x)
                # Return per-iteration metrics as ys
                metric = jnp.sum(x ** 2)
                return x, metric

            inner_body_ckpt = jax.checkpoint(inner_body)
            final_x, metrics = jax.lax.scan(inner_body_ckpt, x, None, length=INNER_N)
            return (final_x, w_next, repeat_id + 1), metrics

        (final_x, _, _), all_metrics = jax.lax.scan(
            outer_body,
            (x_per_stage, w_init, jnp.int32(0)),
            None,
            length=OUTER_N,
        )
        return jnp.sum(final_x) + jnp.sum(all_metrics) * 0.0  # metrics don't affect loss

    grad_fn = jax.value_and_grad(loss_bsw_ys_nnx)
    args = (full_state, x_per_stage)
    return grad_fn, args, "Level 6a (NNX + BSW + ys)"


def build_level_6b(jax, jnp, nnx, nn, mesh, DTYPE):
    """Level 6b: Linen + BSW + scan ys."""
    FFN_Linen = make_ffn_linen(nn, DTYPE)
    model = FFN_Linen(dim=DIM, ff_dim=FF_DIM, dtype=DTYPE)
    single_params = model.init(
        jax.random.PRNGKey(0), jnp.ones((BATCH, SEQ, DIM), dtype=DTYPE)
    )
    stacked_stage = jax.tree.map(
        lambda p: jnp.stack([p] * NUM_STAGES, axis=0), single_params
    )
    full_params = jax.tree.map(
        lambda p: jnp.stack([p] * NUM_REPEATS, axis=0), stacked_stage
    )

    x_per_stage = jax.random.normal(
        jax.random.PRNGKey(2), (NUM_STAGES, BATCH, SEQ, DIM), dtype=DTYPE
    )

    def gather_weights(full_p, repeat_id):
        return jax.tree.map(
            lambda w: jax.lax.dynamic_slice_in_dim(w, repeat_id, 1, axis=0)[0],
            full_p,
        )

    def loss_bsw_ys_linen(full_params, x_per_stage):
        w_init = gather_weights(full_params, 0)

        def outer_body(carry, _):
            x, w_curr, repeat_id = carry
            next_id = jnp.minimum(repeat_id + 1, NUM_REPEATS - 1)
            w_next = gather_weights(full_params, next_id)

            def inner_body(inner_carry, _):
                x = inner_carry

                def per_stage(params_i, x_i):
                    return model.apply(params_i, x_i)

                x = jax.vmap(per_stage)(w_curr, x)
                metric = jnp.sum(x ** 2)
                return x, metric

            inner_body_ckpt = jax.checkpoint(inner_body)
            final_x, metrics = jax.lax.scan(inner_body_ckpt, x, None, length=INNER_N)
            return (final_x, w_next, repeat_id + 1), metrics

        (final_x, _, _), all_metrics = jax.lax.scan(
            outer_body,
            (x_per_stage, w_init, jnp.int32(0)),
            None,
            length=OUTER_N,
        )
        return jnp.sum(final_x) + jnp.sum(all_metrics) * 0.0

    grad_fn = jax.value_and_grad(loss_bsw_ys_linen)
    args = (full_params, x_per_stage)
    return grad_fn, args, "Level 6b (Linen + BSW + ys)"


def build_level_7a(jax, jnp, nnx, nn, mesh, DTYPE):
    """Level 7a: NNX full pipeline complexity.

    All features: BSW dual-buffer, scan ys, shift/rotate state,
    microbatch routing via state_io buffer, repeat_ids in carry.
    """
    FFN_NNX = make_ffn_nnx(nnx, jnp, DTYPE)
    modules = []
    for r in range(NUM_REPEATS):
        stage_mods = []
        for s in range(NUM_STAGES):
            m = FFN_NNX(DIM, FF_DIM, nnx.Rngs(r * NUM_STAGES + s))
            stage_mods.append(m)
        modules.append(stage_mods)

    graph0, state0 = nnx.split(modules[0][0])
    all_states = []
    for r in range(NUM_REPEATS):
        stage_states = [nnx.split(m)[1] for m in modules[r]]
        stacked_stage = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *stage_states)
        all_states.append(stacked_stage)
    full_state = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *all_states)

    x_per_stage = jax.random.normal(
        jax.random.PRNGKey(2), (NUM_STAGES, BATCH, SEQ, DIM), dtype=DTYPE
    )

    def gather_weights(full_st, repeat_id):
        return jax.tree.map(
            lambda w: jax.lax.dynamic_slice_in_dim(w, repeat_id, 1, axis=0)[0],
            full_st,
        )

    def shift_state(state_io, shift):
        """Circular shift of state_io along stage axis (like pipeline shift)."""
        return jnp.roll(state_io, shift=shift, axis=0)

    def loss_full_nnx(full_state, x_per_stage):
        w_init = gather_weights(full_state, 0)
        # state_io: [stages, microbatches+1, batch, seq, dim] (padded IO buffer)
        state_io = jnp.zeros(
            (NUM_STAGES, NUM_MICROBATCHES + 1, BATCH, SEQ, DIM), dtype=DTYPE
        )
        # Insert initial activations into state_io
        state_io = state_io.at[:, 0, :, :, :].set(x_per_stage)

        def outer_body(carry, _):
            state_io, w_curr, repeat_id, shift = carry
            next_id = jnp.minimum(repeat_id + 1, NUM_REPEATS - 1)
            w_next = gather_weights(full_state, next_id)

            def inner_body(inner_carry, _):
                state_io, mb_id = inner_carry
                # Extract current microbatch from state_io
                x = jax.lax.dynamic_slice_in_dim(state_io, mb_id, 1, axis=1)[:, 0]

                def per_stage(state_i, x_i):
                    m = nnx.merge(graph0, jax.tree.map(lambda s, a: a, state0, state_i))
                    return m(x_i)

                x_out = jax.vmap(per_stage)(w_curr, x)
                # Write result back to state_io at next slot
                next_slot = mb_id + 1
                state_io_new = jax.lax.dynamic_update_slice_in_dim(
                    state_io, x_out[:, None], next_slot, axis=1
                )
                metric = jnp.sum(x_out ** 2)
                return (state_io_new, mb_id + 1), metric

            inner_body_ckpt = jax.checkpoint(inner_body)
            (state_io_new, _), metrics = jax.lax.scan(
                inner_body_ckpt, (state_io, jnp.int32(0)), None, length=INNER_N
            )
            # Shift state between repeats (circular pipeline)
            state_io_shifted = shift_state(state_io_new, 1)
            return (state_io_shifted, w_next, repeat_id + 1, shift + 1), metrics

        (final_state_io, _, _, _), all_metrics = jax.lax.scan(
            outer_body,
            (state_io, w_init, jnp.int32(0), jnp.int32(0)),
            None,
            length=OUTER_N,
        )
        # Extract final activations from last slot
        final_x = final_state_io[:, -1]
        return jnp.sum(final_x) + jnp.sum(all_metrics) * 0.0

    grad_fn = jax.value_and_grad(loss_full_nnx)
    args = (full_state, x_per_stage)
    return grad_fn, args, "Level 7a (NNX full pipeline)"


def build_level_7b(jax, jnp, nnx, nn, mesh, DTYPE):
    """Level 7b: Linen full pipeline complexity."""
    FFN_Linen = make_ffn_linen(nn, DTYPE)
    model = FFN_Linen(dim=DIM, ff_dim=FF_DIM, dtype=DTYPE)
    single_params = model.init(
        jax.random.PRNGKey(0), jnp.ones((BATCH, SEQ, DIM), dtype=DTYPE)
    )
    stacked_stage = jax.tree.map(
        lambda p: jnp.stack([p] * NUM_STAGES, axis=0), single_params
    )
    full_params = jax.tree.map(
        lambda p: jnp.stack([p] * NUM_REPEATS, axis=0), stacked_stage
    )

    x_per_stage = jax.random.normal(
        jax.random.PRNGKey(2), (NUM_STAGES, BATCH, SEQ, DIM), dtype=DTYPE
    )

    def gather_weights(full_p, repeat_id):
        return jax.tree.map(
            lambda w: jax.lax.dynamic_slice_in_dim(w, repeat_id, 1, axis=0)[0],
            full_p,
        )

    def shift_state(state_io, shift):
        return jnp.roll(state_io, shift=shift, axis=0)

    def loss_full_linen(full_params, x_per_stage):
        w_init = gather_weights(full_params, 0)
        state_io = jnp.zeros(
            (NUM_STAGES, NUM_MICROBATCHES + 1, BATCH, SEQ, DIM), dtype=DTYPE
        )
        state_io = state_io.at[:, 0, :, :, :].set(x_per_stage)

        def outer_body(carry, _):
            state_io, w_curr, repeat_id, shift = carry
            next_id = jnp.minimum(repeat_id + 1, NUM_REPEATS - 1)
            w_next = gather_weights(full_params, next_id)

            def inner_body(inner_carry, _):
                state_io, mb_id = inner_carry
                x = jax.lax.dynamic_slice_in_dim(state_io, mb_id, 1, axis=1)[:, 0]

                def per_stage(params_i, x_i):
                    return model.apply(params_i, x_i)

                x_out = jax.vmap(per_stage)(w_curr, x)
                next_slot = mb_id + 1
                state_io_new = jax.lax.dynamic_update_slice_in_dim(
                    state_io, x_out[:, None], next_slot, axis=1
                )
                metric = jnp.sum(x_out ** 2)
                return (state_io_new, mb_id + 1), metric

            inner_body_ckpt = jax.checkpoint(inner_body)
            (state_io_new, _), metrics = jax.lax.scan(
                inner_body_ckpt, (state_io, jnp.int32(0)), None, length=INNER_N
            )
            state_io_shifted = shift_state(state_io_new, 1)
            return (state_io_shifted, w_next, repeat_id + 1, shift + 1), metrics

        (final_state_io, _, _, _), all_metrics = jax.lax.scan(
            outer_body,
            (state_io, w_init, jnp.int32(0), jnp.int32(0)),
            None,
            length=OUTER_N,
        )
        final_x = final_state_io[:, -1]
        return jnp.sum(final_x) + jnp.sum(all_metrics) * 0.0

    grad_fn = jax.value_and_grad(loss_full_linen)
    args = (full_params, x_per_stage)
    return grad_fn, args, "Level 7b (Linen full pipeline)"


# ============================================================================
# Test registry
# ============================================================================
# Maps level_id -> (builder_fn, level_name)
LEVEL_BUILDERS = {
    "0":  build_level_0,
    "1":  build_level_1,
    "2":  build_level_2,
    "3a": build_level_3a,
    "3b": build_level_3b,
    "4a": build_level_4a,
    "4b": build_level_4b,
    "5a": build_level_5a,
    "5b": build_level_5b,
    "6a": build_level_6a,
    "6b": build_level_6b,
    "7a": build_level_7a,
    "7b": build_level_7b,
}

ALL_LEVELS = ["0", "1", "2", "3a", "3b", "4a", "4b", "5a", "5b", "6a", "6b", "7a", "7b"]


# ============================================================================
# Runner
# ============================================================================

def run_single_level(level_id):
    """Run a single level and return memory analysis results."""
    jax, jnp, nnx, nn, mesh, DTYPE = setup_jax()

    builder = LEVEL_BUILDERS[level_id]
    grad_fn, args, level_name = builder(jax, jnp, nnx, nn, mesh, DTYPE)

    # Wait for args to be ready
    jax.block_until_ready(jax.tree.leaves(args))

    # Compute weight size
    weight_leaves = jax.tree.leaves(args[0])
    total_weight_bytes = sum(w.nbytes for w in weight_leaves)

    mem_before = get_device_memory(jax)

    # Compile
    t0 = time.time()
    try:
        compiled = jax.jit(grad_fn).lower(*args).compile()
    except Exception as e:
        return {
            "level_id": level_id,
            "name": level_name,
            "status": f"COMPILE_FAILED: {str(e)[:300]}",
        }
    compile_time = time.time() - t0

    # Extract memory_analysis
    xla_temp_bytes = 0
    xla_arg_bytes = 0
    xla_out_bytes = 0
    xla_total_bytes = 0
    xla_code_bytes = 0
    try:
        mem_analysis = compiled.memory_analysis()
        if mem_analysis is not None:
            # Handle both dict-like and object-like returns
            if isinstance(mem_analysis, dict):
                xla_temp_bytes = mem_analysis.get("temp_size_in_bytes", 0)
                xla_arg_bytes = mem_analysis.get("argument_size_in_bytes", 0)
                xla_out_bytes = mem_analysis.get("output_size_in_bytes", 0)
                xla_total_bytes = mem_analysis.get("total_size_in_bytes",
                                                   xla_temp_bytes + xla_arg_bytes + xla_out_bytes)
                xla_code_bytes = mem_analysis.get("generated_code_size_in_bytes", 0)
            else:
                xla_temp_bytes = getattr(mem_analysis, "temp_size_in_bytes", 0) or 0
                xla_arg_bytes = getattr(mem_analysis, "argument_size_in_bytes", 0) or 0
                xla_out_bytes = getattr(mem_analysis, "output_size_in_bytes", 0) or 0
                xla_total_bytes = (getattr(mem_analysis, "total_size_in_bytes", 0) or 0) or \
                                  (xla_temp_bytes + xla_arg_bytes + xla_out_bytes)
                xla_code_bytes = getattr(mem_analysis, "generated_code_size_in_bytes", 0) or 0
    except Exception as e:
        pass  # memory_analysis may not be available on all platforms

    # Run forward+backward
    t0 = time.time()
    try:
        result = compiled(*args)
        jax.block_until_ready(jax.tree.leaves(result))
        run_time = time.time() - t0
        status = "OK"
    except Exception as e:
        run_time = time.time() - t0
        status = f"RUN_FAILED: {str(e)[:300]}"
        result = None

    mem_after = get_device_memory(jax)

    # Cleanup
    del result, compiled
    gc.collect()

    return {
        "level_id": level_id,
        "name": level_name,
        "status": status,
        "scale": SCALE,
        "weight_gb": round(total_weight_bytes / 1e9, 4),
        "compile_s": round(compile_time, 2),
        "run_s": round(run_time, 3) if run_time else -1,
        # XLA memory_analysis
        "xla_temp_gb": round(xla_temp_bytes / 1e9, 4),
        "xla_arg_gb": round(xla_arg_bytes / 1e9, 4),
        "xla_out_gb": round(xla_out_bytes / 1e9, 4),
        "xla_total_gb": round(xla_total_bytes / 1e9, 4),
        "xla_code_kb": round(xla_code_bytes / 1e3, 1),
        # Device memory
        "peak_gb": round(mem_after["peak_gb"], 3) if mem_after else -1,
        "hbm_limit_gb": round(mem_before["limit_gb"], 3) if mem_before else -1,
    }


def worker_main(level_id):
    """Entry point for subprocess worker."""
    try:
        result = run_single_level(level_id)
        print(json.dumps(result), flush=True)
    except Exception as e:
        import traceback
        print(json.dumps({
            "level_id": level_id,
            "name": f"level_{level_id}",
            "status": f"WORKER_FAILED: {str(e)[:300]}",
            "traceback": traceback.format_exc()[-500:],
        }), flush=True)


def run_level_in_subprocess(level_id):
    """Run a level in a subprocess for clean memory isolation."""
    cmd = [
        sys.executable, __file__,
        "--worker", level_id,
        "--scale", str(SCALE),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        # Find JSON line in stdout
        for line in stdout.split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        return {
            "level_id": level_id,
            "name": f"level_{level_id}",
            "status": f"NO_JSON_OUTPUT",
            "stdout": stdout[-500:] if stdout else "",
            "stderr": stderr[-500:] if stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "level_id": level_id,
            "name": f"level_{level_id}",
            "status": "TIMEOUT (600s)",
        }
    except Exception as e:
        return {
            "level_id": level_id,
            "name": f"level_{level_id}",
            "status": f"SUBPROCESS_ERROR: {str(e)[:200]}",
        }


# ============================================================================
# Display
# ============================================================================

def format_results(results):
    """Format results as a comparison table."""
    print("\n" + "=" * 100)
    print(f"NNX vs Linen Memory Gap — Progressive Complexity (scale={SCALE})")
    print(f"Model: dim={DIM}, ff_dim={FF_DIM}, batch={BATCH}, seq={SEQ}")
    print(f"Pipeline: stages={NUM_STAGES}, repeats={NUM_REPEATS}, microbatches={NUM_MICROBATCHES}")
    print("=" * 100)

    # Header
    print(f"\n{'Level':<35s} {'Status':<8s} {'Total':>9s} {'Temp':>9s} {'Args':>9s} "
          f"{'Output':>9s} {'Delta':>12s} {'Compile':>8s}")
    print("-" * 100)

    baseline_temp = None
    prev_temp = {}

    for r in results:
        lid = r["level_id"]
        name = r.get("name", f"level_{lid}")
        status = r.get("status", "?")

        if status != "OK":
            print(f"  {name:<33s} {status}")
            continue

        total_gb = r.get("xla_total_gb", 0)
        temp_gb = r.get("xla_temp_gb", 0)
        arg_gb = r.get("xla_arg_gb", 0)
        out_gb = r.get("xla_out_gb", 0)
        compile_s = r.get("compile_s", -1)

        # Track baseline
        if baseline_temp is None:
            baseline_temp = temp_gb
            delta_str = "[BASELINE]"
        else:
            delta = temp_gb - baseline_temp
            sign = "+" if delta >= 0 else ""
            delta_str = f"[{sign}{delta:.4f} GB]"

        print(f"  {name:<33s} {'OK':<8s} {total_gb:>8.4f}G {temp_gb:>8.4f}G "
              f"{arg_gb:>8.4f}G {out_gb:>8.4f}G {delta_str:>12s} {compile_s:>7.1f}s")

    # NNX vs Linen comparison pairs
    print("\n" + "-" * 100)
    print("NNX vs Linen delta (Temp) at each complexity level:")
    print("-" * 100)

    pairs = [
        ("1", "2", "Simple scan"),
        ("3a", "3b", "+ vmap"),
        ("4a", "4b", "+ nested scan"),
        ("5a", "5b", "+ BSW dual-buf"),
        ("6a", "6b", "+ scan ys"),
        ("7a", "7b", "Full pipeline"),
    ]

    result_map = {r["level_id"]: r for r in results}

    for nnx_id, linen_id, desc in pairs:
        nnx_r = result_map.get(nnx_id)
        lin_r = result_map.get(linen_id)

        if not nnx_r or not lin_r:
            continue
        if nnx_r.get("status") != "OK" or lin_r.get("status") != "OK":
            print(f"  {desc:<25s}  NNX={nnx_r.get('status', '?')}, Linen={lin_r.get('status', '?')}")
            continue

        nnx_temp = nnx_r.get("xla_temp_gb", 0)
        lin_temp = lin_r.get("xla_temp_gb", 0)
        gap = nnx_temp - lin_temp
        sign = "+" if gap >= 0 else ""
        pct = (gap / lin_temp * 100) if lin_temp > 0 else 0

        marker = ""
        if abs(gap) > 0.5:
            marker = " <<<< GAP DETECTED"
        elif abs(gap) > 0.1:
            marker = " << notable"

        print(f"  {desc:<25s}  NNX={nnx_temp:.4f}G  Linen={lin_temp:.4f}G  "
              f"delta={sign}{gap:.4f}G ({sign}{pct:.1f}%){marker}")

    print("\n" + "=" * 100)
    print("Key: Total = Temp + Args + Output (all from XLA memory_analysis)")
    print("     Delta = Temp difference vs Level 0 baseline")
    print("     GAP DETECTED = NNX-Linen temp difference > 0.5 GB")
    print("=" * 100)


# ============================================================================
# Main
# ============================================================================

def main():
    # Parse level list from argv
    levels_to_run = []
    skip_next = False
    for i, arg in enumerate(sys.argv[1:], 1):
        if skip_next:
            skip_next = False
            continue
        if arg in ("--worker", "--scale"):
            skip_next = True
            continue
        if arg in LEVEL_BUILDERS:
            levels_to_run.append(arg)

    if not levels_to_run:
        levels_to_run = ALL_LEVELS

    print(f"Running {len(levels_to_run)} levels: {', '.join(levels_to_run)}")
    print(f"Scale={SCALE}, dim={DIM}, ff_dim={FF_DIM}, batch={BATCH}, seq={SEQ}")
    print(f"Stages={NUM_STAGES}, repeats={OUTER_N}, microbatches={INNER_N}")
    print()

    results = []
    for level_id in levels_to_run:
        name = LEVEL_BUILDERS[level_id].__doc__.split("\n")[0] if LEVEL_BUILDERS[level_id].__doc__ else level_id
        print(f"  Running level {level_id}: {name} ...", end="", flush=True)
        t0 = time.time()
        r = run_level_in_subprocess(level_id)
        elapsed = time.time() - t0
        status = r.get("status", "?")
        if status == "OK":
            print(f" OK ({elapsed:.1f}s, temp={r.get('xla_temp_gb', '?')}G)")
        else:
            print(f" {status} ({elapsed:.1f}s)")
            if "stderr" in r and r["stderr"]:
                # Print last few lines of stderr for debugging
                stderr_lines = r["stderr"].strip().split("\n")
                for line in stderr_lines[-3:]:
                    print(f"    stderr: {line}")
        results.append(r)

    format_results(results)

    # Save JSON for post-analysis
    out_path = os.path.join(os.path.dirname(__file__), "minimal_gap_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    # Check if running as worker subprocess
    if "--worker" in sys.argv:
        idx = sys.argv.index("--worker")
        if idx + 1 < len(sys.argv):
            worker_main(sys.argv[idx + 1])
            sys.exit(0)

    main()
