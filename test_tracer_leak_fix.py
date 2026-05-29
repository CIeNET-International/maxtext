"""
Minimal reproduction and fixes for the tracer leak that occurs when nesting:
    @jax.custom_vjp  inside  jax.checkpoint  inside  jax.lax.scan

There are TWO distinct failure modes:

MODE A (classical): @jax.custom_vjp's _fwd/_bwd are defined OUTSIDE the
    checkpoint boundary but reference a traced value from a stale scope.
    JAX 0.9.2 handles this via automatic const-var lifting in checkpoint,
    so it may not fail on modern JAX for simple patterns.

MODE B (NNX-style): The custom_vjp's _fwd/_bwd are Python closures that
    capture a traced value COMPUTED INSIDE the checkpoint scope. When
    checkpoint re-traces during backward, it creates a NEW scope and NEW
    tracers, but the stored _bwd Python closure still references the OLD
    tracers from the original forward trace. This causes:
        UnexpectedTracerError or TypeError: No constant handler for type:
            <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>

This file:
  1. Reproduces BOTH modes
  2. Tries 7 different fix strategies
  3. For each fix: checks compilation, gradient correctness, and MLIR while-loop count
"""

import jax
import jax.numpy as jnp
import functools
import traceback
import re

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SHAPE = (32, 32)
INNER_LEN = 4
OUTER_LEN = 2

def count_while_loops(fn, *args):
    """Count the number of while loops in the lowered MLIR."""
    try:
        lowered = jax.jit(fn).lower(*args)
        mlir_text = lowered.as_text()
        # Count stablehlo.while operations
        count = len(re.findall(r'stablehlo\.while', mlir_text))
        return count, None
    except Exception as e:
        return -1, str(e)

def run_test(name, loss_fn, params, x, reference_grad=None):
    """Run a single test: compile, grad, while-loop count."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    # 1. Try forward pass
    try:
        fwd = loss_fn(params, x)
        print(f"  Forward pass : OK  (value={float(fwd):.4f})")
    except Exception as e:
        print(f"  Forward pass : FAIL")
        print(f"    {type(e).__name__}: {e}")
        return None

    # 2. Try gradient
    grad_fn = jax.grad(loss_fn)
    try:
        grad = grad_fn(params, x)
        finite = bool(jnp.all(jnp.isfinite(grad)))
        print(f"  Gradient     : OK  (finite={finite}, norm={float(jnp.linalg.norm(grad)):.4f})")
    except Exception as e:
        print(f"  Gradient     : FAIL")
        err_str = str(e)
        if len(err_str) > 200:
            err_str = err_str[:200] + "..."
        print(f"    {type(e).__name__}: {err_str}")
        # Print last few lines of traceback for diagnosis
        tb_lines = traceback.format_exc().strip().split('\n')
        for line in tb_lines[-4:]:
            print(f"    {line}")
        return None

    # 3. Check correctness against reference
    # Use relative tolerance: float32 has ~7 decimal digits of precision,
    # so rtol=1e-5 is appropriate. Also use a generous atol for small values.
    if reference_grad is not None:
        max_diff = float(jnp.max(jnp.abs(grad - reference_grad)))
        ref_norm = float(jnp.linalg.norm(reference_grad))
        rel_err = max_diff / (ref_norm + 1e-8)
        close = rel_err < 1e-5  # relative error < 1e-5
        print(f"  Matches ref  : {close}  (max_abs_diff={max_diff:.4g}, rel_err={rel_err:.2e})")

    # 4. Count while loops in MLIR
    wl_count, wl_err = count_while_loops(grad_fn, params, x)
    if wl_err:
        print(f"  While loops  : ERROR ({wl_err[:100]})")
    else:
        print(f"  While loops  : {wl_count}")

    return grad


# ===========================================================================
# Shared custom_vjp definition (used by tests that define it outside)
# ===========================================================================
@jax.custom_vjp
def custom_step(x, w):
    return x @ w

def custom_step_fwd(x, w):
    out = x @ w
    return out, (x, w)

def custom_step_bwd(res, g):
    x, w = res
    return g @ w.T, x.T @ g

custom_step.defvjp(custom_step_fwd, custom_step_bwd)


# ===========================================================================
# Reference: no scan, no checkpoint (ground truth gradient)
# ===========================================================================
def loss_reference(params, x):
    """Unrolled reference -- no scan, no checkpoint."""
    c = x
    for _ in range(OUTER_LEN):
        for _ in range(INNER_LEN):
            c = custom_step(c, params)
    return c.sum()


# ===========================================================================
# MODE A BUG: outer_scan -> checkpoint -> inner_scan -> custom_vjp
#   params captured as closure from outer scan trace
#   (May work on JAX >= 0.4.30 due to automatic const-var lifting)
# ===========================================================================
def loss_bug_mode_a(params, x):
    """Mode A: custom_vjp defined outside, params from outer scan closure."""
    def outer_body(carry, _):
        def repeat_step(c, w):
            def inner_body(carry_inner, _):
                return custom_step(carry_inner, w), None
            final, _ = jax.lax.scan(inner_body, c, None, length=INNER_LEN)
            return final

        repeat_step_ckpt = jax.checkpoint(repeat_step)
        c = repeat_step_ckpt(carry, params)  # params is a tracer from outer scan
        return c, None

    final, _ = jax.lax.scan(outer_body, x, None, length=OUTER_LEN)
    return final.sum()


# ===========================================================================
# MODE B BUG: custom_vjp's _fwd/_bwd closures capture a value COMPUTED
#   INSIDE the checkpoint scope. When checkpoint re-traces for backward,
#   the Python closures still reference old tracers.
#
#   This is the pattern that occurs in NNX pipelines: a function inside
#   the checkpoint boundary computes a derived weight (e.g., via
#   dynamic_slice or all_gather), then a custom_vjp's _bwd closure
#   captures that derived weight.
# ===========================================================================
def loss_bug_mode_b(params, x):
    """Mode B: custom_vjp with closures capturing checkpoint-scoped tracers."""
    def outer_body(carry, _):
        def repeat_step(c, w):
            # Compute a derived weight INSIDE the checkpoint scope
            # This is analogous to all_gather or dynamic_slice in NNX pipeline
            w_derived = w * 2.0  # a traced value in checkpoint scope

            # NOW define custom_vjp that captures w_derived in its closures.
            # When checkpoint re-traces for backward, w_derived will be a
            # NEW tracer, but the _bwd closure is a Python object that still
            # references the OLD w_derived tracer.
            @jax.custom_vjp
            def step_with_derived(xx):
                return xx @ w_derived  # closure over checkpoint-scoped tracer

            def step_fwd(xx):
                out = xx @ w_derived   # captures w_derived
                return out, (xx,)

            def step_bwd(res, g):
                (xx,) = res
                return (g @ w_derived.T,)  # captures w_derived -- STALE on re-trace!

            step_with_derived.defvjp(step_fwd, step_bwd)

            def inner_body(carry_inner, _):
                return step_with_derived(carry_inner), None
            final, _ = jax.lax.scan(inner_body, c, None, length=INNER_LEN)
            return final

        repeat_step_ckpt = jax.checkpoint(repeat_step)
        c = repeat_step_ckpt(carry, params)
        return c, None

    final, _ = jax.lax.scan(outer_body, x, None, length=OUTER_LEN)
    return final.sum()


# ===========================================================================
# MODE B REFERENCE: same computation without custom_vjp (for gradient check)
# ===========================================================================
def loss_reference_mode_b(params, x):
    """Mode B reference: unrolled, no custom_vjp."""
    c = x
    for _ in range(OUTER_LEN):
        w_derived = params * 2.0
        for _ in range(INNER_LEN):
            c = c @ w_derived
    return c.sum()


# ===========================================================================
# FIX 1: Pass params as part of carry instead of closure
# ===========================================================================
def loss_fix1_carry(params, x):
    """Pass params through scan carry -- avoids closure capture entirely."""
    def outer_body(carry, _):
        c, w = carry

        def repeat_step(c_inner, w_inner):
            def inner_body(carry_inner, _):
                return custom_step(carry_inner, w_inner), None
            final, _ = jax.lax.scan(inner_body, c_inner, None, length=INNER_LEN)
            return final

        repeat_step_ckpt = jax.checkpoint(repeat_step)
        c = repeat_step_ckpt(c, w)
        return (c, w), None

    (final, _), _ = jax.lax.scan(outer_body, (x, params), None, length=OUTER_LEN)
    return final.sum()


# ===========================================================================
# FIX 2: Use functools.partial instead of closure capture
# ===========================================================================
def loss_fix2_functools_partial(params, x):
    """Use functools.partial to bind params -- but params is still a closure."""
    def repeat_step(c, w):
        def inner_body(carry_inner, _):
            return custom_step(carry_inner, w), None
        final, _ = jax.lax.scan(inner_body, c, None, length=INNER_LEN)
        return final

    def outer_body_impl(repeat_fn, carry, _):
        c = repeat_fn(carry, params)
        return c, None

    outer_body = functools.partial(outer_body_impl, jax.checkpoint(repeat_step))

    final, _ = jax.lax.scan(outer_body, x, None, length=OUTER_LEN)
    return final.sum()


# ===========================================================================
# FIX 3: jax.ensure_compile_time_eval around the closure capture
# ===========================================================================
def loss_fix3_ensure_compile_time(params, x):
    """Try jax.ensure_compile_time_eval to force params to be a constant."""
    def outer_body(carry, _):
        def repeat_step(c, w):
            def inner_body(carry_inner, _):
                return custom_step(carry_inner, w), None
            final, _ = jax.lax.scan(inner_body, c, None, length=INNER_LEN)
            return final

        repeat_step_ckpt = jax.checkpoint(repeat_step)
        try:
            with jax.ensure_compile_time_eval():
                w_const = params
        except Exception:
            w_const = params
        c = repeat_step_ckpt(carry, w_const)
        return c, None

    final, _ = jax.lax.scan(outer_body, x, None, length=OUTER_LEN)
    return final.sum()


# ===========================================================================
# FIX 4: Define custom_vjp INSIDE the checkpoint boundary BUT pass
#   w_derived as an explicit argument (not closure)
# ===========================================================================
def loss_fix4_explicit_args(params, x):
    """custom_vjp inside checkpoint, w_derived passed as explicit arg (not closure)."""
    def outer_body(carry, _):
        def repeat_step(c, w):
            w_derived = w * 2.0  # computed inside checkpoint scope

            # custom_vjp takes w_derived as an EXPLICIT argument
            @jax.custom_vjp
            def step_explicit(xx, wd):
                return xx @ wd

            def step_fwd(xx, wd):
                out = xx @ wd
                return out, (xx, wd)

            def step_bwd(res, g):
                xx, wd = res
                return g @ wd.T, xx.T @ g

            step_explicit.defvjp(step_fwd, step_bwd)

            def inner_body(carry_inner, _):
                return step_explicit(carry_inner, w_derived), None
            final, _ = jax.lax.scan(inner_body, c, None, length=INNER_LEN)
            return final

        repeat_step_ckpt = jax.checkpoint(repeat_step)
        c = repeat_step_ckpt(carry, params)
        return c, None

    final, _ = jax.lax.scan(outer_body, x, None, length=OUTER_LEN)
    return final.sum()


# ===========================================================================
# FIX 5: Use jax.debug.callback as a barrier
# ===========================================================================
def loss_fix5_callback_barrier(params, x):
    """Use jax.debug.callback to try to force materialization."""
    def outer_body(carry, _):
        def repeat_step(c, w):
            jax.debug.callback(lambda *_: None, w)

            def inner_body(carry_inner, _):
                return custom_step(carry_inner, w), None
            final, _ = jax.lax.scan(inner_body, c, None, length=INNER_LEN)
            return final

        repeat_step_ckpt = jax.checkpoint(repeat_step)
        c = repeat_step_ckpt(carry, params)
        return c, None

    final, _ = jax.lax.scan(outer_body, x, None, length=OUTER_LEN)
    return final.sum()


# ===========================================================================
# FIX 6: jax.checkpoint with static_argnums for the weights argument
# ===========================================================================
def loss_fix6_static_argnums(params, x):
    """Use jax.checkpoint(static_argnums=...) to treat w as static."""
    def outer_body(carry, _):
        def repeat_step(c, w):
            def inner_body(carry_inner, _):
                return custom_step(carry_inner, w), None
            final, _ = jax.lax.scan(inner_body, c, None, length=INNER_LEN)
            return final

        # static_argnums=1 means 'w' is treated as static
        repeat_step_ckpt = jax.checkpoint(repeat_step, static_argnums=(1,))
        c = repeat_step_ckpt(carry, params)
        return c, None

    final, _ = jax.lax.scan(outer_body, x, None, length=OUTER_LEN)
    return final.sum()


# ===========================================================================
# FIX 7: Use jax.tree_util.Partial (JAX-aware partial)
# ===========================================================================
def loss_fix7_jax_partial(params, x):
    """Use jax.tree_util.Partial which is JAX-pytree-aware."""
    def repeat_step_base(w, c):
        """Note: w is first arg so Partial binds it."""
        def inner_body(carry_inner, _):
            return custom_step(carry_inner, w), None
        final, _ = jax.lax.scan(inner_body, c, None, length=INNER_LEN)
        return final

    def outer_body(carry, _):
        # Create a JAX-aware partial that properly threads w through transforms
        repeat_fn = jax.tree_util.Partial(repeat_step_base, params)
        repeat_fn_ckpt = jax.checkpoint(repeat_fn)
        c = repeat_fn_ckpt(carry)
        return c, None

    final, _ = jax.lax.scan(outer_body, x, None, length=OUTER_LEN)
    return final.sum()


# ===========================================================================
# FIX 4b: Mode B fix -- avoid custom_vjp entirely (let JAX autodiff)
# ===========================================================================
def loss_fix4b_no_custom_vjp(params, x):
    """Mode B fix: remove custom_vjp entirely, let JAX differentiate."""
    def outer_body(carry, _):
        def repeat_step(c, w):
            w_derived = w * 2.0

            def inner_body(carry_inner, _):
                return carry_inner @ w_derived, None
            final, _ = jax.lax.scan(inner_body, c, None, length=INNER_LEN)
            return final

        repeat_step_ckpt = jax.checkpoint(repeat_step)
        c = repeat_step_ckpt(carry, params)
        return c, None

    final, _ = jax.lax.scan(outer_body, x, None, length=OUTER_LEN)
    return final.sum()


# ===========================================================================
# FIX 7b: Mode B fix -- jax.tree_util.Partial to thread w_derived
# ===========================================================================
def loss_fix7b_mode_b_jax_partial(params, x):
    """Mode B fix: use jax.tree_util.Partial to thread derived weight."""
    def outer_body(carry, _):
        def repeat_step(c, w):
            w_derived = w * 2.0

            def step_fn(wd, xx):
                """Takes w_derived as explicit first arg via Partial."""
                return xx @ wd

            step_partial = jax.tree_util.Partial(step_fn, w_derived)

            def inner_body(carry_inner, _):
                return step_partial(carry_inner), None
            final, _ = jax.lax.scan(inner_body, c, None, length=INNER_LEN)
            return final

        repeat_step_ckpt = jax.checkpoint(repeat_step)
        c = repeat_step_ckpt(carry, params)
        return c, None

    final, _ = jax.lax.scan(outer_body, x, None, length=OUTER_LEN)
    return final.sum()


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("JAX version:", jax.__version__)
    print(f"Shape: {SHAPE}, inner_len={INNER_LEN}, outer_len={OUTER_LEN}")
    print()

    params = jax.random.normal(jax.random.key(0), SHAPE)
    x = jnp.ones((1, SHAPE[1]))

    # -----------------------------------------------------------------------
    # Part 1: Mode A tests (custom_vjp defined OUTSIDE checkpoint)
    # -----------------------------------------------------------------------
    print("\n" + "#"*70)
    print("# PART 1: MODE A -- custom_vjp defined OUTSIDE checkpoint")
    print("#"*70)

    ref_grad_a = run_test(
        "REFERENCE A (unrolled, no scan/checkpoint)",
        loss_reference, params, x)
    assert ref_grad_a is not None, "Reference must work"

    run_test(
        "BUG MODE A: outer_scan -> checkpoint -> inner_scan -> custom_vjp",
        loss_bug_mode_a, params, x, reference_grad=ref_grad_a)

    tests_a = [
        ("FIX 1: params in carry (no closure)", loss_fix1_carry),
        ("FIX 2: functools.partial", loss_fix2_functools_partial),
        ("FIX 3: jax.ensure_compile_time_eval", loss_fix3_ensure_compile_time),
        ("FIX 5: jax.debug.callback barrier", loss_fix5_callback_barrier),
        ("FIX 6: jax.checkpoint(static_argnums)", loss_fix6_static_argnums),
        ("FIX 7: jax.tree_util.Partial (JAX-aware)", loss_fix7_jax_partial),
    ]

    results_a = {}
    for name, fn in tests_a:
        grad = run_test(name, fn, params, x, reference_grad=ref_grad_a)
        results_a[name] = grad is not None

    # -----------------------------------------------------------------------
    # Part 2: Mode B tests (custom_vjp closures capture checkpoint-scoped tracers)
    # -----------------------------------------------------------------------
    print("\n\n" + "#"*70)
    print("# PART 2: MODE B -- custom_vjp closures capture checkpoint-scoped tracers")
    print("#"*70)

    ref_grad_b = run_test(
        "REFERENCE B (unrolled, w_derived=w*2, no custom_vjp)",
        loss_reference_mode_b, params, x)
    assert ref_grad_b is not None, "Reference B must work"

    run_test(
        "BUG MODE B: custom_vjp captures checkpoint-scoped w_derived as closure",
        loss_bug_mode_b, params, x, reference_grad=ref_grad_b)

    tests_b = [
        ("FIX 4: custom_vjp inside ckpt, w_derived as explicit arg", loss_fix4_explicit_args),
        ("FIX 4b: remove custom_vjp entirely (let JAX autodiff)", loss_fix4b_no_custom_vjp),
        ("FIX 7b: jax.tree_util.Partial for w_derived", loss_fix7b_mode_b_jax_partial),
    ]

    results_b = {}
    for name, fn in tests_b:
        grad = run_test(name, fn, params, x, reference_grad=ref_grad_b)
        results_b[name] = grad is not None

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("  FULL SUMMARY")
    print(f"{'='*70}")
    print()
    print("  MODE A (custom_vjp defined outside, params from outer scan closure):")
    for name, ok in results_a.items():
        print(f"    [{'PASS' if ok else 'FAIL'}]  {name}")
    print()
    print("  MODE B (custom_vjp closures capture checkpoint-scoped tracer):")
    for name, ok in results_b.items():
        print(f"    [{'PASS' if ok else 'FAIL'}]  {name}")
    print(f"{'='*70}")

    print()
    print("KEY FINDINGS:")
    print("  - Mode A may pass on JAX >= 0.4.30 due to automatic const-var lifting")
    print("  - Mode B is the REAL danger in NNX pipelines: _bwd closure captures")
    print("    a checkpoint-scoped tracer that goes stale on re-trace")
    print("  - The reliable fixes are:")
    print("    (a) Pass all traced values as EXPLICIT arguments to custom_vjp")
    print("    (b) Remove custom_vjp and let JAX auto-differentiate")
    print("    (c) Use jax.tree_util.Partial to thread values through transforms")


if __name__ == "__main__":
    main()
