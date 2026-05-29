"""
Minimal pure-JAX script to explore whether any pattern can make
jax.remat inside jax.lax.scan produce SEPARATE recomputation while loops
(like Linen's nn.scan(nn.remat(body)) which produces 8 HLO while loops
vs raw jax.lax.scan(jax.checkpoint(body)) which produces 4).

Goal: find ANY pure-JAX pattern that produces more than 4 while loops.
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import re

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DIM = 32
SCAN_LEN = 4
KEY = jax.random.PRNGKey(0)

params_global = jax.random.normal(KEY, (DIM, DIM))
init_global = jax.random.normal(jax.random.PRNGKey(1), (DIM,))
xs_global = jax.random.normal(jax.random.PRNGKey(2), (SCAN_LEN, DIM))


def count_while_loops(fn, *args):
    """Lower fn via jax.jit(jax.grad(fn)) and count 'while' loops in MLIR."""
    grad_fn = jax.jit(jax.grad(fn))
    lowered = grad_fn.lower(*args)
    mlir_text = lowered.as_text()
    # Count occurrences of scf.while or stablehlo.while
    count = len(re.findall(r'stablehlo\.while|scf\.while', mlir_text))
    return count, mlir_text


# ===================================================================
# Approach 1: Standard baseline -- jax.checkpoint inside scan body
# ===================================================================
def approach1(params, init, xs):
    def scan_body(carry, x):
        @jax.checkpoint
        def step(c):
            c = jnp.dot(params, c) + x
            c = jax.nn.relu(c)
            c = jnp.dot(params, c)
            return c
        return step(carry), None
    final, _ = jax.lax.scan(scan_body, init, xs)
    return jnp.sum(final)


# ===================================================================
# Approach 2: custom_vjp wrapping the scan body
# ===================================================================
def approach2(params, init, xs):
    @jax.custom_vjp
    def custom_step(params, carry, x):
        c = jnp.dot(params, carry) + x
        c = jax.nn.relu(c)
        c = jnp.dot(params, c)
        return c

    def custom_step_fwd(params, carry, x):
        out = custom_step(params, carry, x)
        return out, (params, carry, x)

    def custom_step_bwd(res, g):
        params, carry, x = res
        # Recompute forward inside bwd
        def fwd(p, c, xi):
            c2 = jnp.dot(p, c) + xi
            c2 = jax.nn.relu(c2)
            c2 = jnp.dot(p, c2)
            return c2
        _, vjp_fn = jax.vjp(fwd, params, carry, x)
        return vjp_fn(g)

    custom_step.defvjp(custom_step_fwd, custom_step_bwd)

    def scan_body(carry, x):
        return custom_step(params, carry, x), None

    final, _ = jax.lax.scan(scan_body, init, xs)
    return jnp.sum(final)


# ===================================================================
# Approach 3: custom_vjp wrapping the SCAN itself (like Linen's pattern)
# ===================================================================
def approach3(params, init, xs):
    @jax.custom_vjp
    def scanned_fn(params, init, xs):
        def scan_body(carry, x):
            c = jnp.dot(params, carry) + x
            c = jax.nn.relu(c)
            c = jnp.dot(params, c)
            return c, None
        final, _ = jax.lax.scan(scan_body, init, xs)
        return final

    def scanned_fn_fwd(params, init, xs):
        # Run forward scan, save intermediates
        def fwd_body(carry, x):
            c = jnp.dot(params, carry) + x
            c = jax.nn.relu(c)
            c = jnp.dot(params, c)
            return c, carry  # save input carry
        final, saved_carries = jax.lax.scan(fwd_body, init, xs)
        return final, (params, saved_carries, xs)

    def scanned_fn_bwd(res, g):
        params, saved_carries, xs = res
        # Backward scan with recomputation
        def bwd_body(carry_grad, packed):
            saved_c, x = packed
            def fwd(p, c, xi):
                c2 = jnp.dot(p, c) + xi
                c2 = jax.nn.relu(c2)
                c2 = jnp.dot(p, c2)
                return c2
            _, vjp_fn = jax.vjp(fwd, params, saved_c, x)
            g_params, g_carry, g_x = vjp_fn(carry_grad)
            return g_carry, (g_params, g_x)

        g_init, (g_params_all, g_xs) = jax.lax.scan(
            bwd_body, g, (saved_carries, xs), reverse=True
        )
        g_params_sum = jax.tree.map(lambda x: jnp.sum(x, axis=0), g_params_all)
        return g_params_sum, g_init, g_xs

    scanned_fn.defvjp(scanned_fn_fwd, scanned_fn_bwd)

    return jnp.sum(scanned_fn(params, init, xs))


# ===================================================================
# Approach 4: jax.remat with prevent_cse=True + different policies
# ===================================================================
def approach4(params, init, xs):
    def scan_body(carry, x):
        @partial(jax.remat, prevent_cse=True)
        def step(c):
            c = jnp.dot(params, c) + x
            c = jax.nn.relu(c)
            c = jnp.dot(params, c)
            return c
        return step(carry), None
    final, _ = jax.lax.scan(scan_body, init, xs)
    return jnp.sum(final)


def approach4b(params, init, xs):
    """remat with policy=everything_saveable (save nothing)."""
    def scan_body(carry, x):
        @partial(jax.remat, prevent_cse=True,
                 policy=lambda *_, **__: False)
        def step(c):
            c = jnp.dot(params, c) + x
            c = jax.nn.relu(c)
            c = jnp.dot(params, c)
            return c
        return step(carry), None
    final, _ = jax.lax.scan(scan_body, init, xs)
    return jnp.sum(final)


# ===================================================================
# Approach 5: Using jax.linear_util wrap + partial_eval hooks
# ===================================================================
def approach5(params, init, xs):
    """
    Try to use jax.checkpoint with a concrete=True-like pattern:
    wrap the step in remat then manually call it as a closed-over
    function inside the scan body.
    """
    @partial(jax.remat, prevent_cse=True)
    def remat_layer(params, c, x):
        c = jnp.dot(params, c) + x
        c = jax.nn.relu(c)
        c = jnp.dot(params, c)
        return c

    def scan_body(carry, x):
        return remat_layer(params, carry, x), None

    final, _ = jax.lax.scan(scan_body, init, xs)
    return jnp.sum(final)


# ===================================================================
# Approach 6: Using jax.custom_jvp instead of custom_vjp
# ===================================================================
def approach6(params, init, xs):
    @jax.custom_jvp
    def step(params, carry, x):
        c = jnp.dot(params, carry) + x
        c = jax.nn.relu(c)
        c = jnp.dot(params, c)
        return c

    @step.defjvp
    def step_jvp(primals, tangents):
        params, carry, x = primals
        dp, dc, dx = tangents
        out = step(params, carry, x)
        # Manual JVP via recomputation
        def f(p, c, xi):
            c2 = jnp.dot(p, c) + xi
            c2 = jax.nn.relu(c2)
            c2 = jnp.dot(p, c2)
            return c2
        _, tangent_out = jax.jvp(f, primals, tangents)
        return out, tangent_out

    def scan_body(carry, x):
        return step(params, carry, x), None

    final, _ = jax.lax.scan(scan_body, init, xs)
    return jnp.sum(final)


# ===================================================================
# Approach 7: Nesting multiple scans with remat between them
# ===================================================================
def approach7(params, init, xs):
    """Two sequential scans, each with remat, to see if nesting changes
    the HLO structure."""
    @jax.checkpoint
    def step(params, c, x):
        c = jnp.dot(params, c) + x
        c = jax.nn.relu(c)
        return c

    def scan1_body(carry, x):
        return step(params, carry, x), None

    mid, _ = jax.lax.scan(scan1_body, init, xs[:SCAN_LEN // 2])

    @jax.checkpoint
    def step2(params, c, x):
        c = jnp.dot(params, c)
        return c

    def scan2_body(carry, x):
        return step2(params, carry, x), None

    final, _ = jax.lax.scan(scan2_body, mid, xs[SCAN_LEN // 2:])
    return jnp.sum(final)


# ===================================================================
# Approach 8: jax.checkpoint with static_argnums
# ===================================================================
def approach8(params, init, xs):
    def scan_body(carry, x):
        @partial(jax.checkpoint, static_argnums=(0,))
        def step(use_relu, c):
            c = jnp.dot(params, c) + x
            if use_relu:
                c = jax.nn.relu(c)
            c = jnp.dot(params, c)
            return c
        return step(True, carry), None
    final, _ = jax.lax.scan(scan_body, init, xs)
    return jnp.sum(final)


# ===================================================================
# Approach 9: jax.remat explicitly (not jax.checkpoint alias check)
# ===================================================================
def approach9(params, init, xs):
    """jax.remat is the same as jax.checkpoint in modern JAX,
    but let's verify and try with all options."""
    def scan_body(carry, x):
        @partial(jax.remat, prevent_cse=True,
                 policy=jax.checkpoint_policies.nothing_saveable)
        def step(c):
            c = jnp.dot(params, c) + x
            c = jax.nn.relu(c)
            c = jnp.dot(params, c)
            return c
        return step(carry), None
    final, _ = jax.lax.scan(scan_body, init, xs)
    return jnp.sum(final)


# ===================================================================
# Approach 10: jax.remat OUTSIDE the scan, wrapping the whole thing
# ===================================================================
def approach10(params, init, xs):
    @partial(jax.remat, prevent_cse=True)
    def scanned_computation(params, init, xs):
        def scan_body(carry, x):
            c = jnp.dot(params, carry) + x
            c = jax.nn.relu(c)
            c = jnp.dot(params, c)
            return c, None
        final, _ = jax.lax.scan(scan_body, init, xs)
        return final

    return jnp.sum(scanned_computation(params, init, xs))


# ===================================================================
# Approach 11 (bonus): custom_vjp on scan body WITH remat inside bwd
# This mimics Linen's actual pattern more closely: the backward
# pass itself uses a scan, and each step recomputes.
# ===================================================================
def approach11(params, init, xs):
    @jax.custom_vjp
    def scanned_fn(params, init, xs):
        def body(carry, x):
            c = jnp.dot(params, carry) + x
            c = jax.nn.relu(c)
            c = jnp.dot(params, c)
            return c, None
        final, _ = jax.lax.scan(body, init, xs)
        return final

    def scanned_fn_fwd(params, init, xs):
        # Forward: just run, save minimal residuals
        final = scanned_fn(params, init, xs)
        # Only save init -- force recomputation in backward
        return final, (params, init, xs)

    def scanned_fn_bwd(res, g):
        params, init, xs = res

        # Recompute all intermediates in a forward scan
        def fwd_body(carry, x):
            c_in = carry
            c = jnp.dot(params, c_in) + x
            c = jax.nn.relu(c)
            c = jnp.dot(params, c)
            return c, c_in  # save input for backward
        _, all_inputs = jax.lax.scan(fwd_body, init, xs)

        # Backward scan
        def bwd_body(g_carry, packed):
            c_in, x = packed
            def fwd_step(p, c, xi):
                c2 = jnp.dot(p, c) + xi
                c2 = jax.nn.relu(c2)
                c2 = jnp.dot(p, c2)
                return c2
            _, vjp_fn = jax.vjp(fwd_step, params, c_in, x)
            gp, gc, gx = vjp_fn(g_carry)
            return gc, (gp, gx)

        g_init, (g_params_all, g_xs) = jax.lax.scan(
            bwd_body, g, (all_inputs, xs), reverse=True
        )
        g_params = jax.tree.map(lambda x: jnp.sum(x, axis=0), g_params_all)
        return g_params, g_init, g_xs

    scanned_fn.defvjp(scanned_fn_fwd, scanned_fn_bwd)
    return jnp.sum(scanned_fn(params, init, xs))


# ===================================================================
# Approach 12 (bonus): Nested remat + scan: remat wraps scan,
# AND remat inside scan body
# ===================================================================
def approach12(params, init, xs):
    @jax.checkpoint
    def outer_remat(params, init, xs):
        def scan_body(carry, x):
            @jax.checkpoint
            def step(c):
                c = jnp.dot(params, c) + x
                c = jax.nn.relu(c)
                c = jnp.dot(params, c)
                return c
            return step(carry), None
        final, _ = jax.lax.scan(scan_body, init, xs)
        return final

    return jnp.sum(outer_remat(params, init, xs))


# ===================================================================
# Run all approaches
# ===================================================================
def main():
    print("=" * 70)
    print("Testing pure-JAX patterns for separate remat while loops")
    print(f"Config: DIM={DIM}, SCAN_LEN={SCAN_LEN}")
    print(f"JAX version: {jax.__version__}")
    print(f"jax.checkpoint is jax.remat: {jax.checkpoint is jax.remat}")
    print("=" * 70)

    approaches = [
        ("Approach 1:  baseline (jax.checkpoint inside scan)", approach1),
        ("Approach 2:  custom_vjp wrapping scan body", approach2),
        ("Approach 3:  custom_vjp wrapping the SCAN itself", approach3),
        ("Approach 4a: jax.remat prevent_cse=True", approach4),
        ("Approach 4b: jax.remat prevent_cse + save-nothing policy", approach4b),
        ("Approach 5:  remat as closed-over fn in scan", approach5),
        ("Approach 6:  custom_jvp instead of custom_vjp", approach6),
        ("Approach 7:  two sequential scans with remat", approach7),
        ("Approach 8:  jax.checkpoint with static_argnums", approach8),
        ("Approach 9:  jax.remat with nothing_saveable policy", approach9),
        ("Approach 10: jax.remat OUTSIDE scan (wrapping whole fn)", approach10),
        ("Approach 11: custom_vjp on scan + recompute fwd in bwd", approach11),
        ("Approach 12: nested remat (outer + inner) with scan", approach12),
    ]

    results = []
    for name, fn in approaches:
        try:
            n_loops, mlir = count_while_loops(fn, params_global, init_global, xs_global)
            results.append((name, n_loops, None))
            print(f"{name}: {n_loops} while loops")
        except Exception as e:
            results.append((name, -1, str(e)))
            print(f"{name}: ERROR - {e}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    baseline = None
    for name, n, err in results:
        if "baseline" in name and err is None:
            baseline = n
            break

    for name, n, err in results:
        if err:
            status = f"ERROR: {err[:60]}"
        elif baseline is not None and n > baseline:
            status = f"{n} loops  *** MORE THAN BASELINE ({baseline}) ***"
        elif baseline is not None and n == baseline:
            status = f"{n} loops  (same as baseline)"
        else:
            status = f"{n} loops"
        print(f"  {name}: {status}")

    # Dump MLIR for any approach with more loops than baseline
    print("\n" + "=" * 70)
    print("DETAILED MLIR FOR APPROACHES WITH MORE LOOPS THAN BASELINE")
    print("=" * 70)
    found_interesting = False
    for name, fn in approaches:
        try:
            n_loops, mlir = count_while_loops(fn, params_global, init_global, xs_global)
            if baseline is not None and n_loops > baseline:
                found_interesting = True
                print(f"\n--- {name} ({n_loops} loops) ---")
                # Print just the while loop signatures
                for i, match in enumerate(re.finditer(
                    r'(stablehlo\.while|scf\.while).*', mlir
                )):
                    print(f"  loop {i}: {match.group()[:120]}")
        except Exception:
            pass

    if not found_interesting:
        print("  No approach produced more while loops than baseline.")
        print("  This confirms that Linen's nn.scan does something")
        print("  beyond simple jax.remat/jax.checkpoint wrapping.")


if __name__ == "__main__":
    main()
