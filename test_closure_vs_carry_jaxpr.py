"""Compare closure vs carry memory in scan+checkpoint.

Three patterns:
  A) params as CLOSURE  — w captured from outer scope
  B) params as CARRY    — w threaded through carry tuple
  C) params as CARRY with stop_gradient — isolates carry-storage overhead from backward overhead

If C < B but C > A → some overhead from carry storage, some from allow_fwds
If C == A           → ALL overhead from carry backward storage; allow_fwds irrelevant
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from jax import random

DIM = 64
LENGTH = 4

# ============ Pattern A: params as CLOSURE ============
def loss_closure(w, x):
    def body(carry, _):
        return jnp.dot(carry, w), None  # w from closure
    body = jax.checkpoint(body)
    final, _ = jax.lax.scan(body, x, None, length=LENGTH)
    return jnp.sum(final)

# ============ Pattern B: params as CARRY ============
def loss_carry(w, x):
    def body(carry, _):
        x, w_carry = carry
        return (jnp.dot(x, w_carry), w_carry), None  # w from carry
    body = jax.checkpoint(body)
    (final, _), _ = jax.lax.scan(body, (x, w), None, length=LENGTH)
    return jnp.sum(final)

# ============ Pattern C: params as CARRY + stop_gradient ============
def loss_carry_stopped(w, x):
    def body(carry, _):
        x, w_carry = carry
        return (jnp.dot(x, jax.lax.stop_gradient(w_carry)), w_carry), None
    body = jax.checkpoint(body)
    (final, _), _ = jax.lax.scan(body, (x, w), None, length=LENGTH)
    return jnp.sum(final)

# --- helpers ---
def inspect_scan_params(jaxpr, label):
    """Print scan equation parameters for a jaxpr."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Top-level equations: {len(jaxpr.jaxpr.eqns)}")
    print(f"  Top-level constvars: {len(jaxpr.jaxpr.constvars)}")
    print(f"  Top-level invars:    {len(jaxpr.jaxpr.invars)}")
    scan_count = 0
    for i, eqn in enumerate(jaxpr.jaxpr.eqns):
        if eqn.primitive.name == 'scan':
            scan_count += 1
            params = eqn.params
            body_jaxpr = params.get('jaxpr', None)
            nc = params.get('num_carry', '?')
            nconst = params.get('num_consts', '?')
            allow = params.get('_allow_fwds', 'NOT PRESENT')
            reverse = params.get('reverse', '?')
            print(f"  Scan #{scan_count} (eqn {i}):")
            print(f"    num_carry   = {nc}")
            print(f"    num_consts  = {nconst}")
            print(f"    reverse     = {reverse}")
            print(f"    _allow_fwds = {allow}")
            print(f"    in vars     = {len(eqn.invars)}")
            print(f"    out vars    = {len(eqn.outvars)}")
            if body_jaxpr:
                raw = body_jaxpr.jaxpr if hasattr(body_jaxpr, 'jaxpr') else body_jaxpr
                print(f"    body constvars = {len(raw.constvars)}")
                print(f"    body invars    = {len(raw.invars)}")
                print(f"    body eqns      = {len(raw.eqns)}")
                for j, sub_eqn in enumerate(raw.eqns):
                    if 'remat' in sub_eqn.primitive.name:
                        sub_j = sub_eqn.params.get('jaxpr', None)
                        if sub_j:
                            sr = sub_j.jaxpr if hasattr(sub_j, 'jaxpr') else sub_j
                            print(f"    Remat sub-jaxpr:")
                            print(f"      constvars = {len(sr.constvars)}")
                            print(f"      invars    = {len(sr.invars)}")
                            print(f"      eqns      = {len(sr.eqns)}")
            # Dump all scan params for transparency
            print(f"    All scan params keys: {sorted(params.keys())}")
    if scan_count == 0:
        print("  (no scan equations found)")


def memory_for(fn, w, x, label):
    """Compile and return memory analysis."""
    compiled = jax.jit(jax.grad(fn)).lower(w, x).compile()
    mem = compiled.memory_analysis()
    temp = getattr(mem, 'temp_size_in_bytes', 0)
    args = getattr(mem, 'argument_size_in_bytes', 0)
    out  = getattr(mem, 'output_size_in_bytes', 0)
    alias = getattr(mem, 'alias_size_in_bytes', 0)
    total = temp + args + out
    print(f"  {label:22s}: Temp={temp:>8,}  Args={args:>8,}  Out={out:>8,}  Alias={alias:>8,}  Total={total:>8,}")
    return temp, args, out


# ============ Main ============
if __name__ == "__main__":
    w = random.normal(random.PRNGKey(0), (DIM, DIM)) * 0.01
    x = random.normal(random.PRNGKey(1), (8, DIM))

    # ---- Jaxpr inspection ----
    patterns = [
        ("A: CLOSURE",              loss_closure),
        ("B: CARRY",                loss_carry),
        ("C: CARRY + stop_gradient", loss_carry_stopped),
    ]

    jaxprs = {}
    for label, fn in patterns:
        jp = jax.make_jaxpr(jax.grad(fn))(w, x)
        jaxprs[label] = jp
        inspect_scan_params(jp, label)

    # ---- Compiled memory analysis ----
    print(f"\n{'=' * 60}")
    print("  COMPILED MEMORY ANALYSIS")
    print(f"{'=' * 60}")
    results = {}
    for label, fn in patterns:
        t, a, o = memory_for(fn, w, x, label)
        results[label] = (t, a, o)

    # ---- Diagnosis ----
    print(f"\n{'=' * 60}")
    print("  DIAGNOSIS")
    print(f"{'=' * 60}")
    tA = results["A: CLOSURE"][0]
    tB = results["B: CARRY"][0]
    tC = results["C: CARRY + stop_gradient"][0]

    if tA > 0:
        ratio_BA = tB / tA
    else:
        ratio_BA = float('inf')

    print(f"  Temp A (closure):         {tA:>10,} bytes")
    print(f"  Temp B (carry):           {tB:>10,} bytes")
    print(f"  Temp C (carry+stopgrad):  {tC:>10,} bytes")
    print(f"  B/A ratio:                {ratio_BA:.2f}x")
    print()

    if tC == tA:
        print("  CONCLUSION: C == A.")
        print("  ALL overhead in B comes from backward gradient storage of w")
        print("  through the carry (scan must store intermediate w for grad).")
        print("  allow_fwds is NOT the issue.")
    elif tC > tA and tC < tB:
        overhead_carry = tC - tA
        overhead_bwd   = tB - tC
        print(f"  CONCLUSION: A < C < B.")
        print(f"  Carry storage overhead (C-A):  {overhead_carry:>10,} bytes")
        print(f"  Backward/allow_fwds (B-C):     {overhead_bwd:>10,} bytes")
        print(f"  Some overhead from carry storage, some from allow_fwds blocking.")
    elif tC >= tB:
        print(f"  CONCLUSION: C >= B (unexpected). stop_gradient did not reduce memory.")
        print(f"  The overhead is from storing w in the forward carry, not backward.")
    else:
        print(f"  CONCLUSION: C < A (unexpected). Needs further investigation.")

    # ---- Quick jaxpr text size comparison ----
    print(f"\n{'=' * 60}")
    print("  JAXPR TEXT SIZES")
    print(f"{'=' * 60}")
    for label, jp in jaxprs.items():
        text = str(jp)
        n_scans = text.count("scan")
        print(f"  {label:30s}: {len(text):>6} chars, {n_scans} scan refs")

    print("\nDone.")
