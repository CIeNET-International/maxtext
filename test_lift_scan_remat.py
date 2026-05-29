"""Test: Using Flax lift.scan + lift.checkpoint with NNX state via minimal Scope.

Goal: Produce 8 HLO while loops (separate remat recomputation per scan iteration)
instead of 4 (which you get without remat).

Approach:
  1. Create NNX parameters (simple linear layer weights)
  2. Package them into a Linen-style variable dict: {"params": {"kernel": ..., "bias": ...}}
  3. Use flax.core.bind() to create a Scope -- NO Linen Module needed
  4. Call lift.scan(lift.checkpoint(body_fn)) on that Scope
  5. Inspect the JAXPR for while_loop count

This is a standalone script. Run: python test_lift_scan_remat.py
"""

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import jax
import jax.numpy as jnp
from jax import random
import re
import functools

# Import Flax core (Scope-level API, no nn.Module needed)
from flax.core import bind, Scope
from flax.core import lift
from flax.core.frozen_dict import freeze, unfreeze


# ============================================================================
# Step 1: Create NNX-style parameters (raw pytrees)
# ============================================================================
def make_nnx_params(key, num_layers, d_model):
    """Create stacked linear layer params as raw arrays (NNX-style)."""
    params = {}
    k1, k2 = random.split(key)
    # Shape: [num_layers, d_model, d_model] -- stacked for scan
    params["kernel"] = random.normal(k1, (num_layers, d_model, d_model)) * 0.02
    params["bias"] = jnp.zeros((num_layers, d_model))
    return params


# ============================================================================
# Step 2: Define the body function operating on a Scope
# ============================================================================
def linear_body(scope: Scope, carry, _):
    """One linear layer step. scope has params, carry is the activation."""
    kernel = scope.get_variable("params", "kernel")
    bias = scope.get_variable("params", "bias")
    carry = carry @ kernel + bias
    carry = jax.nn.relu(carry)
    return carry, None


# ============================================================================
# Step 3: Three strategies to compare
# ============================================================================

def strategy_plain_scan(variables, x, num_layers):
    """Strategy A: lift.scan only (no remat). Expect ~1 while loop."""

    def body(scope, carry):
        kernel = scope.get_variable("params", "kernel")
        bias = scope.get_variable("params", "bias")
        carry = carry @ kernel + bias
        carry = jax.nn.relu(carry)
        return carry, ()

    fn = lift.scan(
        body,
        variable_axes={"params": 0},
        variable_broadcast=False,
        variable_carry=False,
        split_rngs={"params": False},
        length=num_layers,
    )

    from flax.core import apply as core_apply
    apply_fn = core_apply(lambda scope, x: fn(scope, x)[0])
    return apply_fn(variables, x)


def strategy_scan_remat(variables, x, num_layers):
    """Strategy B: lift.scan wrapping lift.checkpoint (remat). Expect more while loops."""

    def body(scope, carry):
        kernel = scope.get_variable("params", "kernel")
        bias = scope.get_variable("params", "bias")
        carry = carry @ kernel + bias
        carry = jax.nn.relu(carry)
        return carry, ()

    rematted_body = lift.checkpoint(
        body,
        variables=True,
        rngs=True,
        prevent_cse=False,  # inside scan, CSE prevention unnecessary
    )

    fn = lift.scan(
        rematted_body,
        variable_axes={"params": 0},
        variable_broadcast=False,
        variable_carry=False,
        split_rngs={"params": False},
        length=num_layers,
    )

    from flax.core import apply as core_apply
    apply_fn = core_apply(lambda scope, x: fn(scope, x)[0])
    return apply_fn(variables, x)


def strategy_remat_scan(variables_flat, x, num_layers):
    """Strategy C: lift.remat_scan with lengths=(outer, inner).
    This is the canonical Linen approach for O(sqrt(N)) memory.
    With lengths=(2, num_layers//2), expect 2 scan loops + remat.

    remat_scan requires params shaped [outer, inner, ...] so the nested
    scans can each slice along axis 0.
    """
    outer = 2
    inner = num_layers // outer

    # Reshape params from [num_layers, ...] to [outer, inner, ...]
    reshaped_params = jax.tree_util.tree_map(
        lambda p: p.reshape((outer, inner) + p.shape[1:]),
        variables_flat["params"],
    )
    variables = {"params": reshaped_params}

    def body(scope, carry):
        kernel = scope.get_variable("params", "kernel")
        bias = scope.get_variable("params", "bias")
        carry = carry @ kernel + bias
        carry = jax.nn.relu(carry)
        return carry

    fn = lift.remat_scan(
        body,
        lengths=(outer, inner),
        variable_axes={"params": 0},
        variable_broadcast=False,
        variable_carry=False,
        split_rngs={"params": False},
    )

    from flax.core import apply as core_apply
    apply_fn = core_apply(lambda scope, x: fn(scope, x))
    return apply_fn(variables, x)


def strategy_remat_scan_deep(variables_flat, x, num_layers):
    """Strategy D: lift.remat_scan with lengths=(2,2,2) -- 3-level nesting.
    This should produce even more while loops.
    """
    # 2*2*2 = 8 layers
    levels = (2, 2, 2)

    # Reshape params from [8, ...] to [2, 2, 2, ...]
    reshaped_params = jax.tree_util.tree_map(
        lambda p: p.reshape(levels + p.shape[1:]),
        variables_flat["params"],
    )
    variables = {"params": reshaped_params}

    def body(scope, carry):
        kernel = scope.get_variable("params", "kernel")
        bias = scope.get_variable("params", "bias")
        carry = carry @ kernel + bias
        carry = jax.nn.relu(carry)
        return carry

    fn = lift.remat_scan(
        body,
        lengths=levels,
        variable_axes={"params": 0},
        variable_broadcast=False,
        variable_carry=False,
        split_rngs={"params": False},
    )

    from flax.core import apply as core_apply
    apply_fn = core_apply(lambda scope, x: fn(scope, x))
    return apply_fn(variables, x)


def strategy_raw_jax(variables, x, num_layers):
    """Strategy E: raw jax.lax.scan + jax.checkpoint (no Flax lift).
    Direct JAX approach, no Scope wrapper at all.
    """
    params = variables["params"]

    @jax.checkpoint
    def body_fn(carry, layer_params):
        kernel, bias = layer_params["kernel"], layer_params["bias"]
        carry = carry @ kernel + bias
        carry = jax.nn.relu(carry)
        return carry, None

    # params is {kernel: [N,...], bias: [N,...]} -> scan over leading axis
    carry, _ = jax.lax.scan(body_fn, x, params)
    return carry


# ============================================================================
# Step 4: Count while loops in JAXPR/HLO
# ============================================================================
def count_while_loops_in_jaxpr(jaxpr_text):
    """Count while_loop occurrences in a JAXPR text representation."""
    return len(re.findall(r'while\[', jaxpr_text))


def count_while_loops_in_hlo(hlo_text):
    """Count 'while(' occurrences in HLO text."""
    return len(re.findall(r'\bwhile\b', hlo_text))


def count_scan_in_jaxpr(jaxpr_text):
    """Count scan occurrences in JAXPR."""
    return len(re.findall(r'scan\[', jaxpr_text))


# ============================================================================
# Step 5: Run and analyze
# ============================================================================
def main():
    NUM_LAYERS = 8
    D_MODEL = 16
    BATCH = 2

    key = random.PRNGKey(42)
    params = make_nnx_params(key, NUM_LAYERS, D_MODEL)
    variables = {"params": params}
    x = jnp.ones((BATCH, D_MODEL))

    print("=" * 70)
    print("TEST: lift.scan + lift.checkpoint with NNX state via Scope")
    print("=" * 70)
    print(f"Config: {NUM_LAYERS} layers, d_model={D_MODEL}, batch={BATCH}")
    print()

    strategies = {
        "A: plain scan (no remat)": strategy_plain_scan,
        "B: scan + checkpoint (remat)": strategy_scan_remat,
        "C: remat_scan(2,4)": strategy_remat_scan,
        "D: remat_scan(2,2,2)": strategy_remat_scan_deep,
        "E: raw jax.lax.scan + jax.checkpoint": strategy_raw_jax,
    }

    for name, strategy_fn in strategies.items():
        print(f"--- {name} ---")

        # Make a JAX function we can trace
        @jax.jit
        def f(x):
            return strategy_fn(variables, x, NUM_LAYERS)

        # Get JAXPR
        jaxpr = jax.make_jaxpr(f)(x)
        jaxpr_str = str(jaxpr)

        scan_count = count_scan_in_jaxpr(jaxpr_str)
        while_count = count_while_loops_in_jaxpr(jaxpr_str)
        remat_count = len(re.findall(r'remat2\[', jaxpr_str))
        checkpoint_count = len(re.findall(r'optimization_barrier', jaxpr_str))

        print(f"  JAXPR scan count:       {scan_count}")
        print(f"  JAXPR while count:      {while_count}")
        print(f"  JAXPR remat2 count:     {remat_count}")
        print(f"  JAXPR opt_barrier count: {checkpoint_count}")

        # Compile and get HLO
        lowered = f.lower(x)
        hlo_text = lowered.as_text()
        hlo_while_count = count_while_loops_in_hlo(hlo_text)
        print(f"  HLO while count:        {hlo_while_count}")

        # Actually run it to verify correctness
        result = f(x)
        print(f"  Output shape:           {result.shape}")
        print(f"  Output mean:            {float(jnp.mean(result)):.6f}")
        print()

    # ------------------------------------------------------------------
    # Bonus: try the gradient path (this is where while-loop count matters)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("GRADIENT (backward pass) analysis -- this is what matters for memory")
    print("=" * 70)
    print()

    for name, strategy_fn in strategies.items():
        print(f"--- {name} (grad) ---")

        @jax.jit
        def grad_f(x, variables=variables):
            def loss(x):
                y = strategy_fn(variables, x, NUM_LAYERS)
                return jnp.sum(y)
            return jax.grad(loss)(x)

        jaxpr = jax.make_jaxpr(grad_f)(x)
        jaxpr_str = str(jaxpr)

        scan_count = count_scan_in_jaxpr(jaxpr_str)
        while_count = count_while_loops_in_jaxpr(jaxpr_str)
        remat_count = len(re.findall(r'remat2\[', jaxpr_str))

        print(f"  JAXPR scan count:       {scan_count}")
        print(f"  JAXPR while count:      {while_count}")
        print(f"  JAXPR remat2 count:     {remat_count}")

        lowered = grad_f.lower(x)
        hlo_text = lowered.as_text()
        hlo_while_count = count_while_loops_in_hlo(hlo_text)
        print(f"  HLO while count:        {hlo_while_count}")

        result = grad_f(x)
        print(f"  Grad shape:             {result.shape}")
        print(f"  Grad mean:              {float(jnp.mean(result)):.6f}")
        print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
MEASURED RESULTS (HLO while-loop count):

Strategy               Fwd   Grad(fwd+bwd)
A: plain scan           1      2
B: scan+checkpoint      1      2  (remat inlined into scan body, NOT separate loop)
C: remat_scan(2,4)      2      5  (nested: 2 outer + inner remat creates extra loops)
D: remat_scan(2,2,2)    3      9  (3-level nesting: most while loops)
E: raw jax scan+ckpt    1      2  (same as B, jax.checkpoint inlines)

KEY FINDINGS:
1. lift.scan + lift.checkpoint (Strategy B) does NOT produce separate remat
   while loops. The remat is inlined into the scan body -- XLA sees ONE while
   loop per scan, with recomputation happening inside each iteration.
   Same as raw jax.lax.scan + jax.checkpoint (E). Both give 2 grad while loops.

2. lift.remat_scan (C, D) DOES produce extra while loops via NESTING, not
   via separate remat loops. Each level of nesting adds scans:
   - (2,4): 2 fwd, 5 grad  (outer + inner + remat-replayed inner loops)
   - (2,2,2): 3 fwd, 9 grad

3. To get 8 HLO while loops in grad from 4 pipeline micro-steps:
   - remat_scan(2,2) on 4 microsteps -> will produce ~5 grad while loops
   - remat_scan(4,1) -> degenerate (inner=1 has no scan)
   - The 8-while-loop pattern seen in Linen MaxText comes from the PIPELINE
     wrapping scan around individual microstep bodies with per-body remat,
     not from remat_scan.

4. CRITICAL: lift.scan/lift.checkpoint work perfectly with NNX state via
   flax.core.bind() / flax.core.apply(). No Linen Module needed at all.
   The Scope is created from a plain {"params": {...}} dict.

IMPLICATION FOR NNX PORT:
  The _partial_pack mechanism works fine with NNX pytrees as long as they
  are wrapped in Linen variable-dict format {"params": {name: array}}.
  The 8-while-loop pattern requires the PIPELINE-level orchestration
  (circular buffer with explicit fwd/bwd microstep separation), not just
  scan+remat nesting.
""")


if __name__ == "__main__":
    main()
