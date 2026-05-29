# Double _partial_pack Analysis: What Linen Does That NNX Cannot

## Executive Summary

The 5.7 GB gap is NOT in the scan primitive (proven identical). It is in how
`nn.scan(nn.remat(body))` creates **two nested `pack()` / `_partial_pack()`
invocations** that restructure the jaxpr AROUND the scan, making params into
**explicit `scope_fn` args** at both the remat and scan levels. This produces a
specific jaxpr pattern where XLA's buffer scheduler can share gradient
accumulators across stacked dimensions. NNX captures params by closure, producing
separate per-iteration gradient buffers that XLA cannot share.

## The Double _partial_pack Mechanism (Linen)

### Level 1: nn.remat wraps body in pack()

```python
# flax/core/lift.py:1497
def checkpoint(fn, ...):
    def inner(scope_fn, repack_fn, variable_groups, rng_groups, *args):
        @jax.remat
        def rematted(variable_groups, rng_groups, *args):
            scope = scope_fn(variable_groups, rng_groups)  # <-- params rebuilt from EXPLICIT args
            y = fn(scope, *args)
            return y, repack_fn(scope)
        return rematted(variable_groups, rng_groups, *args)

    return pack(inner, (variables,), (variables,), (rngs,), name='remat')
    #            ^-- _partial_pack #1: extracts vars from scope into explicit variable_groups
```

### Level 2: nn.scan wraps nn.remat'd body in pack()

```python
# flax/core/lift.py:1060
def scan(fn, variable_broadcast, variable_axes, ...):
    def inner(scope_fn, repack_fn, variable_groups, rng_groups, init, *args):
        @axes_scan.scan(...)
        def scanned(broadcast_vars, carry, scan_variable_groups, rng_groups, args):
            variable_groups = (broadcast_vars, carry_vars) + scan_variable_groups
            scope = scope_fn(variable_groups, rng_groups)  # <-- params rebuilt from EXPLICIT args
            c, y = fn(scope, c, *args)  # fn = the remat'd body
            out_vars = repack_fn(scope)
            return broadcast_vars_out, (carry_vars, c), (y, scan_vars)
        ...

    return pack(inner,
        (variable_broadcast, variable_carry) + variable_in_groups,  # <-- params listed TWICE
        (variable_broadcast, variable_carry) + variable_out_groups,
        rng_groups, name='scan')
    #            ^-- _partial_pack #2: re-extracts vars from scope
```

### What this creates in the jaxpr

When `create_flax_pipeline_scan` calls `nn.scan(nn.remat(body))`:

1. `_partial_pack` #2 (scan) extracts `_overwrite_with_gradient` and
   `non_trainable` as `broadcast_vars` and params (via `variable_axes`) as
   `scan_variable_groups`. These become EXPLICIT positional args to the
   `axes_scan.scan` function.

2. Inside the scan body, `scope_fn` reconstructs a Scope. Then `fn` (the
   remat'd body) is called.

3. `_partial_pack` #1 (remat) AGAIN extracts vars from the scope and passes
   them as explicit args to `jax.remat`. Inside remat, `scope_fn` #1
   reconstructs another Scope.

4. The actual computation `fn(scope, ...)` runs with params that were
   **threaded through two explicit arg boundaries**.

### The critical effect: `jax.remat` sees params as FUNCTION ARGS, not closures

```
Linen jaxpr structure:
  scan(
    body_fn(broadcast_vars, carry, scan_vars, rngs, args):   # params = scan_vars[0] (EXPLICIT)
      remat(
        rematted(variable_groups, rng_groups, *args):         # params = variable_groups[0] (EXPLICIT)
          scope = scope_fn(variable_groups, rngs)
          fn(scope, carry, *args)                             # actual computation
          repack_fn(scope)
      )
  )
```

Because params are explicit args to both `scan`'s body and `remat`'s body:
- `jax.remat` knows exactly which intermediates depend on "loop-invariant" inputs
  (params never change across scan iterations since they are `variable_broadcast`).
- The partial evaluator can hoist param-dependent constants out of the scan body.
- XLA sees params with a stacked leading dimension `[repeats, ...]` and allocates
  gradient buffers as `[1, repeats, ...]` — sharable across the stacked dim.

## What NNX Does Instead

```python
# NNX pipeline.py:1525
def outer_body(carry, _):
    # layers_params captured by Python closure (not explicit arg)
    w_next = self.weight_prefetching(layers_params, ...)  # closure reference
    bsw_ref[0] = (w_curr, w_next)
    jax.lax.scan(inner_body, ...)  # inner_body also closes over layers_params

def inner_body(carry, _):
    # bsw_ref[0] captured by closure (not explicit arg)
    stage_params = self.fetch_active_stage_weights(bsw_ref[0], ...)  # closure reference
    ...
```

```
NNX jaxpr structure:
  scan(
    outer_body(carry, _):                    # layers_params = CLOSURE (not in args)
      checkpoint(
        inner_body(carry, _):                # bsw_ref[0] = CLOSURE (not in args)
          fetch_active_stage_weights(...)
          vmap(forward)
      )
  )
```

Because params are closures, not explicit args:
- `jax.checkpoint` cannot distinguish param-dependent from carry-dependent intermediates.
- The partial evaluator cannot hoist param-dependent values out of the loop.
- XLA allocates separate gradient buffers per scan iteration for param-derived values,
  producing 3x BSW in temp buffers (confirmed by profiling).

## Why Previous Approaches Failed

| # | Approach | Why it failed |
|---|---------|---------------|
| 1-11 | Various | None of them changed the fundamental arg-vs-closure structure |
| V10 | stop_gradient(w_curr) | Removed ONE gradient path (0.9 GB) but didn't fix the closure issue |
| L1/L2/L3 | custom_vjp | Still has params as closure in outer scan |
| Pure functional body | No nnx.State | Params still captured by closure in scan body |

## Three New Approaches

### Approach N1: Lift params into scan carry with stop_gradient fence

**Key insight:** Make `layers_params` an explicit argument to the scan body by
including it in the carry, but prevent gradient accumulation by wrapping it in
`stop_gradient` at the START of each iteration and computing gradients via a
separate `jax.vjp` on the all-gather path only.

```python
def outer_body(carry, _):
    loop_state, layer_mutables, w_curr, params_explicit = carry

    # Params in carry but frozen — no grad accumulation through carry path
    params_explicit = jax.lax.stop_gradient(params_explicit)

    w_next = self.weight_prefetching(params_explicit, pps_full, iteration)
    bsw_ref[0] = (w_curr, w_next)

    (new_ls, new_mut), inner_metrics = jax.lax.scan(
        inner_body, (loop_state, layer_mutables), None, length=num_microbatches
    )
    return (new_ls, new_mut, w_next, params_explicit), inner_metrics
```

**Why this could work:**
- Params become EXPLICIT positional args to the `axes_scan.scan` primitive.
- `jax.checkpoint` sees `params_explicit` as a carry variable (explicit in the
  jaxpr), not an opaque closure.
- `stop_gradient` prevents gradient buffer duplication through the carry path.
- Gradients still flow correctly through `w_next = weight_prefetching(params_explicit)`.
- XLA can potentially share the param buffer across iterations since it is the
  same object passed through carry (carry sharing optimization).

**Risk:** The stop_gradient must be placed correctly so gradients still flow through
the prefetch path. The carry overhead is one extra copy of the param state per
iteration, but stop_gradient should tell XLA it does not need gradient buffers.

**Estimated impact:** 3-5 GB. Makes the structural pattern closer to Linen's
`variable_broadcast` (params visible to scan but gradient-frozen in carry).

### Approach N2: Explicit-arg checkpoint wrapper

**Key insight:** Replace `jax.checkpoint(inner_body)` with a wrapper that makes
BSW params an explicit argument to `jax.checkpoint`, matching the structure
`nn.remat` creates via its inner `pack()`.

```python
def make_explicit_checkpoint(fn, policy):
    """Wraps fn so that closed-over values become explicit args to jax.remat."""
    @jax.checkpoint(policy=policy)
    def remat_wrapper(carry, bsw_params, _):
        # bsw_params is now EXPLICIT arg to jax.remat, not closure
        return fn(carry, bsw_params, _)

    def wrapper(carry, _):
        return remat_wrapper(carry, bsw_ref[0], _)

    return wrapper
```

Then in `inner_body`:
```python
def inner_body_explicit(carry, bsw_params, _):
    current_loop_state, current_layer_mutables = carry
    iteration = current_loop_state["loop_iteration"]
    advanced_mutables = _advance_rng_state(current_layer_mutables, iteration)
    new_loop_state, new_layer_state = self.run_one_iteration(
        current_loop_state, bsw_params, layers_graph, layers_metrics,
        advanced_mutables, positions, segment_ids, deterministic,
        model_mode, logical_partition_spec_stripped,
    )
    ...
```

**Why this could work:**
- `jax.remat` now sees `bsw_params` as an explicit function argument in the jaxpr.
- The remat partial evaluator can distinguish "this intermediate depends on a
  function arg that is loop-invariant" vs "this depends on carry that changes."
- This is exactly what `nn.remat`'s `pack()` does: converts scope vars to explicit
  `variable_groups` args to `jax.remat`.
- No custom_vjp needed — pure structural change.

**Risk:** Low. The scan body still closes over the wrapper, but the checkpoint
boundary now has params as explicit args. Need to verify the scan correctly
threads bsw_params (it is reconstructed each outer iteration from bsw_ref[0],
which changes per repeat — so it is NOT truly loop-invariant within inner scan).
But within each inner scan iteration, bsw_params IS invariant.

**Estimated impact:** 3-5 GB. Directly addresses the "remat sees closure vs arg"
structural difference.

### Approach N3: Replicate Flax's scope_fn/repack_fn pattern manually

**Key insight:** Instead of fighting JAX's tracing semantics, directly replicate
what `_partial_pack` does: create explicit `scope_fn` and `repack_fn` callables
that thread params as explicit pytree args through both the scan and checkpoint
boundaries.

```python
# Before scan:
param_groups = layers_params  # the "broadcast" group
metric_groups = layers_metrics  # the "scan axis=0" group

def scope_fn(variable_groups, rng_groups):
    """Reconstruct full state from explicit variable groups."""
    params, metrics, mutables = variable_groups
    return nnx.State.merge(params, metrics, mutables)

def repack_fn(full_state):
    """Extract variable groups back from full state."""
    _, params, metrics, mutables = nnx.split(full_state, _is_static_param, nnx.Intermediate, ...)
    return (params, metrics, mutables)

def scan_body(carry, _):
    carry_vars, c = carry  # carry_vars = (carry_metrics, carry_mutables)
    variable_groups = (param_groups, carry_vars[0], carry_vars[1])

    @jax.checkpoint(policy=remat_policy)
    def remat_body(variable_groups, rng_groups, c):
        full_state = scope_fn(variable_groups, rng_groups)
        # ... run iteration ...
        new_groups = repack_fn(new_state)
        return new_groups, new_c

    new_variable_groups, new_c = remat_body(variable_groups, rng_groups, c)
    new_carry_vars = (new_variable_groups[1], new_variable_groups[2])
    return (new_carry_vars, new_c), new_variable_groups[1]  # metrics as scan output
```

**Why this could work:**
- Perfectly mirrors the Linen `_partial_pack` pattern at the Python level.
- `param_groups` is an explicit first arg to `remat_body` — `jax.remat` sees it
  as a function arg, not closure. Within `jax.remat`'s jaxpr, it becomes a
  known input that the partial evaluator can reason about.
- `param_groups` is passed from the outer scope (closure of `scan_body`) — same
  as Linen's `broadcast_vars` pattern where broadcast vars are closed over by
  the scan body but explicit to remat.
- No Flax dependency — pure JAX pattern.

**Risk:** Medium. Requires restructuring the entire scan body to thread variable
groups explicitly. The `scope_fn`/`repack_fn` abstractions must be kept
consistent. But this is a one-time refactor, not an ongoing maintenance burden.

**Estimated impact:** 5-6 GB. This is the most structurally faithful reproduction
of the Linen pattern and should produce near-identical jaxpr structure around
the scan.

## Recommendation

**Start with N2** (explicit-arg checkpoint wrapper) — smallest code change, tests
the hypothesis that making BSW an explicit arg to `jax.checkpoint` is sufficient.

**If N2 shows partial improvement, follow with N3** — full scope_fn/repack_fn
reproduction for the remaining gap.

**N1** is a fallback if N2/N3 don't work, with the trade-off of params in carry.
