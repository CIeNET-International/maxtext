import jax
import jax.numpy as jnp
from flax import linen as nn
import functools

class SimpleLayer(nn.Module):
    """Layer that takes (carry, scan_input) -> (carry, scan_output) for nn.scan."""
    features: int
    @nn.compact
    def __call__(self, carry, _scan_input):
        # carry is the activations, _scan_input is unused (we broadcast params)
        x = nn.Dense(self.features)(carry)
        return x, None  # (new_carry, scan_output)

# Create scanned + remat version
ScanLayer = nn.scan(
    nn.remat(SimpleLayer),
    variable_broadcast='params',
    variable_carry=[],
    split_rngs={'params': False},
    length=4,
)

model = ScanLayer(features=32)
x = jnp.ones((1, 32))
# nn.scan needs a scan input; use None-shaped placeholder
variables = model.init(jax.random.key(0), x, None)

# Get the JAXPR
def loss(params, x):
    carry, _ = model.apply(params, x, None)
    return carry.sum()

jaxpr = jax.make_jaxpr(jax.grad(loss))(variables, x)
print("=" * 80)
print("CASE 1: nn.scan(nn.remat) -- Top-level JAXPR primitives")
print("=" * 80)
for eqn in jaxpr.jaxpr.eqns:
    print(f"  {eqn.primitive.name}", end="")
    if hasattr(eqn, 'params'):
        for k, v in eqn.params.items():
            if k in ('jaxpr', 'call_jaxpr', 'body_jaxpr', 'cond_jaxpr'):
                sub = v.jaxpr if hasattr(v, 'jaxpr') else v
                if hasattr(sub, 'eqns'):
                    print(f" [{k}: {len(sub.eqns)} eqns]", end="")
            elif k == 'num_consts':
                print(f" [{k}={v}]", end="")
            elif k == 'num_carry':
                print(f" [{k}={v}]", end="")
            elif k == 'linear':
                print(f" [{k}={v}]", end="")
            elif k == 'reverse':
                print(f" [{k}={v}]", end="")
            elif k == 'length':
                print(f" [{k}={v}]", end="")
            elif k == 'unroll':
                print(f" [{k}={v}]", end="")
    print()

# Count while loops in lowered HLO
lowered = jax.jit(jax.grad(loss)).lower(variables, x)
mlir_text = lowered.as_text()
print(f"\n  -> While loops in MLIR: {mlir_text.count('stablehlo.while')}")

# ========== CASE 2: WITHOUT remat ==========
print("\n" + "=" * 80)
print("CASE 2: nn.scan (NO remat) -- Top-level JAXPR primitives")
print("=" * 80)

ScanLayerNoRemat = nn.scan(
    SimpleLayer,
    variable_broadcast='params',
    variable_carry=[],
    split_rngs={'params': False},
    length=4,
)
model_nr = ScanLayerNoRemat(features=32)
vars_nr = model_nr.init(jax.random.key(0), x, None)
def loss_nr(params, x):
    carry, _ = model_nr.apply(params, x, None)
    return carry.sum()
jaxpr_nr = jax.make_jaxpr(jax.grad(loss_nr))(vars_nr, x)
for eqn in jaxpr_nr.jaxpr.eqns:
    print(f"  {eqn.primitive.name}", end="")
    if hasattr(eqn, 'params'):
        for k, v in eqn.params.items():
            if k in ('jaxpr', 'call_jaxpr', 'body_jaxpr', 'cond_jaxpr'):
                sub = v.jaxpr if hasattr(v, 'jaxpr') else v
                if hasattr(sub, 'eqns'):
                    print(f" [{k}: {len(sub.eqns)} eqns]", end="")
            elif k in ('num_consts', 'num_carry', 'linear', 'reverse', 'length', 'unroll'):
                print(f" [{k}={v}]", end="")
    print()
lowered_nr = jax.jit(jax.grad(loss_nr)).lower(vars_nr, x)
mlir_nr = lowered_nr.as_text()
print(f"\n  -> While loops in MLIR: {mlir_nr.count('stablehlo.while')}")

# ========== CASE 3: raw jax.lax.scan + jax.checkpoint ==========
print("\n" + "=" * 80)
print("CASE 3: raw jax.lax.scan(jax.checkpoint) -- Top-level JAXPR primitives")
print("=" * 80)

params_raw = jax.random.normal(jax.random.key(0), (32, 32))
def raw_loss(params, x):
    def body(carry, _):
        return jax.checkpoint(lambda c: c @ params)(carry), None
    final, _ = jax.lax.scan(body, x, None, length=4)
    return final.sum()

jaxpr_raw = jax.make_jaxpr(jax.grad(raw_loss))(params_raw, x)
for eqn in jaxpr_raw.jaxpr.eqns:
    print(f"  {eqn.primitive.name}", end="")
    if hasattr(eqn, 'params'):
        for k, v in eqn.params.items():
            if k in ('jaxpr', 'call_jaxpr', 'body_jaxpr', 'cond_jaxpr'):
                sub = v.jaxpr if hasattr(v, 'jaxpr') else v
                if hasattr(sub, 'eqns'):
                    print(f" [{k}: {len(sub.eqns)} eqns]", end="")
            elif k in ('num_consts', 'num_carry', 'linear', 'reverse', 'length', 'unroll'):
                print(f" [{k}={v}]", end="")
    print()
lowered_raw = jax.jit(jax.grad(raw_loss)).lower(params_raw, x)
mlir_raw = lowered_raw.as_text()
print(f"\n  -> While loops in MLIR: {mlir_raw.count('stablehlo.while')}")

# ========== DEEP PRIMITIVE TREE ==========
print("\n" + "=" * 80)
print("DEEP PRIMITIVE TREE COMPARISON")
print("=" * 80)

def collect_primitives(jaxpr_obj, depth=0, max_depth=3):
    results = []
    eqns = jaxpr_obj.eqns if hasattr(jaxpr_obj, 'eqns') else []
    for eqn in eqns:
        name = eqn.primitive.name
        results.append((depth, name, eqn.params))
        if depth < max_depth:
            for k, v in eqn.params.items():
                if k in ('jaxpr', 'call_jaxpr', 'body_jaxpr'):
                    sub = v.jaxpr if hasattr(v, 'jaxpr') else v
                    if hasattr(sub, 'eqns'):
                        results.extend(collect_primitives(sub, depth+1, max_depth))
    return results

print("\n--- nn.scan(nn.remat) primitive tree ---")
for depth, name, params in collect_primitives(jaxpr.jaxpr, max_depth=4):
    extra = ""
    for k, v in params.items():
        if k in ('num_consts', 'num_carry', 'reverse', 'length', 'differentiated'):
            extra += f" {k}={v}"
    print(f"  {'  '*depth}{name}{extra}")

print("\n--- raw scan+checkpoint primitive tree ---")
for depth, name, params in collect_primitives(jaxpr_raw.jaxpr, max_depth=4):
    extra = ""
    for k, v in params.items():
        if k in ('num_consts', 'num_carry', 'reverse', 'length', 'differentiated'):
            extra += f" {k}={v}"
    print(f"  {'  '*depth}{name}{extra}")

print("\n--- nn.scan (no remat) primitive tree ---")
for depth, name, params in collect_primitives(jaxpr_nr.jaxpr, max_depth=4):
    extra = ""
    for k, v in params.items():
        if k in ('num_consts', 'num_carry', 'reverse', 'length', 'differentiated'):
            extra += f" {k}={v}"
    print(f"  {'  '*depth}{name}{extra}")

# ========== FULL JAXPR DUMPS ==========
print("\n" + "=" * 80)
print("FULL JAXPR: nn.scan(nn.remat)")
print("=" * 80)
print(jaxpr)

print("\n" + "=" * 80)
print("FULL JAXPR: raw scan+checkpoint")
print("=" * 80)
print(jaxpr_raw)

print("\n" + "=" * 80)
print("FULL JAXPR: nn.scan (no remat)")
print("=" * 80)
print(jaxpr_nr)
