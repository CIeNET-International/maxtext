# Copyright 2026 Google LLC
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

"""NNX-native coordinated scan+remat — equivalent to nn.scan(nn.remat(body)).

Replicates Linen's _partial_pack mechanism for NNX modules:
- Extracts ALL state via nnx.split (like _partial_pack's scope_fn)
- Flattens to raw arrays (like _partial_pack's variable_groups)
- Passes as explicit positional args to jax.remat (like lift.checkpoint)
- Uses axes_scan.scan for broadcast/carry/scan routing (like lift.scan)
- Reconstructs module via nnx.merge inside remat body (like scope_fn)

This mirrors the exact call structure that produces 8 HLO while loops
in Linen, vs 4 in raw jax.lax.scan(jax.checkpoint(body)).
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from flax.core.axes_scan import scan as axes_scan_fn
from flax.core.axes_scan import broadcast as axes_broadcast


def scan_with_remat(
    body_fn: Callable,
    graphdef: Any,
    broadcast_state: nnx.State,
    init_carry_state: nnx.State,
    init_user_carry: Any,
    length: int,
    remat_policy: Any = None,
    prevent_cse: bool = True,
    split_rngs: bool = True,
    unroll: int = 1,
):
  """Coordinated scan+remat for NNX modules.

  Equivalent to Linen's nn.scan(nn.remat(body)) — uses axes_scan.scan
  with broadcast annotation and jax.remat with all state as explicit
  positional args, mirroring _partial_pack's scope isolation.

  Args:
    body_fn: Function (graphdef, state, user_carry) -> (new_user_carry, output).
        Receives the full merged state (broadcast + carry) and returns
        updated carry and scan output. Must NOT modify broadcast state.
    graphdef: Static graph definition from nnx.split(module).
    broadcast_state: Loop-invariant state (e.g. params). NOT in scan carry.
    init_carry_state: Loop-varying state (e.g. RNG). In scan carry.
    init_user_carry: User-defined carry (e.g. loop_state dict).
    length: Number of scan iterations.
    remat_policy: jax.checkpoint policy for rematerialization.
    prevent_cse: Prevent common subexpression elimination across iterations.
    split_rngs: Ignored (kept for API compat — RNG folding handled by caller).
    unroll: Scan unroll factor.

  Returns:
    (final_carry_state, final_user_carry, broadcast_state_out), stacked_outputs
  """
  broadcast_treedef = jax.tree.structure(broadcast_state)
  carry_treedef = jax.tree.structure(init_carry_state)

  broadcast_flat = jax.tree.leaves(broadcast_state)
  carry_flat_init = jax.tree.leaves(init_carry_state)

  # --- The rematted body: ALL state as explicit flat-array args ---
  # This mirrors lift.checkpoint's rematted(variable_groups, rng_groups, *args)
  # where _partial_pack ensures all vars are positional.
  def rematted(broadcast_flat_in, carry_flat_in, user_carry):
    broadcast_st = broadcast_treedef.unflatten(broadcast_flat_in)
    carry_st = carry_treedef.unflatten(carry_flat_in)
    merged_state = nnx.State.merge(broadcast_st, carry_st)

    new_user_carry, output = body_fn(graphdef, merged_state, user_carry)

    _, new_broadcast_st, new_carry_st = nnx.split(
        merged_state, lambda _, v: v in broadcast_st, ...
    )
    new_carry_flat = jax.tree.leaves(new_carry_st)

    return new_carry_flat, new_user_carry, output

  if remat_policy is not None:
    rematted = jax.checkpoint(
        rematted, policy=remat_policy, prevent_cse=prevent_cse,
    )

  # --- axes_scan body: broadcast + carry + scan routing ---
  # Mirrors lift.scan's scanned(broadcast_vars, carry, scan_vars, rngs, args)
  def scan_body(broadcast_in, carry, *_no_xs):
    carry_flat_in, user_carry = carry

    new_carry_flat, new_user_carry, output = rematted(
        broadcast_in, carry_flat_in, user_carry
    )

    return broadcast_in, (new_carry_flat, new_user_carry), output

  scanned = axes_scan_fn(
      scan_body,
      in_axes=(),
      out_axes=0,
      length=length,
      unroll=unroll,
  )

  broadcast_out, (final_carry_flat, final_user_carry), stacked_outputs = scanned(
      broadcast_flat,
      (carry_flat_init, init_user_carry),
  )

  final_carry_state = carry_treedef.unflatten(final_carry_flat)
  final_broadcast_state = broadcast_treedef.unflatten(broadcast_out)

  return (final_broadcast_state, final_carry_state, final_user_carry), stacked_outputs


def scan_with_remat_simple(
    body_fn: Callable,
    broadcast_args: Any,
    init_carry: Any,
    length: int,
    remat_policy: Any = None,
    prevent_cse: bool = True,
    unroll: int = 1,
):
  """Simplified scan+remat for arbitrary pytrees (no NNX module dependency).

  Lower-level API: caller manages state splitting. Broadcast args are
  loop-invariant constants, carry is loop-varying.

  Args:
    body_fn: Function (broadcast_args, carry) -> (new_carry, output).
    broadcast_args: Loop-invariant values. Passed as explicit first arg
        to jax.remat (not closure-captured).
    init_carry: Initial carry value.
    length: Number of iterations.
    remat_policy: jax.checkpoint policy.
    prevent_cse: Prevent CSE across iterations.
    unroll: Scan unroll factor.

  Returns:
    (broadcast_out, final_carry), stacked_outputs
  """
  def rematted(broadcast, carry):
    return body_fn(broadcast, carry)

  if remat_policy is not None:
    rematted = jax.checkpoint(
        rematted, policy=remat_policy, prevent_cse=prevent_cse,
    )

  def scan_body(broadcast_in, carry, *_no_xs):
    new_carry, output = rematted(broadcast_in, carry)
    return broadcast_in, new_carry, output

  scanned = axes_scan_fn(
      scan_body,
      in_axes=(),
      out_axes=0,
      length=length,
      unroll=unroll,
  )

  broadcast_out, final_carry, stacked_outputs = scanned(
      broadcast_args, init_carry,
  )

  return (broadcast_out, final_carry), stacked_outputs
