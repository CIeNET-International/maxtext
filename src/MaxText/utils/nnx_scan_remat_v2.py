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

"""NNX scan_with_remat using Flax core Scope + _partial_pack for scope isolation.

Strategy
--------
Instead of trying to replicate ``_partial_pack`` in NNX land (which all prior
variants attempted -- V1 through V46), this module goes the other way:
it converts NNX state INTO Linen's native format and lets the real
``_partial_pack`` + ``lift.scan`` + ``lift.checkpoint`` do the work.

The flow:

1. Take NNX state (from ``nnx.split``), an arbitrary pytree of arrays.
2. Flatten to named variables, then nest under the ``'params'`` collection
   name to form a standard Linen variable dict:
   ``{'params': {'v0': array0, 'v1': array1, ...}}``.
3. Call ``flax.core.apply(core_fn, mutable=False)({'params': ...}, carry)``
   which creates a real ``Scope`` from that variable dict via ``bind()``.
4. Inside ``core_fn(scope, carry)``, call ``lift.scan(lift.checkpoint(body))``
   which invokes ``pack()`` -> ``_partial_pack()`` on the scope. This is the
   REAL Linen machinery -- no simulation, no approximation.
5. ``_partial_pack`` freezes the broadcast (params) collection, creates an
   isolated ``Scope(parent=None)`` for the body, and feeds variables as
   explicit traced inputs. This is exactly what produces the extra HLO
   ``while`` loops in Linen.
6. The body reads params from the scope, calls the user function with those
   arrays plus the carry, and returns updated carry + outputs.
7. After ``flax.core.apply`` returns, we have the final carry and stacked
   outputs -- pure pytrees of arrays, ready for use in NNX land.

Why this works
--------------
``_partial_pack`` operates on ``Scope`` objects. ``Scope`` is just a container
for ``{collection_name: {var_name: array}}`` dicts -- it does NOT require an
``nn.Module``. ``flax.core.apply`` creates a ``Scope`` from any variable dict
via ``bind()``. So we can use ``_partial_pack``'s scope isolation without ever
touching ``nn.Module``.

The critical property: ``lift.scan`` calls ``pack()`` which calls
``_partial_pack()``, which creates a ``Scope(parent=None)`` -- the isolated
scope that is the traced boundary. Inside that, ``lift.checkpoint`` wraps the
body in ``jax.remat`` with variables as explicit positional args. The nesting
of scan(remat(body)) through pack/pack produces the nested HLO computations
that Linen's pipeline achieves.

While loop count
----------------
A single ``lift.scan(lift.checkpoint(body))`` produces 2 HLO while loops
(1 forward, 1 backward) -- the SAME as raw ``jax.lax.scan(jax.checkpoint())``.
The _partial_pack scope isolation shows up as ``func.call @closed_call`` INSIDE
the while body (a nested computation boundary), not as additional while loops.

To get >4 while loops (the 8-while pattern seen in Linen pipelines), use
``remat_scan`` which does NESTED scans: ``scan(remat(scan(remat(body))))``
with ``lengths=(l1, l2)`` where ``l1 * l2 = total_iterations``. Each nesting
level adds its own forward+backward while pair.
"""

import functools
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from flax.core import lift
from flax.core.scope import apply as flax_core_apply
from flax import traverse_util


__all__ = [
    "scan_with_remat",
    "scan_with_remat_nested",
    "remat_scan",
    "scan_with_remat_full",
    "nnx_state_to_linen_vars",
    "linen_vars_to_flat_dict",
]


# ---------------------------------------------------------------------------
# State conversion helpers
# ---------------------------------------------------------------------------

def nnx_state_to_linen_vars(
    state: nnx.State,
    collection: str = "params",
) -> dict[str, dict]:
    """Convert an NNX State to a Linen-style variable dict.

    NNX State is a pytree whose leaves are arrays (after ``nnx.split``).
    Linen expects ``{collection: {nested_path: array}}``.

    The State's internal structure is already a nested dict (e.g.
    ``{'layers': {'Dense_0': {'kernel': array}}}``), so we just wrap it
    under the collection name.

    Args:
        state: An ``nnx.State`` (the raw dict form from ``nnx.split``).
        collection: The Linen collection name (default ``'params'``).

    Returns:
        ``{collection: nested_dict_of_arrays}`` suitable for ``flax.core.apply``.
    """
    raw = _state_to_nested_dict(state)
    return {collection: raw}


def _state_to_nested_dict(state: Any) -> Any:
    """Recursively convert nnx.State / nnx.Variable wrappers to plain dicts."""
    if isinstance(state, nnx.State):
        return {k: _state_to_nested_dict(v) for k, v in state.items()}
    if isinstance(state, dict):
        return {k: _state_to_nested_dict(v) for k, v in state.items()}
    # Leaf: could be an nnx.Variable wrapper or a raw array
    if hasattr(state, "value"):
        return state.value
    return state


def linen_vars_to_flat_dict(
    variables: dict[str, Any],
    collection: str = "params",
) -> dict[tuple[str, ...], Any]:
    """Flatten a Linen variable dict to ``{path_tuple: array}``."""
    nested = variables.get(collection, {})
    return traverse_util.flatten_dict(nested)


# ---------------------------------------------------------------------------
# Internal helpers for pytree <-> Linen scope conversion
# ---------------------------------------------------------------------------

def _pytree_to_linen_vars(
    broadcast_arrays: Any,
) -> tuple[dict[str, dict], list[str], Any]:
    """Convert an arbitrary pytree to a Linen variable dict.

    Returns:
        (variables, var_names, treedef) where:
        - variables: ``{'params': {'v0': arr0, 'v1': arr1, ...}}``
        - var_names: list of variable names in order
        - treedef: the original pytree structure for reconstruction
    """
    flat_arrays, treedef = jax.tree_util.tree_flatten(broadcast_arrays)
    var_names = [f"v{i}" for i in range(len(flat_arrays))]
    params_dict = {name: arr for name, arr in zip(var_names, flat_arrays)}
    variables = {"params": params_dict}
    return variables, var_names, treedef


def _recover_pytree_from_scope(scope, var_names: list[str], treedef) -> Any:
    """Reconstruct the original pytree from a Linen scope's variables."""
    recovered_flat = []
    for name in var_names:
        recovered_flat.append(scope.get_variable("params", name))
    return treedef.unflatten(recovered_flat)


# ---------------------------------------------------------------------------
# Core: scan_with_remat via flax.core.apply + lift.scan + lift.checkpoint
# ---------------------------------------------------------------------------

def scan_with_remat(
    body_fn: Callable,
    broadcast_arrays: Any,
    init_carry: Any,
    length: int,
    remat_policy: Callable[..., bool] | None = None,
    prevent_cse: bool = False,
    split_rngs: dict[str, bool] | None = None,
    unroll: int = 1,
) -> tuple[Any, Any]:
    """Coordinated scan+remat with _partial_pack scope isolation.

    Uses the real Flax Linen ``lift.scan(lift.checkpoint(body))`` pipeline
    by converting broadcast arrays to Linen variable format and running
    through ``flax.core.apply``. Gets ``_partial_pack``'s tracer isolation.

    Each leaf in ``broadcast_arrays`` is stored as a separate named variable
    in the ``'params'`` Linen collection, giving ``_partial_pack``
    fine-grained control over variable freezing.

    Args:
        body_fn: ``(carry, broadcast_arrays) -> (new_carry, output)``.
            ``broadcast_arrays`` has the same pytree structure as the input.
        broadcast_arrays: Loop-invariant pytree of arrays.
        init_carry: Initial carry value for the scan.
        length: Number of scan iterations.
        remat_policy: Optional ``jax.checkpoint`` policy. ``None`` means no
            rematerialization.
        prevent_cse: Whether to prevent CSE in checkpoint. Default ``False``.
        split_rngs: RNG split configuration. Default ``None`` (no RNGs).
        unroll: Scan unroll factor.

    Returns:
        ``(final_carry, stacked_outputs)``.
    """
    if split_rngs is None:
        split_rngs = {}

    variables, var_names, treedef = _pytree_to_linen_vars(broadcast_arrays)

    def core_fn(scope, carry):
        def scan_body(scope, carry):
            recovered = _recover_pytree_from_scope(scope, var_names, treedef)
            new_carry, output = body_fn(carry, recovered)
            return new_carry, output

        checkpointed_body = lift.checkpoint(
            scan_body,
            variables=True,
            rngs=True,
            prevent_cse=prevent_cse,
            policy=remat_policy,
        )

        scanned_fn = lift.scan(
            checkpointed_body,
            variable_broadcast="params",
            variable_carry=False,
            split_rngs=split_rngs,
            length=length,
            unroll=unroll,
        )

        return scanned_fn(scope, carry)

    result = flax_core_apply(core_fn, mutable=False)(variables, init_carry)
    final_carry, stacked_outputs = result
    return final_carry, stacked_outputs


# Alias for backward compat with earlier naming
scan_with_remat_nested = scan_with_remat


# ---------------------------------------------------------------------------
# remat_scan: nested scan+remat for >4 HLO while loops
# ---------------------------------------------------------------------------

def remat_scan(
    body_fn: Callable,
    broadcast_arrays: Any,
    init_carry: Any,
    lengths: Sequence[int],
    remat_policy: Callable[..., bool] | None = None,
    prevent_cse: bool = False,
    split_rngs: dict[str, bool] | None = None,
) -> tuple[Any, Any]:
    """Nested scan+remat using Linen's ``lift.remat_scan`` via ``flax.core.apply``.

    This is the function that produces >4 HLO while loops. It wraps
    ``lift.remat_scan`` which does nested ``scan(remat(scan(remat(body))))``
    for ``lengths=(l1, l2)``. Each nesting level produces its own
    forward+backward while pair, giving 4+ total while loops.

    With ``lengths=(l1, l2)``, the total iterations are ``l1 * l2``.
    Memory consumption is proportional to ``n^(1/d)`` where
    ``d = len(lengths)``. This is the same mechanism Linen uses for
    O(sqrt(N)) memory with respect to model depth.

    Args:
        body_fn: ``(carry, broadcast_arrays) -> new_carry``.
            NOTE: unlike ``scan_with_remat``, this returns only ``new_carry``
            (no per-step output), matching ``lift.remat_scan``'s interface.
        broadcast_arrays: Loop-invariant pytree of arrays.
        init_carry: Initial carry value.
        lengths: Tuple of lengths for nested scan levels. Total iterations
            = ``prod(lengths)``. E.g. ``(4, 4)`` for 16 iterations with
            O(sqrt(16)) = O(4) memory.
        remat_policy: Optional checkpoint policy.
        prevent_cse: CSE prevention for inner remat.
        split_rngs: RNG split configuration.

    Returns:
        ``(final_carry, stacked_outputs)`` where ``stacked_outputs`` is
        an empty tuple (remat_scan has no per-step output).
    """
    if split_rngs is None:
        split_rngs = {}

    variables, var_names, treedef = _pytree_to_linen_vars(broadcast_arrays)

    def core_fn(scope, carry):
        def remat_scan_body(scope, carry):
            recovered = _recover_pytree_from_scope(scope, var_names, treedef)
            new_carry = body_fn(carry, recovered)
            return new_carry

        scanned = lift.remat_scan(
            remat_scan_body,
            lengths=lengths,
            policy=remat_policy,
            variable_broadcast="params",
            variable_carry=False,
            variable_axes={},
            split_rngs=split_rngs if split_rngs else {True: True},
        )
        final_carry = scanned(scope, carry)
        return final_carry, ()

    result = flax_core_apply(core_fn, mutable=False)(variables, init_carry)
    final_carry, empty = result
    return final_carry, empty


# ---------------------------------------------------------------------------
# Full NNX State variant
# ---------------------------------------------------------------------------

def scan_with_remat_full(
    body_fn: Callable,
    broadcast_state: nnx.State,
    init_carry: Any,
    length: int,
    remat_policy: Callable[..., bool] | None = None,
    prevent_cse: bool = False,
    unroll: int = 1,
) -> tuple[Any, Any]:
    """Scan+remat for full NNX State objects.

    Converts NNX State to Linen variables, runs through
    ``lift.scan(lift.checkpoint(body))`` via ``flax.core.apply``, and returns
    results in NNX-compatible format.

    Args:
        body_fn: ``(carry, state_dict) -> (new_carry, output)``.
            ``state_dict`` is a nested dict with the same structure as the
            input NNX State (but without Variable wrappers).
        broadcast_state: NNX State of loop-invariant params.
        init_carry: Initial carry value.
        length: Number of scan iterations.
        remat_policy: Optional checkpoint policy.
        prevent_cse: CSE prevention flag.
        unroll: Scan unroll factor.

    Returns:
        ``(final_carry, stacked_outputs)``.
    """
    # Convert NNX State to plain nested dict, then use scan_with_remat
    raw_dict = _state_to_nested_dict(broadcast_state)
    return scan_with_remat(
        body_fn=body_fn,
        broadcast_arrays=raw_dict,
        init_carry=init_carry,
        length=length,
        remat_policy=remat_policy,
        prevent_cse=prevent_cse,
        unroll=unroll,
    )
