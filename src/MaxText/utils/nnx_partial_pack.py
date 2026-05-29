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

"""NNX-native equivalent of Flax Linen's ``_partial_pack`` scope isolation.

Background
----------
The MaxText circular pipeline uses ~6.6 GB more compile-time memory under NNX
(29.9 GB) than under Linen (23.3 GB). Linen wins because ``nn.scan`` is built on
``flax.core.lift._partial_pack`` (``flax/core/lift.py``, ~lines 125-278).

``_partial_pack`` works as follows:

1. It groups a scope's variables into collections (``group_collections``).
2. Loop-invariant collections (those NOT in ``out_variable_filters``) are
   ``freeze``-d so they cannot mutate.
3. ``scope_fn`` rebuilds a brand-new ``Scope(parent=None)`` -- an *isolated*
   scope that has no link back to the enclosing module tree.
4. The transformed function (e.g. the scan body) reads variables out of, and
   writes them back into, that isolated scope through ``scope_fn`` /
   ``repack_fn``. Those reads/writes are a *real traced function boundary*: the
   variables enter the body as explicit traced inputs and leave as explicit
   traced outputs.

The consequence at the XLA level: each ``_partial_pack`` boundary that wraps a
``jax.lax.scan`` produces a *nested computation*. The Linen pipeline ends up
with 8 HLO ``while`` loops. The naive NNX port -- ``jax.lax.scan`` over a body
captured as a Python closure -- has no such boundary and produces only 4.

Why the obvious NNX replacements do NOT work
--------------------------------------------
The key requirement is a boundary that **survives lowering to HLO as a real,
separately-scheduled computation**.

* ``jax.remat`` / ``jax.checkpoint`` -- emits ``remat_p``, a *transparent*
  jaxpr marker. It does NOT lower to a separate XLA computation. (V45b: nested
  checkpoint, +6.2 GB worse, only 5 while loops.)
* ``jax.jit(..., inline=True)`` -- the body is inlined into the caller during
  tracing. The resulting jaxpr is byte-identical to the no-boundary baseline.
  (V45: no effect.)
* ``jax.named_call`` -- in JAX 0.9.x this is inlined away during tracing; no
  ``named_call`` / ``closed_call`` primitive survives in the jaxpr.

What DOES work: ``jax.jit`` with ``inline=False`` (the default). It emits the
``pjit`` primitive, which:

* survives in the jaxpr -- verified to remain even *inside* a ``jax.lax.scan``
  body and even after ``jax.grad`` (the ``jit`` eqn appears in both the primal
  and the transposed jaxpr);
* lowers to HLO with nested ``called_computations`` -- a genuine separate XLA
  computation, exactly the kind of boundary ``_partial_pack`` produces.

Mechanism implemented here
--------------------------
``nnx_partial_pack_scan`` replicates ``_partial_pack`` for NNX state:

* ``broadcast_state`` (an ``nnx.State`` of loop-invariant params) plays the role
  of the *frozen* collections -- it is passed as an **explicit positional
  argument** through the ``jax.jit`` boundary, never as a closure capture. This
  is the analogue of ``_partial_pack`` freezing in-only collections and feeding
  them into ``scope_fn`` as explicit inputs.
* ``carry_state`` (an ``nnx.State`` of mutables) plays the role of the
  out-mutable collections -- it is threaded through the ``jax.lax.scan`` carry,
  the analogue of variables that ``repack_fn`` writes back.
* The per-iteration body runs *inside* a ``jax.jit(inline=False)`` call. That
  jit is the isolated, traced boundary -- the NNX-native ``Scope(parent=None)``.

The result is a real nested computation around the scan body, mirroring the
extra HLO ``while`` nesting that ``_partial_pack`` gives Linen.

Caveat
------
This is a structural analogue, not a byte-for-byte reproduction of Linen's
collection bookkeeping. Whether XLA's memory scheduler reacts to the extra
``pjit`` boundary the same way it reacts to ``_partial_pack``'s nesting is an
empirical question -- it MUST be confirmed by measuring HLO ``while`` count and
peak memory.
"""

from typing import Any, Callable

import jax
from flax import nnx


__all__ = ["nnx_partial_pack_scan", "nnx_partial_pack_call"]


def nnx_partial_pack_call(
    body_fn: Callable[..., Any],
    broadcast_state: Any,
    *call_args: Any,
    name: str | None = None,
) -> Any:
  """Run ``body_fn`` once behind an isolated, traced ``jax.jit`` boundary.

  This is the single-call building block (the analogue of one ``scope_fn`` /
  ``repack_fn`` round trip in ``_partial_pack``). ``nnx_partial_pack_scan`` is
  built on top of it.

  ``broadcast_state`` is passed as the FIRST explicit positional argument to a
  ``jax.jit(inline=False)``-wrapped function. Because it is an explicit argument
  (not a closure capture) and because ``inline=False`` keeps the ``pjit``
  primitive in the jaxpr, the loop-invariant params cross a real traced boundary
  -- exactly as variables cross ``_partial_pack``'s ``Scope(parent=None)``.

  Args:
    body_fn: callable ``(broadcast_state, *call_args) -> output``. It is traced
      inside the jit boundary. It MUST be functionally pure: it may read
      ``broadcast_state`` but must not mutate it, and it must not rely on
      Python side effects to communicate results (return them instead).
    broadcast_state: loop-invariant pytree (typically an ``nnx.State`` of
      params). Frozen-by-convention: treated as read-only across the boundary.
    *call_args: remaining arguments forwarded to ``body_fn`` as explicit jit
      arguments (typically the carry pytree and the per-step input slice).
    name: optional debug name for the jit boundary.

  Returns:
    Whatever ``body_fn`` returns.
  """

  def _isolated(b_state, *args):
    # Inside this function we are at a fresh trace level -- the NNX analogue of
    # `_partial_pack`'s `Scope(parent=None)`. `b_state` arrived as an explicit
    # traced input, never as a closure constant.
    return body_fn(b_state, *args)

  # inline=False (the DEFAULT) is load-bearing: it keeps the `pjit` primitive in
  # the jaxpr so the boundary survives to HLO as a real nested computation.
  # inline=True would inline the body and erase the boundary entirely.
  jitted = jax.jit(_isolated, inline=False)
  if name is not None:
    # jax.named_scope only annotates; it does not create a primitive. Harmless.
    with jax.named_scope(name):
      return jitted(broadcast_state, *call_args)
  return jitted(broadcast_state, *call_args)


def nnx_partial_pack_scan(
    body_fn: Callable[..., Any],
    graphdef: Any,
    broadcast_state: Any,
    carry_state: Any,
    length: int,
    carry_filter: Any,
    carry_data: Any = None,
    xs: Any = None,
    remat_policy: Any = None,
    unroll: int = 1,
    name: str = "nnx_partial_pack",
) -> Any:
  """``jax.lax.scan`` whose body runs behind an isolated traced boundary.

  NNX-native analogue of ``nn.scan`` built on ``_partial_pack``. The body is
  executed inside a ``jax.jit(inline=False)`` call, so:

  * ``broadcast_state`` -- the loop-invariant params -- crosses that boundary as
    an explicit traced argument (the analogue of ``_partial_pack`` freezing
    in-only collections and passing them through ``scope_fn``);
  * ``carry_state`` -- the mutables -- is threaded through ``jax.lax.scan`` as
    carry (the analogue of out-mutable collections written by ``repack_fn``).

  The jit boundary inside the scan body is intended to make XLA emit a nested
  computation around the loop, mirroring the extra HLO ``while`` nesting that
  ``_partial_pack`` produces for the Linen pipeline.

  Reconstruction inside the body uses ``nnx.merge(graphdef, merged_state)``;
  ``graphdef`` must be the ``nnx.GraphDef`` that pairs with
  ``nnx.State.merge(broadcast_state, carry_state, ...)``.

  Args:
    body_fn: callable ``(module, carry_data, xs_i) -> (carry_data_out, ys_i)``.
      ``module`` is the NNX module rebuilt from ``graphdef`` + the merged state.
      ``carry_data`` is the non-module scan carry (e.g. a ``loop_state`` dict).
      ``xs_i`` is the current ``xs`` slice (``None`` if ``xs`` is ``None``).
      The body must NOT mutate the broadcast params in place.
    graphdef: ``nnx.GraphDef`` produced alongside ``broadcast_state`` /
      ``carry_state`` by ``nnx.split``. Used to rebuild the module each step.
    broadcast_state: ``nnx.State`` of loop-invariant params. Passed as an
      explicit argument through the jit boundary; constant across iterations.
    carry_state: ``nnx.State`` of mutables. Threaded through the scan carry.
    length: number of scan iterations.
    carry_filter: NNX filter selecting the carry/mutable variables (the same
      filter used to produce ``carry_state``, e.g. ``nnx.Not(nnx.Param)`` or an
      ``RngState`` filter). Used to re-extract the mutated carry portion after
      each step so the scan carry keeps a stable pytree structure.
    carry_data: initial non-module carry (e.g. ``loop_state``). May be ``None``.
    xs: per-step stacked inputs, or ``None``.
    remat_policy: optional ``jax.checkpoint`` policy. When provided, the body is
      additionally wrapped in ``jax.checkpoint`` for activation
      rematerialization. NOTE: ``remat`` is a transparent jaxpr marker -- it
      provides the memory/recompute trade-off but is NOT the isolation boundary;
      the ``jax.jit`` is. ``remat`` is applied INSIDE the jit so the boundary
      still survives.
    unroll: ``jax.lax.scan`` unroll factor.
    name: debug name for the jit boundary.

  Returns:
    ``(final_carry_state, final_carry_data, ys)``:
      * ``final_carry_state`` -- ``nnx.State`` of mutables after the last step;
      * ``final_carry_data`` -- the non-module carry after the last step;
      * ``ys`` -- stacked per-step outputs.
  """

  def _packed_body(b_state, c_state, c_data, xs_i):
    """Pure body run inside the jit boundary -- the isolated 'scope'.

    ``b_state`` (broadcast params) and ``c_state`` (carry mutables) both arrive
    as explicit traced inputs of the jit -- the NNX analogue of variables
    flowing through ``_partial_pack``'s ``Scope(parent=None)``.
    """
    merged = nnx.State.merge(b_state, c_state)
    module = nnx.merge(graphdef, merged)

    c_data_out, ys_i = body_fn(module, c_data, xs_i)

    # Re-extract ONLY the mutated carry portion via `carry_filter`. The
    # broadcast params are static across the scan (their gradients are
    # accumulated by AD as scan constants) and must NOT re-enter the carry, or
    # the scan input/output carry pytrees would differ in structure.
    _, new_carry_state, _ = nnx.split(module, carry_filter, ...)
    return c_data_out, new_carry_state, ys_i

  if remat_policy is not None:
    # remat is applied INSIDE the jit boundary: jit stays the outermost wrapper
    # so the `pjit` primitive (the real boundary) is not buried under `remat_p`.
    _packed_body = jax.checkpoint(_packed_body, policy=remat_policy)

  # inline=False (default) keeps the `pjit` primitive -> real HLO computation.
  _isolated_body = jax.jit(_packed_body, inline=False)

  def _scan_body(carry, xs_i):
    c_state, c_data = carry
    # broadcast_state is CLOSED OVER the scan (it is a loop constant) but is an
    # EXPLICIT ARGUMENT to the jit boundary inside `_isolated_body`. That is the
    # crucial difference from the naive port, where params reach the body only
    # as a closure and XLA sees no boundary at all.
    c_data_out, new_c_state, ys_i = _isolated_body(
        broadcast_state, c_state, c_data, xs_i
    )
    return (new_c_state, c_data_out), ys_i

  (final_carry_state, final_carry_data), ys = jax.lax.scan(
      _scan_body,
      (carry_state, carry_data),
      xs,
      length=length,
      unroll=unroll,
  )
  return final_carry_state, final_carry_data, ys
