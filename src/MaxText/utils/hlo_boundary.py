# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HLO computation-boundary primitive for the NNX pipeline (V49, Solution 6).

Problem
-------
The MaxText circular pipeline ported to Flax NNX consumes ~29.9 GB of
compile-time memory, versus ~23.3 GB for the original Linen implementation --
a 6.6 GB gap. The gap is traced to scope isolation: Linen's ``_partial_pack``
emits *separate* XLA computations (8 distinct HLO ``while`` loops), which gives
XLA's buffer assignment pass tighter, non-overlapping live ranges. The NNX port
uses ``jax.checkpoint`` + ``jax.lax.scan`` and collapses to only 4 ``while``
loops, so XLA fuses regions whose buffers would otherwise be disjoint, inflating
peak memory.

Solution 6 -- a hard HLO boundary
---------------------------------
We need an op that is a *mathematical identity* (no effect on forward values or
on gradients) but that XLA's optimizer **cannot inline, fuse across, or
eliminate**. Inserting such an op around the scan carry forces XLA to treat the
pre-boundary and post-boundary buffers as distinct scheduling regions, the same
property ``_partial_pack`` gets for free.

``jax.lax.optimization_barrier`` is exactly this op:

* It lowers to ``stablehlo.optimization_barrier`` (XLA ``kOptimizationBarrier``),
  a real HLO instruction. XLA's algebraic simplifier, CSE, fusion and
  buffer-assignment passes are all explicitly forbidden from moving data-flow
  across it -- it is the primitive JAX itself uses to implement
  rematerialization. It therefore *survives* every optimization pass; there is
  no "identity op" for XLA to fold away.
* It is a true identity for forward values.
* In current JAX it has registered JVP / transpose / batching rules, all of
  which are the identity, so it is also a true identity for gradients
  (forward tangents pass through unchanged; reverse-mode cotangents pass through
  unchanged). Verified on JAX 0.9.2 under ``grad``, ``jit``, ``jax.checkpoint``,
  ``vmap`` and ``jax.lax.scan``.

Because ``optimization_barrier`` already satisfies every requirement -- identity
forward, identity gradient, un-foldable by XLA -- a bespoke
``jax.extend.core.Primitive`` with hand-written impl / abstract-eval / jvp /
transpose / batching / lowering rules would be strictly more code with no
behavioural difference (a custom primitive would, at best, *also* have to lower
to ``optimization_barrier`` or a ``custom_call`` to be un-foldable). V49 is
therefore implemented as a thin, well-documented wrapper.

Public API
----------
``hlo_boundary(x)``
    Apply the boundary to an arbitrary pytree of JAX arrays. Returns a pytree
    with identical structure, shapes, dtypes and values.

``maybe_hlo_boundary(x, enabled=True)``
    Convenience gate so callers can switch the boundary on/off without
    restructuring code (used by the pipeline to keep an exact-baseline path).
"""

from typing import Any, TypeVar

import jax

__all__ = ["hlo_boundary", "maybe_hlo_boundary"]

T = TypeVar("T")


def hlo_boundary(x: T) -> T:
  """Insert a hard XLA computation boundary around ``x``.

  This is a mathematical identity: the returned pytree has the same structure,
  shapes, dtypes and values as ``x``, and gradients flow through unchanged.
  Its sole effect is on the *compiler*: XLA may not fuse, CSE, or otherwise
  move data flow across the boundary, which keeps the buffers produced before
  and after the call in separate scheduling / live-range regions.

  Implemented via ``jax.lax.optimization_barrier``, which lowers to the
  ``stablehlo.optimization_barrier`` HLO instruction. Unlike a trivial identity
  op (e.g. ``x + 0`` or ``jax.lax.copy``), an optimization barrier is *not*
  removable by any XLA optimization pass -- it is the same primitive XLA uses
  internally to fence rematerialized regions.

  Args:
    x: An arbitrary pytree of JAX arrays (e.g. a scan carry tuple). Non-array
      leaves are not supported -- pass only traced/array values.

  Returns:
    A pytree identical to ``x`` (structure, shapes, dtypes, values, gradients),
    with an HLO boundary inserted.
  """
  # jax.lax.optimization_barrier natively accepts and returns pytrees, applying
  # the barrier leaf-wise while preserving the treedef. All leaves -- including
  # integer index scalars such as ``loop_iteration`` -- pass through unchanged;
  # the barrier needs no AD rule for integer leaves and is the identity for
  # floating-point leaves under both forward- and reverse-mode AD.
  return jax.lax.optimization_barrier(x)


def maybe_hlo_boundary(x: T, enabled: bool = True) -> T:
  """Apply :func:`hlo_boundary` to ``x`` only when ``enabled`` is True.

  When ``enabled`` is False this returns ``x`` completely untouched, producing
  HLO byte-identical to the pre-V49 baseline. This lets the pipeline keep an
  exact-baseline code path behind a single flag.

  Args:
    x: An arbitrary pytree of JAX arrays.
    enabled: Whether to actually insert the boundary.

  Returns:
    ``hlo_boundary(x)`` if ``enabled`` else ``x`` unchanged.
  """
  if not enabled:
    return x
  return hlo_boundary(x)


# Backwards-friendly alias: some call sites may prefer an explicit Any signature.
_apply: Any = hlo_boundary
