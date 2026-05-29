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

"""NNX-native scan with variable group separation.

Replicates Linen's _partial_pack + axes_scan behavior:
1. Separates state into broadcast/carry groups
2. Makes broadcast vars explicit args to jax.remat (not closures in scan carry)
3. Wraps body in jax.remat with explicit variable inputs
4. Runs jax.lax.scan with only carry vars in the scan carry

This addresses the core memory gap between Linen nn.scan(nn.remat(body)) and
NNX's manual jax.checkpoint + jax.lax.scan: in Linen, variable_broadcast
params are explicit inputs to the remat-wrapped function (not captured as
closures), which gives XLA cleaner gradient accumulation paths. This utility
replicates that mechanism for NNX modules.
"""

import jax
from flax import nnx


def scan_with_variable_groups(
    body_fn,
    module,
    carry_init,
    xs,
    length,
    broadcast_filter,
    remat_policy=None,
):
  """Scan over body_fn with explicit variable group separation.

  Like nn.scan(nn.remat(body_fn)), but for NNX modules.

  The key insight: in Linen's nn.scan + nn.remat, broadcast variables (params)
  are passed as EXPLICIT arguments to the remat-wrapped function. They are NOT
  in the scan carry (which would save all intermediate values for backward),
  and they are NOT closures TO remat (which would prevent XLA from seeing their
  gradient paths clearly).

  This function replicates that pattern:
  1. Split module state into broadcast group (e.g. params) and carry group (rest)
  2. Create a pure function where broadcast vars are the FIRST explicit arg
  3. Wrap that pure function in jax.remat
  4. In the scan body, broadcast vars are closed over the SCAN (constant across
     iterations), but are explicit inputs to REMAT

  Args:
    body_fn: function (module, carry_data, xs_i) -> (carry_data_out, ys_i)
        The per-iteration body. Receives the full module (with all state),
        the carry data (e.g. loop_state), and the current scan input slice.
        Returns updated carry data and scan outputs for this iteration.
    module: NNX module whose state is split into groups. The module's state
        is partitioned by broadcast_filter into:
        - broadcast group: constant across scan iterations (like params)
        - carry group: threaded through scan as carry (like RNG state)
    carry_init: initial carry data for the scan (e.g. loop_state dict).
        This is the non-module-state carry that threads through iterations.
    xs: scan inputs stacked along axis 0 (or None for no per-step input).
    length: number of scan iterations.
    broadcast_filter: NNX filter for broadcast variables (e.g., nnx.Param).
        Variables matching this filter are closed over the scan and passed
        as explicit args to remat. Their gradients accumulate as scan
        constants (analogous to Linen variable_broadcast).
    remat_policy: optional checkpoint policy for jax.remat. If None, no
        rematerialization is applied. Typically a policy like
        jax.checkpoint_policies.save_only_these_names(...).

  Returns:
    (final_carry_data, ys) where:
      - final_carry_data: the carry data after the last iteration
      - ys: stacked scan outputs (one per iteration)

  Side effects:
    Updates module in-place with the final broadcast + carry state
    (broadcast state is unchanged; carry state reflects the last iteration).
  """
  # Step 1: Split module into broadcast and carry state groups.
  # broadcast_state: params (constant across iterations, gradient via AD)
  # carry_module_state: everything else (RNG state, mutables, etc.)
  graph, broadcast_state, carry_module_state = nnx.split(
      module, broadcast_filter, ...
  )

  # Step 2: Create a pure body where broadcast vars are an EXPLICIT first arg.
  # This is the key to matching Linen's _partial_pack: the remat wrapper sees
  # broadcast_vars as a proper function input, not a closure capture.
  def remat_body(broadcast_vars, carry_module_vars, carry_data, xs_i):
    """Pure function: explicit broadcast vars + carry vars -> outputs.

    Args:
      broadcast_vars: nnx.State containing broadcast variables (e.g. params).
          These are constant across scan iterations but explicit here so
          jax.remat can track their gradient paths.
      carry_module_vars: nnx.State containing carry variables (e.g. RNG state).
          These change each iteration and are threaded through scan carry.
      carry_data: the non-module carry (e.g. loop_state dict).
      xs_i: current scan input slice (or None).

    Returns:
      (carry_data_out, new_carry_module_vars, ys_i)
    """
    # Merge broadcast + carry into full state and reconstruct module
    full_state = nnx.State.merge(broadcast_vars, carry_module_vars)
    m = nnx.merge(graph, full_state)

    # Run the user's body function with the reconstructed module
    carry_data_out, ys_i = body_fn(m, carry_data, xs_i)

    # Re-split to extract updated carry state (broadcast state is unchanged
    # because params don't mutate during forward -- gradients flow via AD)
    _, _, new_carry_module_vars = nnx.split(m, broadcast_filter, ...)

    return carry_data_out, new_carry_module_vars, ys_i

  # Step 3: Wrap in jax.remat -- broadcast_vars is an EXPLICIT first arg,
  # not a closure. This gives XLA the same visibility into gradient paths
  # that Linen's nn.remat provides via _partial_pack.
  if remat_policy is not None:
    remat_body = jax.remat(remat_body, policy=remat_policy)

  # Step 4: Scan body -- broadcast state is CLOSED OVER the scan (constant),
  # while carry_module_vars and carry_data thread through as scan carry.
  def scan_body(carry, xs_i):
    carry_module_vars, carry_data = carry

    # broadcast_state is captured from enclosing scope -- NOT in carry.
    # This means scan does not save per-iteration copies of params for
    # backward. Instead, params are scan constants with gradient
    # accumulation handled by AD (matching Linen variable_broadcast).
    carry_data_out, new_carry_module_vars, ys_i = remat_body(
        broadcast_state,  # CLOSED OVER -- not in carry
        carry_module_vars,
        carry_data,
        xs_i,
    )
    return (new_carry_module_vars, carry_data_out), ys_i

  # Step 5: Run jax.lax.scan
  (final_carry_module_vars, final_carry_data), ys = jax.lax.scan(
      scan_body,
      (carry_module_state, carry_init),
      xs,
      length=length,
  )

  # Step 6: Update module with final state.
  # broadcast_state is unchanged (params are static in scan).
  # carry_module_state is updated to the final iteration's values.
  final_state = nnx.State.merge(broadcast_state, final_carry_module_vars)
  nnx.update(module, nnx.merge(graph, final_state))

  return final_carry_data, ys
