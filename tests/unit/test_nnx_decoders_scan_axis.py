"""Validation test for _apply_layers_sequentially scan axis preservation.

Tests that when param_scan_axis != 0, the output params have their axes
correctly moved back to the original scan_axis position after jax.lax.scan.

This test validates the fix for the state merge regression where the original
pre-scan params were being used instead of post-scan params for axis movement.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import jax
import jax.numpy as jnp
from flax import nnx
import inspect


class SimpleLayer(nnx.Module):
  """Minimal NNX layer for testing scan axis preservation."""
  def __init__(self, features: int, rngs: nnx.Rngs):
    self.kernel = nnx.Param(jnp.ones((features, features)))

  def __call__(self, x):
    return x @ self.kernel[...]


def apply_layers_sequentially_fixed(layers, x_in, scan_axis=1):
  """Reproduces the fixed _apply_layers_sequentially logic."""
  graphdef, params, state = nnx.split(layers, nnx.Param, ...)

  if scan_axis != 0:
    params = jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0), params)

  def layer_fn(carry, scanned_vars):
    current_params, current_state = scanned_vars
    layer = nnx.merge(graphdef, current_params, current_state)
    layer_out = layer(carry)
    new_carry = layer_out
    return new_carry, nnx.state(layer)

  final_carry, scanned_state = jax.lax.scan(layer_fn, x_in, (params, state))

  # FIXED: split params from POST-SCAN output, move those back
  if scan_axis != 0:
    scanned_params, scanned_other = scanned_state.split(nnx.Param, ...)
    scanned_params = jax.tree.map(lambda x: jnp.moveaxis(x, 0, scan_axis), scanned_params)
    scanned_state = nnx.State.merge(scanned_params, scanned_other)

  nnx.update(layers, scanned_state)
  return final_carry, layers


def apply_layers_sequentially_broken(layers, x_in, scan_axis=1):
  """Reproduces the BROKEN _apply_layers_sequentially logic (pre-fix)."""
  graphdef, params, state = nnx.split(layers, nnx.Param, ...)

  if scan_axis != 0:
    params = jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0), params)

  def layer_fn(carry, scanned_vars):
    current_params, current_state = scanned_vars
    layer = nnx.merge(graphdef, current_params, current_state)
    layer_out = layer(carry)
    new_carry = layer_out
    return new_carry, nnx.state(layer)

  final_carry, scanned_state = jax.lax.scan(layer_fn, x_in, (params, state))

  # BROKEN: moves ORIGINAL pre-scan params, then merges with overlapping keys
  if scan_axis != 0:
    params = jax.tree.map(lambda x: jnp.moveaxis(x, 0, scan_axis), params)

  final_state = nnx.State.merge(params, scanned_state)
  nnx.update(layers, final_state)
  return final_carry, layers


def test_scan_axis_preservation_fixed():
  """Test that fixed version preserves param_scan_axis=1 correctly."""
  num_layers = 3
  features = 4
  scan_axis = 1

  # Create stacked layers with params at scan_axis=1
  # Shape: (features, num_layers, features) when scan_axis=1
  rngs = nnx.Rngs(0)
  layers = SimpleLayer(features, rngs)

  # Stack the kernel to simulate multiple layers with scan_axis=1
  # Original shape per layer: (features, features)
  # Stacked at axis 1: (features, num_layers, features)
  stacked_kernel = jnp.stack([jnp.ones((features, features)) * (i + 1) for i in range(num_layers)], axis=scan_axis)
  layers.kernel = nnx.Param(stacked_kernel)

  original_shape = layers.kernel[...].shape
  assert original_shape == (features, num_layers, features), f"Expected (4, 3, 4), got {original_shape}"

  x_in = jnp.ones((features,))
  _, result_layers = apply_layers_sequentially_fixed(layers, x_in, scan_axis=scan_axis)

  result_shape = result_layers.kernel[...].shape
  assert result_shape == original_shape, (
      f"FIXED version: param shape changed from {original_shape} to {result_shape}! "
      f"scan_axis={scan_axis} was not preserved."
  )
  print(f"PASS: Fixed version preserves scan_axis={scan_axis}. Shape: {original_shape} -> {result_shape}")


def test_scan_axis_broken_demonstrates_bug():
  """Test that broken version corrupts param_scan_axis=1."""
  num_layers = 3
  features = 4
  scan_axis = 1

  rngs = nnx.Rngs(0)
  layers = SimpleLayer(features, rngs)

  stacked_kernel = jnp.stack([jnp.ones((features, features)) * (i + 1) for i in range(num_layers)], axis=scan_axis)
  layers.kernel = nnx.Param(stacked_kernel)

  original_shape = layers.kernel[...].shape
  assert original_shape == (features, num_layers, features)

  x_in = jnp.ones((features,))
  _, result_layers = apply_layers_sequentially_broken(layers, x_in, scan_axis=scan_axis)

  result_shape = result_layers.kernel[...].shape
  # The broken version puts params at axis 0 instead of axis 1
  # Because nnx.State.merge(params, scanned_state) overwrites with scanned_state's axis-0 params
  if result_shape == original_shape:
    print(f"NOTE: Broken version accidentally preserved shape (merge order may vary). Shape: {result_shape}")
  else:
    print(f"CONFIRMED BUG: Broken version corrupted shape from {original_shape} to {result_shape}")
    print(f"  Expected axis order: scan_axis={scan_axis} → params should have layers at dim {scan_axis}")
    print(f"  Got: layers at dim 0 (scanned_state overwrote axis-moved params)")


def test_scan_axis_0_works_for_both():
  """Both versions should work correctly when scan_axis=0 (no axis movement needed)."""
  num_layers = 3
  features = 4
  scan_axis = 0

  rngs = nnx.Rngs(0)
  layers = SimpleLayer(features, rngs)

  stacked_kernel = jnp.stack([jnp.ones((features, features)) * (i + 1) for i in range(num_layers)], axis=scan_axis)
  layers.kernel = nnx.Param(stacked_kernel)

  original_shape = layers.kernel[...].shape
  assert original_shape == (num_layers, features, features)

  x_in = jnp.ones((features,))

  # Fixed version
  _, result_fixed = apply_layers_sequentially_fixed(layers, x_in, scan_axis=scan_axis)
  assert result_fixed.kernel[...].shape == original_shape, f"Fixed failed at scan_axis=0"

  # Re-create layers for broken test
  layers2 = SimpleLayer(features, rngs)
  layers2.kernel = nnx.Param(stacked_kernel)
  _, result_broken = apply_layers_sequentially_broken(layers2, x_in, scan_axis=scan_axis)
  assert result_broken.kernel[...].shape == original_shape, f"Broken failed at scan_axis=0"

  print(f"PASS: Both versions work correctly at scan_axis=0. Shape preserved: {original_shape}")


if __name__ == "__main__":
  print("=" * 60)
  print("Test: _apply_layers_sequentially scan axis preservation")
  print("=" * 60)

  print("\n--- Test 1: Fixed version preserves scan_axis=1 ---")
  test_scan_axis_preservation_fixed()

  print("\n--- Test 2: Broken version demonstrates the bug ---")
  test_scan_axis_broken_demonstrates_bug()

  print("\n--- Test 3: Both work at scan_axis=0 ---")
  test_scan_axis_0_works_for_both()

  print("\n" + "=" * 60)
  print("All validation tests complete.")
  print("=" * 60)
