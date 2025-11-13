from flax import nnx

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec

from MaxText.common_types import Config
from MaxText.layers import deepseek
from MaxText.layers.models import Transformer

from typing import Any, Tuple


# --- New function for NNX Models ---
def get_abstract_state_nnx(model: Transformer, config: Config, mesh: Mesh) -> Tuple[Any, Any, Any]:
  """Get a shaped abstraction of the state for an NNX model (inference mode)."""

  # 1. Get the full state tree from the model
  state_tree = nnx.state(model)

  # 2. Function to extract PartitionSpec from NNX Variable metadata
  def extract_pspec(variable: Any) -> PartitionSpec:
    if isinstance(variable, nnx.Variable):
      sharding = variable.sharding
      if isinstance(sharding, PartitionSpec):
        return sharding
    return PartitionSpec()  # Default to Replicated

  # 3. Map over the state tree to get PartitionSpec for each leaf
  state_logical_annotations = jax.tree_util.tree_map(
      extract_pspec, state_tree, is_leaf=lambda x: isinstance(x, nnx.Variable)
  )

  # 4. Convert PartitionSpecs to NamedShardings using the mesh
  state_mesh_shardings = jax.tree_util.tree_map(lambda pspec: NamedSharding(mesh, pspec), state_logical_annotations)

  # 5. Parameter host offload for Model Parameters
  if config.parameter_memory_host_offload:
    assert config.param_scan_axis == 0, "You must set the scan axis 0 to enable parameter offloading."

    def move_to_host(sharding: NamedSharding) -> NamedSharding:
      # print(f"NNX: Applying pinned_host memory kind to {jax.tree_util.keystr(path)}")
      return sharding.with_memory_kind(kind="pinned_host")

    state_mesh_shardings = jax.tree_util.tree_map_with_path(
        move_to_host, state_mesh_shardings, is_leaf=lambda x: isinstance(x, NamedSharding)
    )

  # Get the abstract values (shapes and dtypes) from the state tree
  abstract_state_tree = jax.eval_shape(lambda: state_tree)

  # 6. Create abstract sharded state with ShapeDtypeStruct
  def create_sharded_aval(tensor_shape: jax.ShapeDtypeStruct, sharding: NamedSharding) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(tensor_shape.shape, tensor_shape.dtype, sharding=sharding)

  abstract_sharded_state = jax.tree_util.tree_map(create_sharded_aval, abstract_state_tree, state_mesh_shardings)

  return (abstract_sharded_state, state_logical_annotations, state_mesh_shardings)


def set_nnx_param(model: nnx.Module, path: tuple, value: jax.Array):
  module = model
  keys = []
  for p in path:
    if isinstance(p, jax.tree_util.DictKey):
      keys.append(p.key)
    elif isinstance(p, jax.tree_util.SequenceKey):
      keys.append(p.idx)
    elif isinstance(p, jax.tree_util.GetAttrKey):
      keys.append(p.name)
    else:
      raise TypeError(f"Unsupported path key type: {type(p)} in {jax.tree_util.keystr(path)}")

  current_path_str = "model"
  for _, key in enumerate(keys[:-1]):
    key_str = str(key)
    if isinstance(module, nnx.Dict) and key in module:
      module = module[key]
      current_path_str += f"['{key_str}']"
    elif hasattr(module, key_str):
      module = getattr(module, key_str)
      current_path_str += f".{key_str}"
    else:
      raise AttributeError(
          f"Module {type(module).__name__} at path '{current_path_str}' has no attribute or key '{key_str}'. Path: {jax.tree_util.keystr(path)}"
      )

  param_name = str(keys[-1])
  if not hasattr(module, param_name):
    raise AttributeError(
        f"Module {type(module).__name__} at path '{current_path_str}' has no attribute '{param_name}'. Path: {jax.tree_util.keystr(path)}"
    )

  param_attr = getattr(module, param_name)

  if not isinstance(param_attr, (nnx.Param, nnx.Variable)):
    raise TypeError(
        f"Attribute '{param_name}' at path {current_path_str}.{param_name} is not an nnx.Param or nnx.Variable, got {type(param_attr)}"
    )
  if param_attr.value.shape != value.shape:
    print(
        f"Warning: Shape mismatch for {jax.tree_util.keystr(path)}: NNX has {param_attr.value.shape}, loading {value.shape}"
    )
  param_attr.value = value


def load_weights_into_deepseek_layer(
    nnx_model: deepseek.DeepSeekMoELayer | deepseek.DeepSeekDenseLayer, loaded_params: dict[str, Any]
):
  """
  Loads weights from a Linen-style parameter dictionary into deepseek nnx layer.

  Args:
      nnx_model: An instance of the DeepSeekMoELayer or DeepSeekDenseLayer.
      loaded_params: A nested dictionary containing the weights, matching the
                     structure expected by the nnx_model's attributes.
                     This should be the part of the checkpoint corresponding to 'params'.
  """
  print("Starting weight loading process...")

  def _load_leaf(path, leaf_array):
    if not isinstance(leaf_array, (jax.Array, jnp.ndarray)):
      return

    try:
      set_nnx_param(nnx_model, path, leaf_array)
    except (AttributeError, TypeError, KeyError) as e:
      print(f"Error loading {jax.tree_util.keystr(path)}: {e}")

  jax.tree_util.tree_map_with_path(_load_leaf, loaded_params)
  print("Weight loading process finished.")


def pytree_has_arrays(tree: Any) -> bool:
  """Checks if any leaf in the PyTree is a JAX or NumPy array."""
  found = False
  for x in jax.tree_util.tree_leaves(tree):
    if isinstance(x, (jax.Array, jnp.ndarray)):
      found = True
      break
  return found


def validate_loaded_params(nnx_model: nnx.Module, loaded_params: dict[str, Any]):
  print("--- Validating if loaded_params can be applied to nnx_model ---")
  if not loaded_params:
    print("ERROR: loaded_params dictionary is empty.")
    print("--- Validation complete ---")
    return

  has_loaded_arrays = pytree_has_arrays(loaded_params)
  if not has_loaded_arrays:
    print("WARNING: No jax.Array or jnp.ndarray found in loaded_params.")

  expected_state = nnx.state(nnx_model, nnx.Param)
  model_has_params = bool(expected_state) and bool(jax.tree_util.tree_leaves(expected_state))

  if not model_has_params:
    print("WARNING: NNX model appears to have no nnx.Param attributes.")

  if not has_loaded_arrays and not model_has_params:
    print("INFO: Both loaded_params (arrays) and model (nnx.Params) are empty. Validation is trivial.")
    print("--- Validation complete ---")
    return
  elif not has_loaded_arrays:
    print("ERROR: loaded_params has no arrays, but model expects params.")
    print("--- Validation complete ---")
    return
  elif not model_has_params:
    print("ERROR: Model has no nnx.Param attributes, but loaded_params contains arrays.")
    print("--- Validation complete ---")
    return

  has_errors = False
  loaded_array_paths = set()

  def check_leaf(path, loaded_array):
    nonlocal has_errors
    if not isinstance(loaded_array, (jax.Array, jnp.ndarray)):
      return  # Skip non-arrays

    path_str = jax.tree_util.keystr(path)
    loaded_array_paths.add(path_str)
    module = nnx_model
    keys = []
    try:
      # Recreate path parts from jax.tree_util.PathKey
      for p in path:
        if isinstance(p, jax.tree_util.DictKey):
          keys.append(p.key)
        elif isinstance(p, jax.tree_util.SequenceKey):
          keys.append(p.idx)
        elif isinstance(p, jax.tree_util.GetAttrKey):
          keys.append(p.name)
        else:
          raise TypeError(f"Unsupported path key type: {type(p)}")

      current_path_str = "model"
      # Navigate to the parent module
      for key in keys[:-1]:
        key_str = str(key)
        if isinstance(module, nnx.Dict) and key in module:
          module = module[key]
        elif hasattr(module, key_str):
          module = getattr(module, key_str)
        else:
          raise AttributeError(f"Module at path '{current_path_str}' has no attribute/key '{key_str}'")
        current_path_str += f"['{key_str}']" if isinstance(key, (int, str)) else f".{key_str}"

      param_name = str(keys[-1])
      if not hasattr(module, param_name):
        raise AttributeError(f"Module at path '{current_path_str}' has no attribute '{param_name}'")

      param_attr = getattr(module, param_name)

      if not isinstance(param_attr, nnx.Param):
        print(f"ERROR: Path {path_str}: Attribute '{param_name}' is not an nnx.Param, got {type(param_attr).__name__}")
        has_errors = True
        return

      expected_array = param_attr.value
      if expected_array.shape != loaded_array.shape:
        print(f"  WARNING: Path {path_str}: Shape mismatch. Model: {expected_array.shape}, Loaded: {loaded_array.shape}")
      if expected_array.dtype != loaded_array.dtype:
        print(f"  WARNING: Path {path_str}: Dtype mismatch. Model: {expected_array.dtype}, Loaded: {loaded_array.dtype}")

    except Exception as e:
      print(f"ERROR: Path {path_str}: Cannot access/validate in nnx_model: {e}")
      has_errors = True

  jax.tree_util.tree_map_with_path(check_leaf, loaded_params)

  # Check for params in model not present in loaded_params
  model_param_paths = set()
  for path, leaf in jax.tree_util.tree_leaves_with_path(expected_state):
    if isinstance(leaf, nnx.Param):
      # Path to the nnx.Param object itself
      model_param_paths.add(jax.tree_util.keystr(path))
    elif isinstance(leaf, (jax.Array, jnp.ndarray)):
      # This case happens if jax.tree_util descends into nnx.Param
      # We need to remove the '.value' part from the path
      if path and isinstance(path[-1], jax.tree_util.GetAttrKey) and path[-1].name == "value":
        model_param_paths.add(jax.tree_util.keystr(path[:-1]))
      else:
        # Should not happen if leaf is an array from nnx.state(..., nnx.Param)
        print(f"UNEXPECTED: Array leaf at {jax.tree_util.keystr(path)} not from a .value attribute")

  missing_from_loaded = model_param_paths - loaded_array_paths
  if missing_from_loaded:
    print(f"\nWARNING: nnx.Param paths in model not found in loaded_params arrays: {sorted(list(missing_from_loaded))}")

  if not has_errors:
    print("\nSUCCESS: loaded_params structure seems compatible with nnx_model for assignment.")
  else:
    print("\nValidation finished with potential issues.")
  print("--- Validation complete ---")


def validate_post_load(nnx_model: nnx.Module, loaded_params: dict[str, Any], rtol=1e-6, atol=1e-6):
  """
  Validates that the nnx.Param values in nnx_model match the arrays in loaded_params.
  Call this *after* loading weights into the nnx_model.
  """
  print("--- Validating NNX Model State Against Loaded Params Dict ---")
  if not loaded_params:
    print("ERROR: loaded_params dictionary is empty.")
    print("--- Validation complete ---")
    return

  has_loaded_arrays = pytree_has_arrays(loaded_params)
  if not has_loaded_arrays:
    print("WARNING: No jax.Array or jnp.ndarray found in loaded_params.")

  expected_state = nnx.state(nnx_model, nnx.Param)
  model_has_params = bool(expected_state) and bool(jax.tree_util.tree_leaves(expected_state))

  if not model_has_params:
    print("WARNING: NNX model appears to have no nnx.Param attributes.")

  if not has_loaded_arrays and not model_has_params:
    print("INFO: Both loaded_params (arrays) and model (nnx.Params) are empty. Validation is trivial.")
    print("--- Validation complete ---")
    return
  elif not has_loaded_arrays:
    print("ERROR: loaded_params has no arrays, but model expects params.")
    print("--- Validation complete ---")
    return
  elif not model_has_params:
    print("ERROR: Model has no nnx.Param attributes, but loaded_params contains arrays.")
    print("--- Validation complete ---")
    return

  has_errors = False
  has_warnings = False
  loaded_array_paths = set()

  def check_leaf(path, loaded_array):
    nonlocal has_errors, has_warnings
    if not isinstance(loaded_array, (jax.Array, jnp.ndarray)):
      return  # Skip non-arrays

    path_str = jax.tree_util.keystr(path)
    loaded_array_paths.add(path_str)
    module = nnx_model
    keys = []
    try:
      # Build keys list from path
      for p in path:
        if isinstance(p, jax.tree_util.DictKey):
          keys.append(p.key)
        elif isinstance(p, jax.tree_util.SequenceKey):
          keys.append(p.idx)
        elif isinstance(p, jax.tree_util.GetAttrKey):
          keys.append(p.name)
        else:
          raise TypeError(f"Unsupported path key type: {type(p)}")

      # Navigate to the parent module
      current_path_str = "model"
      for key in keys[:-1]:
        key_str = str(key)
        if isinstance(module, nnx.Dict) and key in module:
          module = module[key]
        elif hasattr(module, key_str):
          module = getattr(module, key_str)
        else:
          raise AttributeError(f"Module at '{current_path_str}' has no attribute/key '{key_str}'")
        current_path_str += f"['{key_str}']" if isinstance(key, (int, str)) else f".{key_str}"

      param_name = str(keys[-1])
      if not hasattr(module, param_name):
        raise AttributeError(f"Module at '{current_path_str}' has no attribute '{param_name}'")

      param_attr = getattr(module, param_name)

      if not isinstance(param_attr, nnx.Param):
        print(
            f"ERROR: Path {path_str}: Attribute '{param_name}' in model is not an nnx.Param, got {type(param_attr).__name__}"
        )
        has_errors = True
        return

      model_array = param_attr.value  # This is the array *after* assignment

      # Shape Check
      if model_array.shape != loaded_array.shape:
        print(f"  ERROR: Path {path_str}: Shape mismatch. Model: {model_array.shape}, Loaded: {loaded_array.shape}.")
        has_errors = True
        return
      # Dtype Check
      if model_array.dtype != loaded_array.dtype:
        print(
            f"  WARNING: Path {path_str}: Dtype mismatch. Model: {model_array.dtype}, Loaded: {loaded_array.dtype}. Assignment might involve cast."
        )
        has_warnings = True

      # Numerical Value Validation
      if jnp.array_equal(model_array, loaded_array):
        pass
      elif jnp.allclose(model_array, loaded_array, rtol=rtol, atol=atol):
        print(f"  OK: Path {path_str}: Weights allclose (rtol={rtol}, atol={atol}).")
      else:
        diff = jnp.abs(model_array - loaded_array)
        print(f"  ERROR: Path {path_str}: Numerical difference detected between model array and loaded array!")
        print(f"    Max absolute difference: {jnp.max(diff)}")
        print(f"    Mean absolute difference: {jnp.mean(diff)}")
        has_errors = True

    except Exception as e:
      print(f"ERROR: Path {path_str}: Exception during validation: {e}")
      has_errors = True

  jax.tree_util.tree_map_with_path(check_leaf, loaded_params)
  print(f"--- Finished comparing {len(loaded_array_paths)} array paths from loaded_params. ---")

  # Check for any nnx.Param in the model that WASN'T in loaded_params
  model_param_paths = set()
  for path, leaf in jax.tree_util.tree_leaves_with_path(expected_state):
    if not isinstance(leaf, nnx.Param):
      continue
    model_param_paths.add(jax.tree_util.keystr(path))

  missing_from_loaded = model_param_paths - loaded_array_paths
  if missing_from_loaded:
    print(f"\nWARNING: nnx.Param paths in model not found in loaded_params arrays: {sorted(list(missing_from_loaded))}")
    has_warnings = True

  if not has_errors and not has_warnings:
    print("\nSUCCESS: NNX model weights are consistent with the loaded_params dictionary.")
  elif not has_errors:
    print("\nValidation finished with warnings.")
  else:
    print("\nValidation finished with ERRORS.")
  print("--- Validation complete ---")
