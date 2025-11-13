# Copyright 2023–2025 Google LLC
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


r"""Layerwise quantization for large models

Provides a utility to load and quantize a checkpoint layer by layer. Currently, it supports DeepSeek-family models only.

Example cmd:

python3 -m MaxText.layerwise_quantization  src/MaxText/configs/base.yml \
  tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${LOAD_PARAMS_PATH} \
  model_name=deepseek2-16b ici_fsdp_parallelism=1 ici_autoregressive_parallelism=1 \
  ici_tensor_parallelism=-1 scan_layers=false weight_dtype=bfloat16 per_device_batch_size=1 \
  attention=dot_product quantization=int8 async_checkpointing=false enable_single_controller=true \
  tokenizer_type=huggingface megablox=false sparse_matmul=false \
  save_quantized_params_path=${SAVE_PARAMS_PATH} checkpoint_storage_use_ocdbt=False \
  checkpoint_storage_use_zarr3=False

"""

import os
from typing import Any, Sequence

from tqdm import tqdm
import jax
import jax.numpy as jnp
from absl import app

from flax.linen import partitioning as nn_partitioning
from flax import nnx

from MaxText import checkpointing
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText import common_types
from MaxText.layers import models, quantizations, deepseek
import orbax.checkpoint as ocp


IGNORE = ocp.PLACEHOLDER
PRNGKeyType = Any


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


def validate_loaded_params(nnx_model: nnx.Module, loaded_params: dict[str, Any]):
  """Compares loaded_params structure and shapes/dtypes with the nnx_model's expected nnx.Params."""
  expected_state = nnx.state(nnx_model, nnx.Param)

  expected_leaves = {
      jax.tree_util.keystr(path): leaf for path, leaf in jax.tree_util.tree_leaves_with_path(expected_state)
  }
  loaded_leaves = {
      jax.tree_util.keystr(path): leaf
      for path, leaf in jax.tree_util.tree_leaves_with_path(loaded_params)
      if isinstance(leaf, (jax.Array, jnp.ndarray))
  }

  expected_paths = set(expected_leaves.keys())
  loaded_paths = set(loaded_leaves.keys())

  missing_from_loaded = expected_paths - loaded_paths
  if missing_from_loaded:
    print(f"ERROR: Parameters expected by model but not found in loaded_params: {missing_from_loaded}")

  extra_in_loaded = loaded_paths - expected_paths
  if extra_in_loaded:
    print(f"WARNING: Parameters found in loaded_params but not expected as nnx.Param in model: {extra_in_loaded}")

  common_paths = expected_paths.intersection(loaded_paths)
  for path_str in common_paths:
    expected_leaf = expected_leaves[path_str]
    loaded_leaf = loaded_leaves[path_str]

    if not isinstance(expected_leaf, (nnx.Param, nnx.Variable)):
      print(f"INTERNAL WARNING: Expected leaf at {path_str} is not nnx.Param/Variable: {type(expected_leaf)}")
      continue

    expected_shape = expected_leaf.value.shape
    loaded_shape = loaded_leaf.shape
    if expected_shape != loaded_shape:
      print(f"WARNING: Shape mismatch at {path_str}: " f"Model expects {expected_shape}, loaded has {loaded_shape}")

    expected_dtype = expected_leaf.value.dtype
    loaded_dtype = loaded_leaf.dtype
    if expected_dtype != loaded_dtype:
      print(f"WARNING: Dtype mismatch at {path_str}: " f"Model expects {expected_dtype}, loaded has {loaded_dtype}")
  print("Validation complete.")


class LayerwiseQuantization:
  """
  Layerwise quantization for large models.
  """

  def __init__(self, config: Any):
    self.config = config

    # TODO(ranlihao): Remove this assertion once the Layerwise quantization is supported for other decoder blocks.
    assert (
        config.decoder_block == common_types.DecoderBlockType.DEEPSEEK
    ), f"Layerwise quantization is only supported for {common_types.DecoderBlockType.DEEPSEEK}\
      , but got {config.decoder_block}."

    # Mesh definition
    devices_array = maxtext_utils.create_device_mesh(config=config)
    self._mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

    # Model and quantization config
    self.quant = quantizations.configure_quantization(config)
    model = models.transformer_as_linen(
        config, mesh=self._mesh, quant=self.quant, model_mode=common_types.MODEL_MODE_TRAIN
    )
    rng = jax.random.PRNGKey(1234)
    self.unboxed_abstract_state, _, _ = maxtext_utils.get_abstract_state(model, None, self.config, rng, self._mesh, False)

  def load_and_quantize(self, layer_rngs: nnx.Rngs) -> None:
    """
    Load parameters layer by layer and quantize them.
    """

    quantized_params = {}
    quantized_params["params"] = {"decoder": {}}
    quantized_params["aqt"] = {"decoder": {}}

    config = self.config

    self.quant.quant_mode = quantizations.get_quant_mode("convert")

    model_mode = common_types.MODEL_MODE_PREFILL

    # Layer configurations
    layer_configs = [
        ("moe_layers", config.num_decoder_layers - config.first_num_dense_layers, deepseek.DeepSeekMoELayer),
        ("dense_layers", config.first_num_dense_layers, deepseek.DeepSeekDenseLayer),
    ]

    # Prepare dummy inputs for quantization
    dummy_inputs = jnp.ones(
        (1, self.config.max_prefill_predict_length, self.config.base_emb_dim), dtype=self.config.dtype
    )
    dummy_decoder_segment_ids = jnp.zeros((1, self.config.max_prefill_predict_length), dtype=jnp.int32)
    dummy_positions = None

    for layer_prefix, num_layers, layer_class in layer_configs:
      for index in tqdm(range(num_layers)):
        print(f"Quantizing layer {layer_prefix}_{index}")

        layer_name = f"{layer_prefix}_{index}"
        print(f"\nDEBUG: --- Processing layer: {layer_name} ---")

        # Create a new layer instance (NNX modules are stateful)
        # Note: Don't pass quant to layer creation to avoid unbound Linen module errors
        # The quantization will be handled separately in the layerwise approach
        layer = layer_class(config=config, mesh=self._mesh, quant=None, model_mode=model_mode, rngs=layer_rngs)

        # Call the layer directly (NNX style) - this will quantize the weights
        _ = layer(
            inputs=dummy_inputs,
            decoder_segment_ids=dummy_decoder_segment_ids,
            decoder_positions=dummy_positions,
            deterministic=True,
            model_mode=model_mode,
        )

        # Convert NNX state to a format compatible with the checkpoint saving
        # The state_dict contains both Param and other Variable types
        # We need to separate params from AQT quantization state

        # Extract params (excluding AQT-related variables)
        params_only = nnx.state(layer, nnx.Param)

        print(f"DEBUG: Loading params for {layer_name}...")

        # Load checkpoint params
        params = self._load_layer(layer_name)
        layer_params = params["params"]["decoder"][layer_name]

        print(f"DEBUG: Loaded params shapes for {layer_name}:")
        jax.tree_util.tree_map_with_path(
            lambda path, x: print(f"  {jax.tree_util.keystr(path)}: {x.shape}"), layer_params
        )

        validate_loaded_params(layer, layer_params)

        # Load the checkpoint weights into the NNX module
        load_weights_into_deepseek_layer(layer, layer_params)

        # For now, extract the full params
        # The quantizations.remove_quantized_params function will handle filtering
        layer_params_dict = jax.tree_util.tree_map(lambda x: x.value if hasattr(x, "value") else x, params_only)

        # Try to extract AQT state if it exists
        # AQT state might be stored in a separate collection or as specific attributes
        aqt_state = {}
        if hasattr(layer, "aqt"):
          aqt_state = nnx.state(layer.aqt) if isinstance(layer.aqt, nnx.Module) else {}

        # Use the existing helper to remove quantized params and get clean params
        quantized_layer_params = quantizations.remove_quantized_params(layer_params_dict, aqt_state)

        quantized_params["aqt"]["decoder"][layer_name] = aqt_state
        quantized_params["params"]["decoder"][layer_name] = quantized_layer_params

    # Load and save the layers that should not be quantized.
    unquantized_layers = ["decoder_norm", "logits_dense"]
    for unquantized_layer in unquantized_layers:
      params = self._load_layer(unquantized_layer)
      quantized_params["params"]["decoder"][unquantized_layer] = params["params"]["decoder"][unquantized_layer]

    quantized_params["params"]["token_embedder"] = self._load_layer("token_embedder")["params"]["token_embedder"]

    maxtext_utils.save_quantized_checkpoint_if_configured(self.config, quantized_params)

  def _load_layer(self, layer_name):
    """Loads a specific layer's parameters from the checkpoint."""

    config = self.config
    with nn_partitioning.axis_rules(config.logical_axis_rules):
      params = checkpointing.load_params_from_path(
          config.load_parameters_path,
          self._create_partial_abstract_params(self.unboxed_abstract_state.params, layer_name),
          config.checkpoint_storage_concurrent_gb,
          config.checkpoint_storage_use_ocdbt,
          config.checkpoint_storage_use_zarr3,
      )
    return params

  def _create_partial_abstract_params(self, abstract_unboxed_params, layer):
    """Creates a partial abstract params structure using ocp.PLACEHOLDER.

    Args:
        abstract_params: The full abstract params structure (e.g., from a TrainState).
        layer: The layer name to keep in the abstract params.

    Returns:
        A new abstract params structure with ocp.PLACEHOLDER for skipped nodes.
    """

    def _should_keep(path, _):
      # True if the layer name is part of the path
      return any(isinstance(key, jax.tree_util.DictKey) and key.key == layer for key in path)

    def _map_fn(path, value):
      if not _should_keep(path, value):
        return IGNORE
      if isinstance(value, jax.ShapeDtypeStruct):
        zeros_array = jnp.zeros(value.shape, value.dtype)
        if value.sharding is not None:
          try:
            return jax.device_put(zeros_array, value.sharding)
          except Exception as e:
            print(f"Error applying sharding for path {path}: {e}")
            return zeros_array
        return zeros_array
      return value

    return jax.tree_util.tree_map_with_path(_map_fn, abstract_unboxed_params)


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  config = pyconfig.initialize(argv)
  validate_config(config)
  max_utils.print_system_information()

  quantization = LayerwiseQuantization(config)
  rng = nnx.Rngs(jax.random.PRNGKey(1234))

  # load_and_quantize will load a checkpoint and quantize if the following parameters are set:
  # quantization=$valid_quantization_type \
  # save_quantized_params_path=$gsbucket_path \
  # checkpoint_is_quantized=false (default)
  quantization.load_and_quantize(rng)


def validate_config(config):
  assert (
      config.load_full_state_path == ""
  ), "Operation on full states not supported! Convert to parameter checkpoint first."


if __name__ == "__main__":
  app.run(main)
