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
from aqt.jax.v2 import aqt_tensor

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
DictKey = jax.tree_util.DictKey


def match_aqt_and_unquantized_param(aqt_params, params):
  """
  Matches AQT quantized components to original parameter paths.

  Returns:
      A PyTree structured like the modules containing AqtDotGeneral_,
      where the leaves are the tuple paths to the corresponding original
      'kernel' parameters in the params PyTree.
  """
  aqt_flat_map = nnx.traversals.flatten_mapping(aqt_params, sep='/')
  params_flat_map = nnx.traversals.flatten_mapping(params, sep='/')

  # This dictionary will map the string path of a module
  # (e.g., 'self_attention/out') to the full JAX tuple path
  # of the original kernel (e.g., (DictKey('self_attention'), DictKey('out'), DictKey('kernel'))).
  module_to_param_path = {}

  for path_str in aqt_flat_map.keys():
      if 'AqtDotGeneral_' not in path_str or path_str.startswith('AqtEinsum_'):
          continue

      path = path_str.split('/')
      try:
          # Find the part of the path representing the module,
          # right before the 'AqtDotGeneral_' part.
          aqt_dot_general_index = next(i for i, part in enumerate(path) if 'AqtDotGeneral_' in part)
          module_path_list = path[:aqt_dot_general_index]
          module_key = '/'.join(module_path_list)

          # We only need to find the kernel path once per module
          if module_key in module_to_param_path:
              continue

          # Construct the expected path to the original kernel
          original_param_path_list = module_path_list + ['kernel']
          original_param_key = '/'.join(original_param_path_list)

          if original_param_key in params_flat_map:
              # Create the JAX path tuple
              param_tuple_path = tuple(jax.tree_util.DictKey(key=k) for k in original_param_path_list)
              module_to_param_path[module_key] = param_tuple_path
          else:
              print(f"Kernel not found for AQT module: {module_key} (Expected: {original_param_key})")

      except StopIteration:
          # Should not happen given the check above
          continue

  if not module_to_param_path:
      print("No parameters found to be quantized by AqtDotGeneral.")
      return {}

  # Unflatten the map to create a PyTree. The leaves of this tree
  # are the tuple paths to the original parameters.
  return  nnx.traversals.unflatten_mapping(module_to_param_path, sep='/')

def _get_aqt_key_paths(aqt_vars, params):
  """Generate a list of paths which have aqt state"""
  aqt_to_unquantized_key_path = match_aqt_and_unquantized_param(aqt_vars, params)
  if not aqt_to_unquantized_key_path:
      return []
  aqt_key_paths, _ = jax.tree_util.tree_flatten(aqt_to_unquantized_key_path, is_leaf=lambda x: isinstance(x, tuple))
  return list(aqt_key_paths)


def remove_quantized_params(params, aqt_vars):
  """Remove param values with aqt tensors to Null to optimize memory."""
  quantized_param_paths = _get_aqt_key_paths(aqt_vars, params)
  if not quantized_param_paths:
      print("No parameters to remove.")
      return params

  print(f"Attempting to remove {len(quantized_param_paths)} quantized parameter paths.")
  tree_flat, tree_struct = jax.tree_util.tree_flatten_with_path(params)

  new_tree_flat = []
  removed_count = 0
  for k_path, v in tree_flat:
    if k_path in quantized_param_paths:
      new_tree_flat.append({})  # Replace with empty dict
      removed_count += 1
    else:
      new_tree_flat.append(v)

  if removed_count != len(quantized_param_paths):
      print(f"Warning: Expected to remove {len(quantized_param_paths)} but only removed {removed_count}")

  print(f"Successfully marked {removed_count} parameters for removal.")
  return jax.tree_util.tree_unflatten(tree_struct, new_tree_flat)

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

  def load_and_quantize(self, rng: PRNGKeyType) -> None:
    """
    Load parameters layer by layer and quantize them.
    """
    quantized_params = {}
    quantized_params["params"] = {"decoder": {}}
    quantized_params["aqt"] = {"decoder": {}}
    config = self.config
    self.quant.quant_mode = quantizations.get_quant_mode("convert")
    model_mode = common_types.MODEL_MODE_PREFILL
    _, rng_quant_params = jax.random.split(rng)

    layers = [
       deepseek.DeepSeekMoELayerToLinen(
            config=config, mesh=self._mesh, quant=self.quant, model_mode=model_mode, rngs=nnx.Rngs(rng)
        ),
        deepseek.DeepSeekDenseLayerToLinen(
            config=config, mesh=self._mesh, quant=self.quant, model_mode=model_mode, rngs=nnx.Rngs(rng)
        ),
    ]
    layer_prefixes = ["moe_layers", "dense_layers"]
    num_moe_layers = config.num_decoder_layers - config.first_num_dense_layers
    num_layers_list = [num_moe_layers, config.first_num_dense_layers]

    def model_apply(_p, _rng, layer):
      return layer.apply(
          _p | {"aqt": {}},
          jnp.ones((1, self.config.max_prefill_predict_length, self.config.base_emb_dim), dtype=jnp.int32),
          None,
          jnp.zeros((1, self.config.max_prefill_predict_length), dtype=jnp.int32),
          True,
          model_mode=model_mode,
          rngs={"params": _rng},
          mutable=True,
      )

    for layer, num_layers, layer_prefix in zip(layers, num_layers_list, layer_prefixes):
      for index in tqdm(range(num_layers)):
        layer_name = f"{layer_prefix}_{index}"
        params = self._load_layer(layer_name)
        params["params"] = params["params"]["decoder"][layer_name]

        _, new_vars = model_apply(params, rng_quant_params, layer)

        if "aqt" not in new_vars:
          print(f"Warning: 'aqt' not found in new_vars for {layer_name}. Skipping AQT processing for this layer.")
          quantized_params["params"]["decoder"][layer_name] = params["params"]  # Keep original params
          continue

        quantized_params["aqt"]["decoder"][layer_name] = new_vars["aqt"]

        try:
          removed_params = remove_quantized_params(params["params"], new_vars["aqt"])
          quantized_params["params"]["decoder"][layer_name] = removed_params
        except Exception as e:
          print(f"ERROR: Failed to remove quantized params for {layer_name}: {e}")
          print(f"DEBUG: Dumping params['params'] keys for {layer_name}:")
          jax.tree_util.tree_map_with_path(lambda path, _: print(f"  {jax.tree_util.keystr(path)}"), params["params"])
          print(f"DEBUG: Dumping new_vars['aqt'] keys for {layer_name}:")
          jax.tree_util.tree_map_with_path(lambda path, _: print(f"  {jax.tree_util.keystr(path)}"), new_vars["aqt"])
          # Optional: re-raise the exception if you want to stop execution
          raise

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

      # print(f"DEBUG: Abstract params for {layer_name}: {jax.tree_map(lambda x: x.shape, partial_abstract_params)}")
      params = checkpointing.load_params_from_path(
          config.load_parameters_path,
          self._create_partial_abstract_params(self.unboxed_abstract_state.params, layer_name),
          config.checkpoint_storage_concurrent_gb,
          config.checkpoint_storage_use_ocdbt,
          config.checkpoint_storage_use_zarr3,
      )
    return params

  def _create_partial_abstract_params(self, abstract_unboxed_params, layer):
    """Creates a partial abstract params structure using ocp.PLACEHOLDER."""

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
  rng = jax.random.PRNGKey(1234)
  quantization.load_and_quantize(rng)


def validate_config(config):
  assert (
      config.load_full_state_path == ""
  ), "Operation on full states not supported! Convert to parameter checkpoint first."


if __name__ == "__main__":
  app.run(main)
