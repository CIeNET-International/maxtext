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

from aqt.jax.v2 import aqt_tensor
from tqdm import tqdm

import jax
import jax.numpy as jnp
from absl import app

from flax.linen import partitioning as nn_partitioning

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


def get_original_path_key(aqt_k_tuple: tuple[DictKey, ...]) -> tuple[DictKey, ...] | None:
  """
  Maps an AQT PyTree path (tuple of keys) to its corresponding original parameter path.
  Only returns a path if it corresponds to a parameter to be removed.
  """
  aqt_k = list(aqt_k_tuple)
  str_path = jax.tree_util.keystr(aqt_k_tuple)

  # AqtEinsum_ wraps an operation, not a parameter, so no original param to remove.
  if "AqtEinsum_" in str_path:
    return None

  # AqtDotGeneral_ wraps a weight tensor, typically a 'kernel'.
  if "AqtDotGeneral_" in str_path:
    aqt_module_index = -1
    for i, key in enumerate(aqt_k):
      if isinstance(key, DictKey) and key.key.startswith("AqtDotGeneral_"):
        aqt_module_index = i
        break

    if aqt_module_index != -1:
      # Check if the path structure beyond the AQT module matches the expected
      # QTensor location: ['Aqt...']['qrhs']['frozen']
      if (
          len(aqt_k) > aqt_module_index + 2
          and isinstance(aqt_k[aqt_module_index + 1], DictKey)
          and aqt_k[aqt_module_index + 1].key == "qrhs"
          and isinstance(aqt_k[aqt_module_index + 2], DictKey)
          and aqt_k[aqt_module_index + 2].key == "frozen"
      ):
        # The original parameter path is the prefix before the AQT module,
        # and we assume the original tensor was named 'kernel'.
        parent_path = tuple(aqt_k[:aqt_module_index])
        return parent_path + (DictKey("kernel"),)
      else:
        print(f"Warning: Unexpected structure under AqtDotGeneral_ at {str_path}")
        return None
  return None


def get_quantized_param_paths(aqt_params: Any, params: Any) -> set[tuple[DictKey, ...]]:
  """
  Identifies the set of paths in the original params tree that have been quantized.
  """
  is_qtensor = lambda x: isinstance(x, aqt_tensor.QTensor)
  aqt_param_flat, _ = jax.tree_util.tree_flatten_with_path(aqt_params, is_leaf=is_qtensor)

  if not aqt_param_flat:
    return set()

  param_tree_flat_with_path, _ = jax.tree_util.tree_flatten_with_path(params)
  params_path_set: set[tuple[DictKey, ...]] = {tuple(k) for k, _ in param_tree_flat_with_path}

  original_param_paths_to_remove: Set[Tuple[DictKey, ...]] = set()

  for aqt_k_tuple, _ in aqt_param_flat:
    original_k_tuple = get_original_path_key(aqt_k_tuple)

    if original_k_tuple is not None:
      if original_k_tuple in params_path_set:
        original_param_paths_to_remove.add(original_k_tuple)
      else:
        # This error is critical as it indicates a mismatch between
        # the AQT structure and the expected original params structure.
        params_keys_str = {jax.tree_util.keystr(k) for k in params_path_set}
        raise ValueError(
            f"Mapped AQT path {jax.tree_util.keystr(aqt_k_tuple)} to {jax.tree_util.keystr(original_k_tuple)}, "
            f"but this path was not found in the original params tree. "
            f"Available param paths: {params_keys_str}"
        )
  return original_param_paths_to_remove

def remove_quantized_params(params: Any, aqt_vars: Any) -> Any:
  """Replaces the values in the original params tree that are now quantized with empty dicts."""
  quantized_param_path_set = get_quantized_param_paths(aqt_vars, params)

  if not quantized_param_path_set:
    print("No parameters to remove.")
    return params

  print(f"Attempting to remove {len(quantized_param_path_set)} quantized parameter paths.")

  def _map_fn(path, value):
    if tuple(path) in quantized_param_path_set:
      return {}  # Replace quantized parameter with an empty dict
    return value

  return jax.tree_util.tree_map_with_path(_map_fn, params)


class LayerwiseQuantization:
  """
  Layerwise quantization for large models.
  """

  def __init__(self, config: Any, rng: PRNGKeyType):
    self.config = config
    self.rng = rng
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
    self.unboxed_abstract_state, _, _ = maxtext_utils.get_abstract_state(model, None, self.config, self.rng, self._mesh, False)

  def load_and_quantize(self) -> None:
    """
    Load parameters layer by layer and quantize them.
    """
    quantized_params = {}
    quantized_params["params"] = {"decoder": {}}
    quantized_params["aqt"] = {"decoder": {}}
    config = self.config
    self.quant.quant_mode = quantizations.get_quant_mode("convert")
    model_mode = common_types.MODEL_MODE_PREFILL
    _, rng_quant_params = jax.random.split(self.rng)

    layers = [
        deepseek.DeepSeekDenseLayer(
            config=config,
            mesh=self._mesh,
            quant=self.quant,
            model_mode=model_mode,
        ),
        deepseek.DeepSeekMoELayer(
            config=config,
            mesh=self._mesh,
            quant=self.quant,
            model_mode=model_mode,
        ),
    ]
    layer_prefixes = ["dense_layers", "moe_layers"]
    num_moe_layers = config.num_decoder_layers - config.first_num_dense_layers
    num_layers_list = [config.first_num_dense_layers, num_moe_layers]

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
      if layer in [x.key for x in path]:
        return True
      return False

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
  rng = jax.random.PRNGKey(1234)
  quantization = LayerwiseQuantization(config,rng)
  quantization.load_and_quantize()


def validate_config(config):
  assert (
      config.load_full_state_path == ""
  ), "Operation on full states not supported! Convert to parameter checkpoint first."


if __name__ == "__main__":
  app.run(main)
