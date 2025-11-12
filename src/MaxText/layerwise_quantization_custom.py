import os
from typing import Any, Sequence

from tqdm import tqdm
import jax
import jax.numpy as jnp
from absl import app
import pprint

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
    print("DEBUG: LayerwiseQuantization initialized.")
    print(f"DEBUG: Config: {config}")

  def load_and_quantize(self, rng: None | PRNGKeyType = None) -> None:
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
    rngs = nnx.Rngs(rng)

    print(f"DEBUG: Starting load_and_quantize. Quantization mode: {self.quant.quant_mode}")

    layers = [
        deepseek.DeepSeekDenseLayer(config=config, mesh=self._mesh, quant=self.quant, model_mode=model_mode),
        deepseek.DeepSeekMoELayer(config=config, mesh=self._mesh, quant=self.quant, model_mode=model_mode),
    ][::-1]
    layer_prefixes = ["dense_layers", "moe_layers"][::-1]
    num_moe_layers = config.num_decoder_layers - config.first_num_dense_layers
    num_layers_list = [config.first_num_dense_layers, num_moe_layers][::-1]

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
        print(f"\nDEBUG: --- Processing layer: {layer_name} ---")

        print(f"DEBUG: Loading params for {layer_name}...")
        params = self._load_layer(layer_name)
        params["params"] = params["params"]["decoder"][layer_name]
        print(f"DEBUG: Loaded params shapes for {layer_name}:")
        jax.tree_util.tree_map_with_path(
            lambda path, x: print(f"  {jax.tree_util.keystr(path)}: {x.shape}"),
            params["params"]
        )

        print(f"DEBUG: Running model_apply for {layer_name}...")
        _, new_vars = model_apply(params, rng_quant_params, layer)

        if "aqt" not in new_vars:
            print(f"Warning: 'aqt' not found in new_vars for {layer_name}. Skipping AQT processing for this layer.")
            quantized_params["params"]["decoder"][layer_name] = params["params"] # Keep original params
            continue

        quantized_params["aqt"]["decoder"][layer_name] = new_vars["aqt"]
        print(f"DEBUG: new_vars['aqt'] shapes for {layer_name}:")
        jax.tree_util.tree_map_with_path(
            lambda path, x: print(f"  {jax.tree_util.keystr(path)}: {x.shape}"),
            new_vars["aqt"]
        )

        print(f"DEBUG: Calling quantizations.remove_quantized_params for {layer_name}...")
        try:
            removed_params = quantizations.remove_quantized_params(
                params["params"], new_vars["aqt"]
            )
            quantized_params["params"]["decoder"][layer_name] = removed_params
            print(f"DEBUG: Successfully removed quantized params for {layer_name}.")
            print(f"DEBUG: Resulting param keys for {layer_name}:")
            jax.tree_util.tree_map_with_path(
                lambda path, x: print(f"  {jax.tree_util.keystr(path)}: type {type(x)}"),
                removed_params
            )
        except Exception as e:
            print(f"ERROR: Failed to remove quantized params for {layer_name}: {e}")
            print(f"DEBUG: Dumping params['params'] keys for {layer_name}:")
            jax.tree_util.tree_map_with_path(lambda path, _: print(f"  {jax.tree_util.keystr(path)}"), params["params"])
            print(f"DEBUG: Dumping new_vars['aqt'] keys for {layer_name}:")
            jax.tree_util.tree_map_with_path(lambda path, _: print(f"  {jax.tree_util.keystr(path)}"), new_vars["aqt"])
            # Optional: re-raise the exception if you want to stop execution
            # raise

    print("\nDEBUG: --- Processing unquantized layers ---")
    unquantized_layers = ["decoder_norm", "logits_dense"]
    for unquantized_layer in unquantized_layers:
      print(f"DEBUG: Loading {unquantized_layer}...")
      params = self._load_layer(unquantized_layer)
      quantized_params["params"]["decoder"][unquantized_layer] = params["params"]["decoder"][unquantized_layer]
    print(f"DEBUG: Loading token_embedder...")
    quantized_params["params"]["token_embedder"] = self._load_layer("token_embedder")["params"]["token_embedder"]

    print("\nDEBUG: --- Final quantized_params structure ---")
    final_struct = jax.tree_util.tree_map(lambda x: type(x), quantized_params)
    pprint.pprint(final_struct)

    print("\nDEBUG: Saving checkpoint if configured...")
    maxtext_utils.save_quantized_checkpoint_if_configured(self.config, quantized_params)
    print("DEBUG: Layerwise quantization finished.")

  def _load_layer(self, layer_name):
    config = self.config
    with nn_partitioning.axis_rules(config.logical_axis_rules):

      partial_abstract_params = self._create_partial_abstract_params(self.unboxed_abstract_state.params, layer_name)
      # print(f"DEBUG: Abstract params for {layer_name}: {jax.tree_map(lambda x: x.shape, partial_abstract_params)}")
      params = checkpointing.load_params_from_path(
          config.load_parameters_path,
          partial_abstract_params,
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

