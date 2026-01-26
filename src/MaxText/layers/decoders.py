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

"""Module for decoder layers"""

# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Any, Callable, Tuple
import functools
import inspect

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh

from flax import linen as nn
from flax import nnx
from flax import core
from flax.nnx import wrappers as nnx_wrappers

from MaxText.configs.types import PositionalEmbedding
from MaxText.common_types import DecoderBlockType, ShardMode, Config, EP_AS_CONTEXT
from MaxText.common_types import MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE
from MaxText import max_logging
from MaxText import max_utils
from MaxText.sharding import create_sharding
from MaxText.inference import page_manager
from MaxText.layers import linears
from MaxText.layers import initializers
from MaxText.layers import normalizations
from MaxText.layers import quantizations
from MaxText.layers import pipeline
from MaxText import maxtext_utils
from MaxText import multimodal_utils
from MaxText import sharding
from MaxText.layers.attentions import attention_as_linen, Attention
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.embeddings import Embed, attend_on_embedding, embed_as_linen, positional_embedding_as_linen
from MaxText.layers.quantizations import AqtQuantization as Quant

# Import specific layer definitions
from MaxText.layers import (
    deepseek,
    deepseek_batchsplit,
    gemma,
    gemma2,
    gemma3,
    gpt3,
    gpt_oss,
    llama2,
    llama4,
    mistral,
    mixtral,
    qwen3,
    simple_layer,
)


class DecoderLayer(nnx.Module):
  """
  Transformer decoder layer converted to NNX.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant = None,
      name: str = "decoder_layer",
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant

    cfg = self.config

    self.pre_self_attention_norm = RMSNorm(
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    self.self_attention = Attention(
        config=self.config,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        inputs_q_shape=(1, 1, cfg.emb_dim),
        inputs_kv_shape=(1, 1, cfg.emb_dim),
        mesh=mesh,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        dropout_rate=cfg.dropout_rate,
        float32_qk_product=cfg.float32_qk_product,
        float32_logits=cfg.float32_logits,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
        prefill_cache_axis_order=tuple(map(int, cfg.prefill_cache_axis_order.split(","))),
        ar_cache_axis_order=tuple(map(int, cfg.ar_cache_axis_order.split(","))),
        compute_axis_order=tuple(map(int, cfg.compute_axis_order.split(","))),
        reshape_q=cfg.reshape_q,
        model_mode=model_mode,
        rngs=rngs,
    )

    self.mlp = linears.MlpBlock(
        in_features=cfg.emb_dim,
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        model_mode=model_mode,
        config=cfg,
        quant=self.quant,
        mesh=self.mesh,
        rngs=rngs,
    )

    self.dropout = linears.Dropout(rate=cfg.dropout_rate, rngs=rngs, broadcast_dims=(-2,))

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot: None | int = None,
      page_state: None | page_manager.PageState = None,
      kv_cache: jax.Array | None = None,
      attention_metadata: dict[str, Any] | None = None,
      is_moe_layer: bool = False,
      is_nope_layer: bool = False,
      layer_idx: int = 0,
      **kwargs,
  ):
    cfg = self.config
    mesh = self.mesh
    _maybe_shard_with_logical = functools.partial(
        sharding.maybe_shard_with_logical,
        mesh=mesh,
        shard_mode=cfg.shard_mode,
    )

    if self.model_mode == MODEL_MODE_PREFILL:
      logical_axis_names = ("activation_batch", "prefill_activation_length", "activation_embed")
    elif self.config.expert_shard_attention_option == EP_AS_CONTEXT and self.model_mode == MODEL_MODE_TRAIN:
      logical_axis_names = ("activation_batch_no_exp", "activation_length", "activation_embed")
    else:
      logical_axis_names = ("activation_batch", "activation_length_no_exp", "activation_embed")

    inputs = _maybe_shard_with_logical(inputs, logical_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    lnx = self.pre_self_attention_norm(inputs)
    lnx = _maybe_shard_with_logical(lnx, logical_axis_names)

    # Pass specific metadata to attention if needed
    attention_lnx, kv_cache = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
        is_nope_layer=is_nope_layer,
    )
    attention_lnx = _maybe_shard_with_logical(attention_lnx, logical_axis_names)

    mlp_lnx = self.mlp(lnx, deterministic=deterministic)
    mlp_lnx = _maybe_shard_with_logical(mlp_lnx, logical_axis_names)

    next_layer_addition = mlp_lnx + attention_lnx
    next_layer_addition_dropped_out = self.dropout(next_layer_addition, deterministic=deterministic)

    layer_output = next_layer_addition_dropped_out + inputs
    layer_output = _maybe_shard_with_logical(layer_output, logical_axis_names)

    if cfg.record_internal_nn_metrics:
      self.sow(nnx.Intermediate, "activation_mean", jnp.mean(layer_output))
      self.sow(nnx.Intermediate, "activation_stdev", jnp.std(layer_output))
      self.sow(
          nnx.Intermediate,
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output, kv_cache


class Decoder(nnx.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture, using NNX."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: None | Quant = None,
      model_mode: str = MODEL_MODE_TRAIN,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode
    self.rngs = rngs
    decoder_block_classes = self.get_decoder_layers()

    self.decoder_norm = self.get_norm_layer(num_features=config.emb_dim, rngs=rngs)(
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        parameter_memory_host_offload=config.parameter_memory_host_offload,
    )

    if config.trainable_position_size > 0:
      self.position_embedder = Embed(
          num_embeddings=config.trainable_position_size,
          num_features=config.emb_dim,
          dtype=config.dtype,
          embedding_init=nn.initializers.normal(stddev=1.0),
          config=config,
          mesh=self.mesh,
          rngs=rngs,
      )

    self.dropout = linears.Dropout(rate=config.dropout_rate, rngs=rngs, broadcast_dims=(-2,))

    self.positional_embedding = PositionalEmbedding(embedding_dims=config.base_emb_dim)

    if not config.logits_via_embedding:
      self.logits_dense = linears.DenseGeneral(
          in_features_shape=config.emb_dim,
          out_features_shape=config.vocab_size,
          weight_dtype=config.weight_dtype,
          dtype=jnp.float32 if config.logits_dot_in_fp32 else config.dtype,
          kernel_axes=("embed", "vocab"),
          shard_mode=config.shard_mode,
          matmul_precision=self.config.matmul_precision,
          parameter_memory_host_offload=config.parameter_memory_host_offload,
          rngs=rngs,
      )

    self.scanned_layers = None
    self.is_deepseek = self.config.decoder_block == DecoderBlockType.DEEPSEEK

    if self.config.scan_layers:
      if self.is_deepseek:
        assert len(decoder_block_classes) == 2
        dense_cls, moe_cls = decoder_block_classes

        num_dense = config.first_num_dense_layers
        self.dense_stack = self._create_scanned_layers(dense_cls, length=num_dense, rngs=rngs)

        num_moe = config.num_decoder_layers - config.first_num_dense_layers
        self.moe_stack = self._create_scanned_layers(moe_cls, length=num_moe, rngs=rngs)
      else:
        layer_cls = decoder_block_classes[0]
        num_layers = config.num_decoder_layers
        self.layers = self._create_scanned_layers(layer_cls, length=num_layers, rngs=rngs)
    else:
      self.layers = nnx.List([])
      if self.is_deepseek:
        dense_cls, moe_cls = decoder_block_classes
        for i in range(config.first_num_dense_layers):
          self._create_and_register_layer(dense_cls, rngs, "dense_layer", i)
        for i in range(config.num_decoder_layers - config.first_num_dense_layers):
          self._create_and_register_layer(moe_cls, rngs, "moe_layer", i)
      else:
        layer_cls = decoder_block_classes[0]
        for i in range(config.num_decoder_layers):
          self._create_and_register_layer(layer_cls, rngs, "layers", i)

  def _create_and_register_layer(self, layer_cls, rngs, base_name, i):
    layer = self._create_single_layer(layer_cls, rngs)
    self.layers.append(layer)

  def _create_single_layer(self, decoder_layer_class, rngs, **kwargs):
    """Helper to create a single layer (Linen or NNX)."""
    if issubclass(decoder_layer_class, nnx.Module):
      return decoder_layer_class(
          config=self.config, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode, rngs=rngs, **kwargs
      )
    else:
      layer_linen = decoder_layer_class(
          config=self.config, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode, **kwargs
      )
      return nnx_wrappers.ToNNX(layer_linen, rngs=rngs)

  def _create_scanned_layers(self, decoder_layer_class, length: int, rngs: nnx.Rngs, **layer_kwargs):
    """Creates a VMapped stack of layers, forcing Host memory allocation to avoid OOM."""
    max_logging.log(f"[_create_scanned_layers] vmap len={length}")

    def create_layer_fn(rng):
      layer = decoder_layer_class(
          config=self.config, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode, rngs=rng, **layer_kwargs
      )
      return layer

    nnx.split_rngs(rngs, splits=length)

    # OOM FIX: Force initialization on CPU (Host Memory)
    try:
      cpu_device = jax.devices("cpu")[0]
      with jax.default_device(cpu_device):
        layers_vmapped = nnx.vmap(
            create_layer_fn,
            in_axes=0,
            out_axes=0,
            axis_name="layers",
            transform_metadata={nnx.PARTITION_NAME: "layers"},
        )(rngs)
    except Exception as e:
      max_logging.log(f"[_create_scanned_layers] CPU fallback failed: {e}. Using default device.")
      layers_vmapped = nnx.vmap(
          create_layer_fn,
          in_axes=0,
          out_axes=0,
          axis_name="layers",
          transform_metadata={nnx.PARTITION_NAME: "layers"},
      )(rngs)

    return layers_vmapped

  def _apply_layers_sequentially(self, layers, x_in, *args, length: int, scan_metadata: dict = None, **kwargs):
    policy = self.get_remat_policy()
    prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)

    if scan_metadata is None:
      scan_metadata = {}

    graphdef, params, state = nnx.split(layers, nnx.Param, ...)

    layer_cls = layers.__class__
    sig = inspect.signature(layer_cls.__call__)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    if accepts_kwargs:
      valid_broadcast_kwargs = kwargs
    else:
      valid_broadcast_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    def layer_fn(carry, scanned_inputs):
      (current_params, current_state), metadata_slice = scanned_inputs

      if self.config.parameter_memory_host_offload:
        current_params = jax.tree_map(lambda x: jax.device_put(x, max_utils.device_space()), current_params)

      layer = nnx.merge(graphdef, current_params, current_state)

      step_kwargs = valid_broadcast_kwargs.copy()
      if accepts_kwargs:
        step_kwargs.update(metadata_slice)
      else:
        for k, v in metadata_slice.items():
          if k in sig.parameters:
            step_kwargs[k] = v

      layer_out = layer(carry, *args, **step_kwargs)

      aux_data = None
      if isinstance(layer_out, tuple):
        new_carry = layer_out[0]
        if len(layer_out) > 1:
          aux_data = layer_out[1]
      else:
        new_carry = layer_out

      _, new_params, new_state = nnx.split(layer, nnx.Param, ...)
      return new_carry, (new_params, new_state, aux_data)

    layer_fn = jax.checkpoint(layer_fn, policy=policy, prevent_cse=prevent_cse)
    final_x, (scanned_params, scanned_state, scanned_aux) = jax.lax.scan(layer_fn, x_in, ((params, state), scan_metadata))

    nnx.update(layers, scanned_state)
    return final_x, scanned_aux

  def get_decoder_layers(self):
    """Retrieves decoder layer classes based on config."""
    cfg = self.config

    def get_scannable(normal_cls, scannable_cls):
      return [scannable_cls] if cfg.scan_layers else [normal_cls]

    def get_deepseek():
      if cfg.use_batch_split_schedule:
        return [deepseek_batchsplit.DeepSeekDenseLayer, deepseek_batchsplit.DeepSeekMoELayer]
      return [deepseek.DeepSeekDenseLayer, deepseek.DeepSeekMoELayer]

    layer_map = {
        DecoderBlockType.DEFAULT: [DecoderLayer],
        DecoderBlockType.LLAMA2: [llama2.LlamaDecoderLayer],
        DecoderBlockType.MISTRAL: [mistral.MistralDecoderLayer],
        DecoderBlockType.MIXTRAL: [mixtral.MixtralDecoderLayer],
        DecoderBlockType.GEMMA: [gemma.GemmaDecoderLayer],
        DecoderBlockType.GEMMA2: [gemma2.Gemma2DecoderLayer],
        DecoderBlockType.GEMMA3: [gemma3.Gemma3DecoderLayer],
        DecoderBlockType.GPT3: [gpt3.Gpt3DecoderLayer],
        DecoderBlockType.QWEN3: [qwen3.Qwen3DecoderLayer],
        DecoderBlockType.QWEN3_MOE: [qwen3.Qwen3MoeDecoderLayer],
        DecoderBlockType.SIMPLE: [simple_layer.SimpleDecoderLayer],
        DecoderBlockType.SIMPLE_MLP: [simple_layer.SimpleMlpDecoderLayer],
        DecoderBlockType.DEEPSEEK: get_deepseek(),
        DecoderBlockType.GPT_OSS: get_scannable(gpt_oss.GptOssDecoderLayer, gpt_oss.GptOssScannableBlock),
        DecoderBlockType.QWEN3_NEXT: get_scannable(qwen3.Qwen3NextDecoderLayer, qwen3.Qwen3NextScannableBlock),
        DecoderBlockType.LLAMA4: get_scannable(llama4.Llama4DecoderLayer, llama4.Llama4ScannableBlock),
    }
    return layer_map[cfg.decoder_block]

  def minimal_policy(self, with_context=False):
    names = ["query_proj", "value_proj", "key_proj", "qkv_proj", "out_proj", "mlpwi_0", "mlpwi_1", "mlpwi", "mlpwo"]
    if with_context:
      names.append("context")
    return jax.checkpoint_policies.save_only_these_names(*names)

  def get_remat_policy(self):
    policy = None
    cfg = self.config
    if cfg.remat_policy != "none":
      if cfg.remat_policy in ("minimal_with_context", "minimal_flash"):
        policy = self.minimal_policy(with_context=True)
      elif cfg.remat_policy == "minimal":
        policy = self.minimal_policy()
      elif cfg.remat_policy == "save_dot_with_context_except_mlp":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj", "value_proj", "key_proj", "qkv_proj", "context", "out_proj"
        )
      elif cfg.remat_policy == "save_dot_except_mlpwi":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj", "value_proj", "key_proj", "qkv_proj", "out_proj", "mlpwo"
        )
      elif cfg.remat_policy == "save_dot_except_mlp":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj", "value_proj", "key_proj", "qkv_proj", "out_proj"
        )
      elif cfg.remat_policy == "save_qkv_proj":
        policy = jax.checkpoint_policies.save_only_these_names("query_proj", "value_proj", "key_proj", "qkv_proj")
      elif cfg.remat_policy == "qkv_proj_offloaded":
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            [], ["query_proj", "value_proj", "key_proj"], "device", "pinned_host"
        )
      elif cfg.remat_policy == "minimal_offloaded":
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            [],
            [
                "query_proj",
                "value_proj",
                "key_proj",
                "qkv_proj",
                "out_proj",
                "mlpwi_0",
                "mlpwi_1",
                "mlpwi",
                "mlpwo",
            ],
            "device",
            "pinned_host",
        )
      elif cfg.remat_policy == "custom":
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            cfg.tensors_on_device, cfg.tensors_to_offload, "device", "pinned_host"
        )
      elif cfg.remat_policy == "save_out_proj":
        policy = jax.checkpoint_policies.save_only_these_names("out_proj")
      else:
        policy = None
    return policy

  def get_norm_layer(self, num_features: int, rngs: nnx.Rngs):
    if self.config.decoder_block == DecoderBlockType.GPT3:
      return functools.partial(
          gpt3.Gpt3LayerNorm, num_features=num_features, reductions_in_fp32=False, use_bias=True, rngs=rngs
      )
    else:
      return functools.partial(RMSNorm, num_features=num_features, shard_mode=self.config.shard_mode, rngs=rngs)

  def _apply_embedding(
      self,
      shared_embedding,
      decoder_input_tokens,
      decoder_positions,
      deterministic,
      model_mode,
      image_embeddings=None,
      bidirectional_mask=None,
      image_masks=None,
  ):
    cfg = self.config
    y = shared_embedding(decoder_input_tokens.astype("int32"), model_mode=model_mode)
    if image_embeddings is not None and cfg.use_multimodal:
      y = multimodal_utils.merge_mm_embeddings(
          text_embeddings=y, vision_embeddings=image_embeddings, mask=bidirectional_mask, image_masks=image_masks
      )
    y = self.dropout(y, deterministic=deterministic)
    y = y.astype(cfg.dtype)
    if cfg.use_untrainable_positional_embedding:
      y = self.positional_embedding(y, decoder_positions)
    if cfg.trainable_position_size > 0 and self.position_embedder:
      y += self.position_embedder(decoder_positions.astype("int32"), model_mode=model_mode)
    return y

  def apply_output_head(self, shared_embedding, y, deterministic, model_mode):
    cfg = self.config
    norm_out_sharding = (
        create_sharding(self.mesh, ("activation_batch", "activation_length_no_exp", "activation_embed"))
        if cfg.shard_mode == ShardMode.EXPLICIT
        else None
    )
    y = self.decoder_norm(y, out_sharding=norm_out_sharding)
    y = self.dropout(y, deterministic=deterministic)

    if model_mode in (MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE):
      out_sharding = create_sharding(self.mesh, (None, None, "activation_vocab"))
    else:
      out_sharding = create_sharding(
          self.mesh, ("activation_embed_and_logits_batch", "activation_length_no_exp", "activation_vocab")
      )

    if cfg.logits_via_embedding:
      if isinstance(shared_embedding, nnx.Module):
        embedding_table = shared_embedding.embedding.value
      else:
        embedding_table = shared_embedding.variables["params"]["embedding"]
      if isinstance(embedding_table, nn.spmd.LogicallyPartitioned):
        embedding_table = embedding_table.unbox()
      attend_dtype = jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype
      logits = attend_on_embedding(y, embedding_table, attend_dtype, self.config, out_sharding)
      if self.config.normalize_embedding_logits:
        logits = logits / jnp.sqrt(y.shape[-1])
      if cfg.final_logits_soft_cap:
        logits = jnp.tanh(logits / cfg.final_logits_soft_cap) * cfg.final_logits_soft_cap
    else:
      logits = self.logits_dense(y, out_sharding=out_sharding)

    if self.config.cast_logits_to_fp32:
      logits = logits.astype(jnp.float32)
    return logits

  def __call__(
      self,
      shared_embedding: Any,
      decoder_input_tokens,
      decoder_positions,
      decoder_segment_ids=None,
      deterministic=False,
      model_mode=MODEL_MODE_TRAIN,
      previous_chunk=None,
      slot: None | int = None,
      page_state: None | page_manager.PageState = None,
      bidirectional_mask: None | Any = None,
      image_embeddings: None | jnp.ndarray = None,
      image_masks: None | jnp.ndarray = None,
      kv_caches: list[jax.Array] | None = None,
      attention_metadata=None,
  ):
    cfg = self.config

    y = self._apply_embedding(
        shared_embedding,
        decoder_input_tokens,
        decoder_positions,
        deterministic,
        model_mode,
        image_embeddings,
        bidirectional_mask,
        image_masks,
    )
    layer_args = (decoder_segment_ids, decoder_positions, deterministic, model_mode)
    layer_kwargs = {
        "previous_chunk": previous_chunk,
        "page_state": page_state,
        "slot": slot,
        "attention_metadata": attention_metadata,
    }
    if cfg.decoder_block == DecoderBlockType.GEMMA3:
      layer_kwargs["bidirectional_mask"] = bidirectional_mask

    if cfg.scan_layers:
      if self.is_deepseek:
        y, _ = self._apply_layers_sequentially(
            self.dense_stack, y, *layer_args, length=cfg.first_num_dense_layers, **layer_kwargs
        )
        num_moe = cfg.num_decoder_layers - cfg.first_num_dense_layers
        y, _ = self._apply_layers_sequentially(self.moe_stack, y, *layer_args, length=num_moe, **layer_kwargs)
      else:
        scan_metadata = {}
        if cfg.decoder_block == DecoderBlockType.LLAMA4:
          scan_metadata["is_nope_layer"] = jnp.array(
              [llama4.determine_is_nope_layer(i, cfg.nope_layer_interval) for i in range(cfg.num_decoder_layers)]
          )
          scan_metadata["is_moe_layer"] = jnp.array(
              [llama4.determine_is_moe_layer(i, cfg.interleave_moe_layer_step) for i in range(cfg.num_decoder_layers)]
          )
        elif cfg.decoder_block == DecoderBlockType.QWEN3_NEXT:
          scan_metadata["layer_idx"] = jnp.arange(cfg.num_decoder_layers)
        y, _ = self._apply_layers_sequentially(
            self.layers,
            y,
            *layer_args,
            length=cfg.num_decoder_layers,
            scan_metadata=scan_metadata,
            **layer_kwargs,
        )
    else:
      for i, layer in enumerate(self.layers):
        kv_cache = kv_caches[i] if kv_caches is not None else None
        step_kwargs = layer_kwargs.copy()
        if cfg.decoder_block == DecoderBlockType.LLAMA4:
          step_kwargs["is_nope_layer"] = llama4.determine_is_nope_layer(i, cfg.nope_layer_interval)
          step_kwargs["is_moe_layer"] = llama4.determine_is_moe_layer(i, cfg.interleave_moe_layer_step)
        if cfg.decoder_block == DecoderBlockType.QWEN3_NEXT:
          step_kwargs["layer_idx"] = i
        out = layer(y, *layer_args, kv_cache=kv_cache, **step_kwargs)
        if isinstance(out, tuple):
          y, kv_cache_out = out[0], out[1] if len(out) > 1 else None
        else:
          y, kv_cache_out = out, None
        if kv_caches is not None:
          kv_caches[i] = kv_cache_out

    assert isinstance(y, jax.Array)
    hidden_state = y
    if cfg.attention == "vllm_rpa":
      _ = self.apply_output_head(shared_embedding, hidden_state, deterministic, model_mode)
    if cfg.attention == "vllm_rpa" or (cfg.num_vocab_tiling > 1 and self.model_mode == MODEL_MODE_TRAIN):
      logits = None
    else:
      logits = self.apply_output_head(shared_embedding, hidden_state, deterministic, model_mode)
    return logits, hidden_state, kv_caches


def decoder_as_linen(config: Config, mesh: Mesh, rngs: nnx.Rngs, model_mode: str, quant: None | Quant = None):
  module = nnx_wrappers.to_linen(
      Decoder,
      config=config,
      mesh=mesh,
      model_mode=model_mode,
      rngs=rngs,
      quant=quant,
      name="decoder",
      metadata_fn=initializers.variable_to_logically_partitioned,
  )
  return module
