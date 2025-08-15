#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

""""Module for decoder layers."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Any, Callable, Optional, Union, Type
import functools

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh

from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.core.spmd import logical_axis_rules as nn_logical_axis_rules
from flax import nnx
import numpy as np

from MaxText.common_types import Array, DecoderBlockType, Config, MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE
from MaxText import max_logging
from MaxText import max_utils
from MaxText.inference import page_manager
from MaxText.layers import linears
from MaxText.layers import quantizations
from MaxText.layers import pipeline
from MaxText import maxtext_utils
from MaxText import multimodal_utils
from MaxText.layers.attentions import attention_as_linen,Attention
from MaxText.layers.normalizations import rms_norm,RMSNorm
from MaxText.layers.embeddings import attend_on_embedding, embed_as_linen, positional_embedding_as_linen
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.layers import (
    deepseek,
    gemma,
    gemma2,
    gemma3,
    gpt3,
    llama2,
    llama4,
    mistral,
    mixtral,
    qwen3,
    simple_layer,
)

# ------------------------------------------------------------------------------
# The network: Decoder Definitions
# ------------------------------------------------------------------------------


class DecoderLayer(nnx.Module):
  """
  Transformer decoder layer that attends to the encoder.
  This is the core, reusable building block for both the main model's
  decoder stack and the auxiliary MTP layers.
  """

  def __init__(self, config: Config, mesh: Mesh, quant: Optional[Quant] = None, model_mode: str = MODEL_MODE_TRAIN, rngs: nnx.Rngs | None = None):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode
    self.rngs = rngs if rngs is not None else nnx.Rngs(0)
    inputs_shape = (
      int(self.config.per_device_batch_size),
      int(self.config.max_target_length),
      int(self.config.emb_dim),
    )

    self.mlp = linears.MlpBlock(
      in_features=inputs_shape[-1],
      intermediate_dim=self.config.mlp_dim,
      activations=self.config.mlp_activations,
      intermediate_dropout_rate=self.config.dropout_rate,
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      config=self.config,
      quant=quant,
      model_mode=model_mode,
      rngs=self.rngs
    )

    self.drop_out = nnx.Dropout(rate=self.config.dropout_rate, broadcast_dims=(-2,),rngs=self.rngs)

    self.pre_self_attention_norm = RMSNorm(
      num_features=inputs_shape[-1],
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      kernel_axes=("norm", ),
      epsilon=self.config.normalization_layer_epsilon,
      rngs=self.rngs
    )

    self.self_attention = Attention(
        config=self.config,
        num_query_heads=self.config.num_query_heads,
        num_kv_heads=self.config.num_kv_heads,
        head_dim=self.config.head_dim,
        max_target_length=self.config.max_target_length,
        max_prefill_predict_length=self.config.max_prefill_predict_length,
        attention_kernel=self.config.attention,
        inputs_q_shape=inputs_shape,
        inputs_kv_shape=inputs_shape,
        mesh=mesh,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        dropout_rate=self.config.dropout_rate,
        name="self_attention",
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(self.config),
        prefill_cache_axis_order=tuple(map(int, self.config.prefill_cache_axis_order.split(","))),
        ar_cache_axis_order=tuple(map(int, self.config.ar_cache_axis_order.split(","))),
        compute_axis_order=tuple(map(int, self.config.compute_axis_order.split(","))),
        reshape_q=self.config.reshape_q,
        model_mode=model_mode,
    )

  def __call__(
      self,
      inputs : Array,
      decoder_segment_ids : Array,
      decoder_positions : Array,
      deterministic : bool,
      model_mode : str ,
      previous_chunk: Optional[Array] = None,
      slot: Optional[int] = None,
      page_state: Optional[page_manager.PageState] = None,
  ):
    cfg = self.config
    logical_axis_names = (
      ("activation_batch", "prefill_activation_length", "activation_embed")
      if model_mode == MODEL_MODE_PREFILL
      else ("activation_batch", "activation_length", "activation_embed")
    )

    inputs = nn.with_logical_constraint(inputs, logical_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    lnx = self.pre_self_attention_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, logical_axis_names)

    attention_lnx = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )
    attention_lnx = nn.with_logical_constraint(attention_lnx, logical_axis_names)

    # MLP block.
    mlp_lnx = self.mlp(lnx, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, logical_axis_names)

    next_layer_addition = mlp_lnx + attention_lnx

    next_layer_addition_dropped_out = self.drop_out(
        next_layer_addition, deterministic=deterministic
    )

    layer_output = next_layer_addition_dropped_out + inputs
    layer_output = nn.with_logical_constraint(
        layer_output,
        logical_axis_names,
    )

    if cfg.record_internal_nn_metrics:
      self.sow(nnx.Intermediate, "activation_mean", jnp.mean(layer_output))
      self.sow(nnx.Intermediate, "activation_stdev", jnp.std(layer_output))
      self.sow(
          nnx.Intermediate,
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    return layer_output, None if cfg.scan_layers else layer_output


class SequentialBlockDecoderLayers(nnx.Module):
  """Sequential unscanned series of decoder layers."""

  def __init__(self,decoder_layer:Any,num_decoder_layers:int, config: Config, mesh: Mesh, quant: Quant, model_mode: str = MODEL_MODE_TRAIN, rngs: nnx.Rngs | None = None):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode
    self.rngs = rngs if rngs is not None else nnx.Rngs(0)
    self.num_decoder_layers = num_decoder_layers
    self.decoder_layer = decoder_layer

  def __call__(
      self,
      inputs: Array,
      decoder_segment_ids: Array,
      decoder_positions : Array,
      deterministic: bool,
      model_mode : str,
      slot: Optional[int] = None,
      page_state: Optional[page_manager.PageState] = None,
  ) -> Union[Array, tuple[Array, None]]:
    for lyr in range(self.num_decoder_layers):
      inputs = self.decoder_layer(
          config=self.config, mesh=self.mesh, name=f"layers_{lyr}", quant=self.quant, model_mode=model_mode
      )(
          inputs,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          slot=slot,
          page_state=page_state,
      )
      if self.config.scan_layers:
        inputs = inputs[0]  #  When scan_layers is True the decoder layers return (outputs, None).
    if self.config.scan_layers:
      return inputs, None  # pytype: disable=bad-return-type
    return inputs

class Decoder(nnx.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""

  def __init__(
      self,
      config: Config,
      shared_embedding: nn.Module,
      mesh: Mesh,
      quant: Quant,
      model_mode: str = MODEL_MODE_TRAIN,
      rngs : nnx.Rngs | None = None,
  ):
    """Initialize the decoder module.

    Args:
        config: Configuration object containing model parameters.
        shared_embedding: Shared embedding layer for token embeddings.
        mesh: JAX mesh for distributed computation.
        quant: Optional quantization configuration.
        model_mode: Mode of the model (e.g., training, prefill, autoregressive).
    """
    self.config = config
    self.shared_embedding = shared_embedding
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode
    self.rngs = rngs if rngs is not None else nnx.Rngs(0)

    """Initialize decoder layer."""
    #TODO: retrieve more layers from __call__ function
    self.decoder_layer = self.get_decoder_layers()
    self.norm_layer = self.get_norm_layer(num_features=self.config.emb_dim)
    self._pipeline_module: Optional[pipeline.Pipeline] = None
      
    if self.config.using_pipeline_parallelism:
      remat_policy = self.get_remat_policy()
      pipeline_stage_module = self.get_pipeline_stage_module(self.decoder_layer)
      self._pipeline_module = pipeline.Pipeline(
          config=self.config,
          mesh=self.mesh,
          layers=pipeline_stage_module,
          remat_policy=remat_policy,
      )
      if self.config.decoder_block == DecoderBlockType.DEEPSEEK:
          self.upper_layer = self._build_deepseek_pipeline_layers()
      else:
          self.upper_layer = self._build_pipeline_layers()
    else:
      
      self._scanned_layers: Optional[Callable] = None
      self._unscanned_layers: Optional[list[nnx.Module]] = None

      self._init_non_pipeline_layers()

      
    # Embedding-related layers
    self.dropout_layer = nn.Dropout(
        rate=self.config.dropout_rate, 
        broadcast_dims=(-2,)
    )

    # Untrainable (static) positional embedding
    self.static_pos_embedding = None
    if self.config.use_untrainable_positional_embedding:
        self.static_pos_embedding = positional_embedding_as_linen(
            embedding_dims=self.config.base_emb_dim
        )

    # Trainable position embedding
    self.trainable_pos_embedding = None
    if self.config.trainable_position_size > 0:
        self.trainable_pos_embedding = embed_as_linen(
            num_embeddings=self.config.trainable_position_size,
            num_features=self.config.emb_dim,
            dtype=self.config.dtype,
            embedding_init=nn.initializers.normal(stddev=1.0),
            name="position_embedder",
            config=self.config,
        )
        
  def _init_non_pipeline_layers(self):
      cfg = self.config
      mesh = self.mesh
      in_axes_tuple = (nn.broadcast,) * 4

      if cfg.scan_layers:
          if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
              dense_layer, moe_layer = self.decoder_layer
              scan_fn_dense = self.scan_decoder_layers(cfg, dense_layer, cfg.first_num_dense_layers, "dense_layers", mesh, in_axes_tuple, model_mode=self.model_mode)
              scan_fn_moe = self.scan_decoder_layers(cfg, moe_layer, cfg.num_decoder_layers - cfg.first_num_dense_layers, "moe_layers", mesh, in_axes_tuple, model_mode=self.model_mode)

              self._scanned_layers = lambda y, *args, **kwargs: scan_fn_moe(*scan_fn_dense(y, *args, **kwargs), *args, **kwargs)
          elif cfg.decoder_block == DecoderBlockType.GEMMA3:
              self._init_gemma3_scanned_layers()
          else:
              remat_policy = self.get_remat_policy()
              block_layer = self.set_remat_policy(self.decoder_layer, remat_policy)[0]
              scan_length = cfg.num_decoder_layers // cfg.inhomogeneous_layer_cycle_interval
              scan_fn = self.scan_decoder_layers(cfg, block_layer, scan_length, "layers", mesh, in_axes_tuple, model_mode=self.model_mode)

              self._scanned_layers = lambda y, *args, **kwargs: scan_fn(y, *args, **kwargs)[0]

      else:
          self._unscanned_layers = []
          if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
              dense_layer, moe_layer = self.decoder_layer
              for i in range(cfg.first_num_dense_layers):
                  self._unscanned_layers.append(dense_layer(config=cfg, mesh=mesh, quant=self.quant, model_mode=self.model_mode, name=f"dense_layer_{i}"))
              for i in range(cfg.num_decoder_layers - cfg.first_num_dense_layers):
                  self._unscanned_layers.append(moe_layer(config=cfg, mesh=mesh, quant=self.quant, model_mode=self.model_mode, name=f"moe_layer_{i}"))
          else:
              layer_cls = self.decoder_layer[0]
              for i in range(cfg.num_decoder_layers):
                  layer_kwargs = {}
                  if cfg.decoder_block == DecoderBlockType.GEMMA3:
                      layer_kwargs["attention_type"] = gemma3.get_attention_type(layer_id=i)
                  elif cfg.decoder_block == DecoderBlockType.LLAMA4:
                      layer_kwargs["is_nope_layer"] = llama4.determine_is_nope_layer(i, cfg.nope_layer_interval)
                      layer_kwargs["is_moe_layer"] = llama4.determine_is_moe_layer(i, cfg.interleave_moe_layer_step)

                  self._unscanned_layers.append(
                      layer_cls(
                          config=cfg,
                          mesh=mesh,
                          quant=self.quant,
                          model_mode=self.model_mode,
                          name=f"layer_{i}",
                          **layer_kwargs,
                      )
                  )

  def _build_deepseek_pipeline_layers(self) -> Callable:
      """Constructs DeepSeek pipeline logic with dense + moe scanned layers + pipeline."""
      cfg = self.config
      mesh = self.mesh
      layers = self.decoder_layer
      if len(layers) != 2:
          raise ValueError(f"Expected 2 layers for DeepSeek, got {len(layers)}")

      dense_layer, moe_layer = layers

      # Create scanned layers for the part *before* pipeline
      scan_fn_dense = None
      scan_fn_moe = None
      num_dense = cfg.first_num_dense_layers
      num_moe_total = cfg.num_decoder_layers - num_dense
      num_moe_outside = num_moe_total - cfg.pipeline_parallel_layers

      logical_axis_rules_pp_as_dp = maxtext_utils.logical_axis_rules_pp_act_as_dp(cfg.logical_axis_rules)
      with mesh, nn_logical_axis_rules(logical_axis_rules_pp_as_dp):
          if num_dense > 0:
              scan_fn_dense = self.scan_decoder_layers(
                  cfg,
                  dense_layer,
                  num_dense,
                  "dense_layers",
                  mesh,
                  in_axes_tuple=(nn.broadcast,) * 4,
                  model_mode=self.model_mode,
              )

          if num_moe_outside > 0:
              scan_fn_moe = self.scan_decoder_layers(
                  cfg,
                  moe_layer,
                  num_moe_outside,
                  "moe_layers",
                  mesh,
                  in_axes_tuple=(nn.broadcast,) * 4,
                  model_mode=self.model_mode,
              )

      # Get partition spec if needed
      partition_spec = None
      if cfg.pipeline_fsdp_ag_once:
          dummy_input = jnp.zeros((1, 1, cfg.emb_dim), dtype=cfg.dtype)
          dummy_args = tuple(jnp.zeros((1, 1), dtype="int32") for _ in range(4))
          partition_spec = self.pipeline_module.get_weight_sharding(dummy_input, *dummy_args)

      def forward_fn(y: Array, decoder_segment_ids, decoder_positions, deterministic, model_mode):
          if scan_fn_dense:
              y, _ = scan_fn_dense(y, decoder_segment_ids, decoder_positions, deterministic, model_mode)
          if scan_fn_moe:
              y, _ = scan_fn_moe(y, decoder_segment_ids, decoder_positions, deterministic, model_mode)
          y = self.pipeline_module(y, decoder_segment_ids, decoder_positions, deterministic, model_mode, partition_spec=partition_spec)
          return y

      return forward_fn

  def _build_pipeline_layers(self) -> Callable:
    cfg = self.config
    mesh = self.mesh
    layer_cls = self.decoder_layer[0]  # Standard case: only one decoder layer

    # Layers outside the pipeline (if any)
    scan_fn_outside = None
    remaining_layers = cfg.num_decoder_layers - cfg.pipeline_parallel_layers
    if remaining_layers > 0:
        logical_axis_rules_pp_as_dp = maxtext_utils.logical_axis_rules_pp_act_as_dp(cfg.logical_axis_rules)
        with mesh, nn_logical_axis_rules(logical_axis_rules_pp_as_dp):
            scan_fn_outside = self.scan_decoder_layers(
                cfg,
                layer_cls,
                remaining_layers,
                "layers_outside_pipeline",
                mesh,
                in_axes_tuple=(nn.broadcast,) * 4,
                model_mode=self.model_mode,
            )

    # Get weight sharding partition spec if needed
    partition_spec = None
    if cfg.pipeline_fsdp_ag_once:
        dummy_input = jnp.zeros((1, 1, cfg.emb_dim), dtype=cfg.dtype)
        dummy_args = tuple(jnp.zeros((1, 1), dtype="int32") for _ in range(4))
        partition_spec = self.pipeline_module.get_weight_sharding(dummy_input, *dummy_args)

    # Return a composite callable that applies pipeline + remaining scan
    def forward_fn(y: Array, decoder_segment_ids, decoder_positions, deterministic, model_mode):
        y = self.pipeline_module(
            y,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
            partition_spec=partition_spec,
        )
        if scan_fn_outside is not None:
            y, _ = scan_fn_outside(
                y,
                decoder_segment_ids,
                decoder_positions,
                deterministic,
                model_mode,
            )
        return y

    return forward_fn

  def _get_layer_kwargs_for_decoder(self, decoder_block, layer_number=0):
      cfg = self.config
      if decoder_block == DecoderBlockType.GEMMA3:
          return {"attention_type": gemma3.get_attention_type(layer_number)}
      if decoder_block == DecoderBlockType.LLAMA4:
          return {
              "is_nope_layer": llama4.determine_is_nope_layer(layer_number, cfg.nope_layer_interval),
              "is_moe_layer": llama4.determine_is_moe_layer(layer_number, cfg.interleave_moe_layer_step),
          }
      return {}

  def _build_non_pipeline_layers(self):
      cfg = self.config
      mesh = self.mesh
      in_axes_tuple = (nn.broadcast,) * 4
      remat_policy = self.get_remat_policy()

      if cfg.scan_layers:
          if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
              if len(self.decoder_layer) != 2:
                  raise ValueError("Expected 2 layers for DeepSeek.")
              dense_layer, moe_layer = self.decoder_layer

              dense_scan_fn = None
              moe_scan_fn = None

              if cfg.first_num_dense_layers > 0:
                  dense_scan_fn = self.scan_decoder_layers(
                      cfg, dense_layer, cfg.first_num_dense_layers, "dense_layers", mesh,
                      in_axes_tuple, model_mode=self.model_mode,
                  )

              if cfg.num_decoder_layers - cfg.first_num_dense_layers > 0:
                  moe_scan_fn = self.scan_decoder_layers(
                      cfg, moe_layer, cfg.num_decoder_layers - cfg.first_num_dense_layers,
                      "moe_layers", mesh, in_axes_tuple, model_mode=self.model_mode,
                  )

              def scanned_deepseek_fn(y, *args, **kwargs):
                  if dense_scan_fn:
                      y, _ = dense_scan_fn(y, *args, **kwargs)
                  if moe_scan_fn:
                      y, _ = moe_scan_fn(y, *args, **kwargs)
                  return y

              self._scanned_layers = scanned_deepseek_fn

          elif cfg.decoder_block == DecoderBlockType.GEMMA3:
              self._init_gemma3_scanned_layers()  # Already defined earlier

          else:
              # All other decoder types with scanning
              layer_cls = self.set_remat_policy([self.decoder_layer[0]], remat_policy)[0]
              scan_len = cfg.num_decoder_layers // cfg.inhomogeneous_layer_cycle_interval

              scan_fn = self.scan_decoder_layers(
                  cfg,
                  layer_cls,
                  scan_len,
                  "layers",
                  mesh,
                  in_axes_tuple,
                  model_mode=self.model_mode,
                  **self._get_layer_kwargs_for_decoder(cfg.decoder_block),
              )

              def scanned_fn(y, *args, **kwargs):
                  return scan_fn(y, *args, **kwargs)[0]

              self._scanned_layers = scanned_fn

      else:
          # UNscanned path
          layers = []
          for i in range(cfg.num_decoder_layers):
              layer_kwargs = self._get_layer_kwargs_for_decoder(cfg.decoder_block, layer_number=i)
              layer_cls = self.set_remat_policy([self.decoder_layer[0]], remat_policy)[0]
              layer = layer_cls(
                  config=cfg,
                  mesh=mesh,
                  name=f"layer_{i}",
                  quant=self.quant,
                  model_mode=self.model_mode,
                  **layer_kwargs
              )
              layers.append(layer)

          def unscanned_fn(y, *args, previous_chunk=None, slot=None, page_state=None, bidirectional_mask=None, **kwargs):
              for i, layer in enumerate(layers):
                  call_kwargs = {}
                  if cfg.decoder_block in {DecoderBlockType.GEMMA3, DecoderBlockType.LLAMA4}:
                      call_kwargs["bidirectional_mask"] = bidirectional_mask
                  y = layer(y, *args, previous_chunk=previous_chunk, slot=slot, page_state=page_state, **call_kwargs)
              return y

          self._unscanned_layers = unscanned_fn


  def get_remat_policy(self)-> Optional[Callable[..., bool]]:
    """Get remat policy from configuration."""
    cfg = self.config
    policy_name = cfg.remat_policy

    if policy_name == "none" or policy_name == "full":
      return None
    
    # Static policy mapping
    static_policies : dict[str,Any] = {
        "minimal": jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
        "save_dot_with_context_except_mlp": jax.checkpoint_policies.save_only_these_names(
            "query_proj", "value_proj", "key_proj", "qkv_proj", "context", "out_proj"
        ),
        "save_dot_except_mlpwi": jax.checkpoint_policies.save_only_these_names(
            "query_proj", "value_proj", "key_proj", "qkv_proj", "out_proj", "mlpwo"
        ),
        "save_dot_except_mlp": jax.checkpoint_policies.save_only_these_names(
            "query_proj", "value_proj", "key_proj", "qkv_proj", "out_proj"
        ),
        "save_qkv_proj": jax.checkpoint_policies.save_only_these_names(
            "query_proj", "value_proj", "key_proj", "qkv_proj"
        ),
        "save_out_proj": jax.checkpoint_policies.save_only_these_names("out_proj"),
        "minimal_flash": jax.checkpoint_policies.save_from_both_policies(
            jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
            jax.checkpoint_policies.save_only_these_names("context")
        ),
    }

    # Dynamic policy mapping
    dynamic_policies : dict[str,Callable] = {
        "qkv_proj_offloaded": lambda: jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=[],
            names_which_can_be_offloaded=["query_proj", "value_proj", "key_proj"],
            offload_src="device",
            offload_dst="pinned_host"
        ),
        "minimal_offloaded": lambda: jax.checkpoint_policies.offload_dot_with_no_batch_dims(
            offload_src="device",
            offload_dst="pinned_host"
        ),
        "custom": lambda: jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=cfg.tensors_on_device,
            names_which_can_be_offloaded=cfg.tensors_to_offload,
            offload_src="device",
            offload_dst="pinned_host"
        ),
    }

    if policy_name in static_policies:
      return static_policies[policy_name]
    elif policy_name in dynamic_policies:
      # Call the lambda
      return dynamic_policies[policy_name]()
    raise ValueError(f"Unknown remat policy: '{policy_name}'")

  def get_decoder_layers(self)->list[Type[nnx.Module]]:
    """Retrieves a list of decoder layer classes based on the `decoder_block` config.

    Returns:
        A list containing one or more `nn.Module` classes for the decoder.
    """

    # Mapping of decoder types to their corresponding layers
    decoder_layer_map = {
        DecoderBlockType.DEFAULT: [DecoderLayer],
        DecoderBlockType.LLAMA2: [llama2.LlamaDecoderLayer],
        DecoderBlockType.MISTRAL: [mistral.MistralDecoderLayer],
        DecoderBlockType.MIXTRAL: [mixtral.MixtralDecoderLayer],
        DecoderBlockType.DEEPSEEK: [deepseek.DeepSeekDenseLayer, deepseek.DeepSeekMoELayer],
        DecoderBlockType.GEMMA: [gemma.GemmaDecoderLayer],
        DecoderBlockType.GEMMA2: [gemma2.Gemma2DecoderLayer],
        DecoderBlockType.GEMMA3: [gemma3.Gemma3DecoderLayer],
        DecoderBlockType.GPT3: [gpt3.Gpt3DecoderLayer],
        DecoderBlockType.QWEN3: [qwen3.Qwen3DecoderLayer],
        DecoderBlockType.SIMPLE: [simple_layer.SimpleDecoderLayer],
        DecoderBlockType.SIMPLE_MLP: [simple_layer.SimpleMlpDecoderLayer],
        DecoderBlockType.LLAMA4: (
           [llama4.Llama4ScannableBlock] 
           if self.config.scan_layers 
           else [llama4.Llama4DecoderLayer]
        ),
    }

    decoder_type = self.config.decoder_block

    if decoder_type in decoder_layer_map:
        return decoder_layer_map[decoder_type]

    raise ValueError(f"Incorrect decoder_block name: {decoder_type.value}")

  def set_remat_policy(self, block_layers, policy : Callable[..., bool]|None = None)->list[Type[nnx.Module]]:
    """Set remat policy"""
    RemattedBlockLayers = []
    for block_layer in block_layers:
      if self.config.parameter_memory_host_offload:
        # Define parameter movement with mesh-based sharding
        def move_to_device(variables):
          """Move parameters to device with proper sharding."""

          def map_fn(path, value):
            max_logging.log(f"models.py: Moving parameter {path} to device")
            return jax.device_put(
                value, max_utils.device_space()
            )

          return jax.tree_util.tree_map_with_path(map_fn, variables)

        # Transform layer class before remat
        graphdef, params = nnx.split(block_layer, nnx.Param)
        params = move_to_device(params)
        block_layer = nnx.merge(graphdef, params)

      # Apply remat policy to layer
      layer = nnx.remat(
          block_layer,
          prevent_cse=not self.config.scan_layers,
          policy=policy,
          static_argnums=(4, 5),  # Deterministic and model mode are static arguments.
      )
      RemattedBlockLayers.append(layer)
    return RemattedBlockLayers

  def get_norm_layer(self, num_features: int)-> Callable[...,Any]:
    """get normalization layer (return type inherits from nn.Module)"""
    if self.config.decoder_block in (
        DecoderBlockType.DEFAULT,
        DecoderBlockType.LLAMA2,
        DecoderBlockType.MISTRAL,
        DecoderBlockType.MIXTRAL,
        DecoderBlockType.DEEPSEEK,
        DecoderBlockType.GEMMA,
        DecoderBlockType.GEMMA2,
        DecoderBlockType.GEMMA3,
        DecoderBlockType.QWEN3,
        DecoderBlockType.SIMPLE,
        DecoderBlockType.SIMPLE_MLP,
        DecoderBlockType.LLAMA4,
    ):
      return functools.partial(RMSNorm, num_features=num_features)
    elif self.config.decoder_block == DecoderBlockType.GPT3:
      return functools.partial(gpt3.Gpt3LayerNorm, num_features=num_features, reductions_in_fp32=False, use_bias=True)
    raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block.value=}")

  def scan_decoder_layers(self, cfg:Config, decoder_layer: Callable, length:int, metadata_axis_name:str, mesh:Mesh, in_axes_tuple:Any, model_mode:str, **kwargs):
    """scan decoder layers, calls `flax.linen.transforms.scan`"""
    params_spec = cfg.param_scan_axis
    cache_spec = 0
    scan_fn = nn.scan(
        decoder_layer,
        variable_axes={
            "params": params_spec,
            "cache": cache_spec,
            "intermediates": 0,
            "aqt": 0,
            "_overwrite_with_gradient": 0,
        },
        split_rngs={
            "params": True,
            "dropout": cfg.enable_dropout,
        },
        in_axes=in_axes_tuple,
        length=length,
        metadata_params={nn.PARTITION_NAME: metadata_axis_name},
    )

    return scan_fn(
        config=cfg,
        mesh=mesh,
        name=metadata_axis_name,
        quant=self.quant,
        model_mode=model_mode,
        **kwargs
    )

  def get_pipeline_stage_module(self, decoder_blocks:list[Type[nnx.Module]]) -> nnx.Module:
    """get pipeline stage module"""

    def get_layer_to_pipeline(blocks: list[Type[nnx.Module]], cfg:Config)->Callable[..., nnx.Module]:
      if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
        return blocks[1]  # return the sparse block
      return blocks[0]

    cfg = self.config
    base_stage = get_layer_to_pipeline(decoder_blocks, cfg)
    if cfg.set_remat_policy_on_layers_per_stage:
      policy = self.get_remat_policy()
      base_stage = self.set_remat_policy([base_stage], policy)[0]
    
    if cfg.num_layers_per_pipeline_stage == 1:
      stage_module = base_stage(config=cfg, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode)
    elif cfg.scan_layers_per_stage:
      stage_module = self.scan_decoder_layers(
          cfg,
          base_stage,
          cfg.num_layers_per_pipeline_stage,
          "layers_per_stage",
          self.mesh,
          in_axes_tuple=(nn.broadcast,) * 4,
          model_mode=self.model_mode,
      )
    else:
      stage_module = SequentialBlockDecoderLayers(
          decoder_layer=base_stage,
          num_decoder_layers=cfg.num_layers_per_pipeline_stage,
          config=cfg,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=self.model_mode,
      )
    return stage_module
  
  def _apply_embedding(
      self,
      decoder_input_tokens: Array,
      decoder_positions: Array,
      deterministic: bool,
      model_mode: str,
      image_embeddings: np.ndarray | Array | None = None,
      bidirectional_mask=None,
  ) -> Array:
      """Applies token and positional embeddings to the input tokens."""
      cfg = self.config

      # Token embeddings
      y = self.shared_embedding(
          decoder_input_tokens.astype("int32"),
          model_mode=model_mode
      )

      # Multimodal support
      if image_embeddings is not None and cfg.use_multimodal:
          if cfg.model_name not in {
              "gemma3-4b", "gemma3-12b", "gemma3-27b", 
              "llama4-17b-16e", "llama4-17b-128e"
          }:
              raise ValueError(f"Unsupported model_name for multimodal: {cfg.model_name}")

          y = multimodal_utils.merge_mm_embeddings(
              text_embeddings=y,
              vision_embeddings=image_embeddings,
              mask=bidirectional_mask,
          )

      # Dropout
      y = self.dropout_layer(y, deterministic=deterministic)
      y = y.astype(cfg.dtype)

      # Static positional embedding
      if self.static_pos_embedding is not None:
          y = self.static_pos_embedding(y, decoder_positions)

      # Trainable positional embedding
      if self.trainable_pos_embedding is not None:
          y += self.trainable_pos_embedding(decoder_positions, model_mode=model_mode)

      return y

  def _apply_output_head(self, y:Array, deterministic:bool, model_mode:str)->Array:
    """Applies final normalization and projects hidden states to logits."""

    cfg = self.config
    y = self.get_norm_layer(num_features=y.shape[-1])(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        # name="decoder_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
        parameter_memory_host_offload=cfg.parameter_memory_host_offload,
        rngs=self.rngs
    )(y)
    y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(y, deterministic=deterministic)

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      embedding_table = self.shared_embedding.variables["params"]["embedding"]
      if isinstance(embedding_table, nn.spmd.LogicallyPartitioned):
        embedding_table = embedding_table.unbox()
      attend_dtype = jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype
      logits = attend_on_embedding(y, embedding_table, attend_dtype, self.config)

      if self.config.normalize_embedding_logits:
        # Correctly normalize pre-softmax logits for this shared case.
        logits = logits / jnp.sqrt(y.shape[-1])
      if cfg.final_logits_soft_cap:
        logits = logits / cfg.final_logits_soft_cap
        logits = jnp.tanh(logits) * cfg.final_logits_soft_cap
    else:
      logits = linears.dense_general(
          inputs_shape=y.shape,
          out_features_shape=cfg.vocab_size,
          weight_dtype=cfg.weight_dtype,
          dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
          kernel_axes=("embed", "vocab"),
          name="logits_dense",
          matmul_precision=self.config.matmul_precision,
          parameter_memory_host_offload=cfg.parameter_memory_host_offload,
      )(
          y
      )  # We do not quantize the logits matmul.
    
    # Add logical constraints to logits.
    logical_axis_resource = (
       (None, None, "activation_vocab") 
       if model_mode in {MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE} 
       else ("activation_embed_and_logits_batch", "activation_length", "activation_vocab")
    )
    logits = nn.with_logical_constraint(
        logits, logical_axis_resource
    )

    if self.config.cast_logits_to_fp32:
      logits = logits.astype(jnp.float32)

    return logits

  def __call__(
      self,
      decoder_input_tokens: Array,
      decoder_positions: Array,
      decoder_segment_ids: Array|None=None,
      deterministic:bool=False,
      model_mode:str=MODEL_MODE_TRAIN,
      previous_chunk: Optional[Array]=None,
      slot: Optional[int] = None,
      page_state: Optional[page_manager.PageState] = None,
      bidirectional_mask: Optional[Array] = None,
      image_embeddings: Optional[Array] = None,
  )->tuple[Array,Array]:
      cfg = self.config
      if decoder_input_tokens.ndim != 2: 
        raise ValueError(
            f"`decoder_input_tokens` must have shape [batch, length], "
            f"but got array with shape {decoder_input_tokens.shape}."
        )

      y = self._apply_embedding(
          decoder_input_tokens,
          decoder_positions,
          deterministic,
          model_mode,
          image_embeddings,
          bidirectional_mask,
      )

      # scan does not support kwargs in layer call, passing broadcast_args as positional arg
      broadcast_args = (
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
      )

      if cfg.using_pipeline_parallelism:
          y = self.upper_layer(y, *broadcast_args)
      else:
          y = self._run_without_pipeline_parallelism(
              y,
              broadcast_args,
              previous_chunk,
              slot,
              page_state,
              bidirectional_mask,
          )


      if not isinstance(y, jax.Array):
        raise TypeError(
            f"Expected `y` to be a jax.Array, but got {type(y).__name__}."
        )

      # After the final transformer layer, `y` holds the raw, un-normalized hidden state.
      hidden_state = y

      logits = self._apply_output_head(hidden_state, deterministic, model_mode)

      # The API of the Decoder is now a tuple, providing both the main output
      # and the raw hidden state needed for auxiliary tasks.
      return logits, hidden_state
  
  def _run_without_pipeline_parallelism(
    self,
    y: Array,
    broadcast_args: tuple[Any, ...],
    previous_chunk: Optional[Array] = None,
    slot: Optional[int] = None,
    page_state: Optional[page_manager.PageState] = None,
    bidirectional_mask: Optional[Any] = None,
) -> Array:
    if self._scanned_layers:
        return self._scanned_layers(
            y,
            *broadcast_args,
            previous_chunk=previous_chunk,
            slot=slot,
            page_state=page_state,
            bidirectional_mask=bidirectional_mask,
        )

    for layer in self._unscanned_layers:
        layer_kwargs = {
            "previous_chunk": previous_chunk,
            "slot": slot,
            "page_state": page_state,
        }
        if hasattr(layer, "bidirectional_mask"):
            layer_kwargs["bidirectional_mask"] = bidirectional_mask

        y = layer(y, *broadcast_args, **layer_kwargs)

    return y

  def _init_gemma3_scanned_layers(self):
    cfg = self.config
    mesh = self.mesh
    in_axes_tuple = (nn.broadcast,) * 4
    remat_policy = self.get_remat_policy()

    layers = []
    for i in range(cfg.num_decoder_layers):
        layer = self.decoder_layer[0](
            config=cfg,
            mesh=mesh,
            quant=self.quant,
            model_mode=self.model_mode,
            attention_type=gemma3.get_attention_type(i),
            name=f"layer_{i}",
        )
        layer = self.set_remat_policy(layer, remat_policy)
        layers.append(layer)

    scan_fn = self.scan_decoder_layers(cfg, layers[0], cfg.num_decoder_layers, "layers", mesh, in_axes_tuple, model_mode=self.model_mode)
    self._scanned_layers = lambda y, *args, **kwargs: scan_fn(y, *args, **kwargs)[0]


  @property
  def pipeline_module(self) -> pipeline.Pipeline:
      if self._pipeline_module is None:
          raise RuntimeError("Pipeline module is not initialized. Set 'ici_pipeline_parallelism' or `dcn_pipeline_parallelism` value larger than 1 in config to enable pipeline parallelism.")
      return self._pipeline_module
