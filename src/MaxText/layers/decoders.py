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

"""Module for decoder layers - NNX Version"""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Any, List, Optional, Tuple, Type, Dict
import functools

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh

from flax import nnx
from flax.linen import partitioning as linen_partitioning

from MaxText.common_types import DecoderBlockType, ShardMode, Config, EP_AS_CONTEXT
from MaxText.common_types import MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE
from MaxText import max_logging
from MaxText import max_utils
from MaxText.sharding import create_sharding
from MaxText.inference import page_manager
from MaxText.layers import linears
from MaxText.layers import normalizations
from MaxText.layers import quantizations
from MaxText.layers import pipeline_test as pipeline
from MaxText import maxtext_utils
from MaxText import multimodal_utils
from MaxText import sharding

# Assumes these modules are adapted for NNX
from MaxText.layers.attentions import Attention 
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.embeddings import Embed, attend_on_embedding, PositionalEmbedding
from MaxText.layers.quantizations import AqtQuantization as Quant
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

# ------------------------------------------------------------------------------
# The network: Decoder Definitions
# ------------------------------------------------------------------------------

class DecoderLayer(nnx.Module):
  """
  Transformer decoder layer that attends to the encoder.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: Optional[Quant] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode

    # Initialize Pre-Attention Norm
    self.pre_self_attention_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    # Initialize Attention
    self.self_attention = Attention(
        config=self.config,
        num_query_heads=self.config.num_query_heads,
        num_kv_heads=self.config.num_kv_heads,
        head_dim=self.config.head_dim,
        max_target_length=self.config.max_target_length,
        max_prefill_predict_length=self.config.max_prefill_predict_length,
        attention_kernel=self.config.attention,
        inputs_q_shape=(1, 1, self.config.emb_dim), 
        inputs_kv_shape=(1, 1, self.config.emb_dim),
        mesh=mesh,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        dropout_rate=self.config.dropout_rate,
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(config),
        prefill_cache_axis_order=tuple(map(int, self.config.prefill_cache_axis_order.split(","))),
        ar_cache_axis_order=tuple(map(int, self.config.ar_cache_axis_order.split(","))),
        compute_axis_order=tuple(map(int, self.config.compute_axis_order.split(","))),
        reshape_q=self.config.reshape_q,
        model_mode=model_mode,
        rngs=rngs
    )

    # Initialize MLP
    self.mlp = linears.MlpBlock(
        in_features=self.config.emb_dim,
        intermediate_dim=self.config.mlp_dim,
        activations=self.config.mlp_activations,
        intermediate_dropout_rate=self.config.dropout_rate,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        model_mode=model_mode,
        config=config,
        quant=self.quant,
        mesh=self.mesh,
        rngs=rngs
    )

    # Initialize Dropout
    self.dropout = linears.Dropout(rate=config.dropout_rate, rngs=rngs, broadcast_dims=(-2,))

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
    
    # 1. Norm
    lnx = self.pre_self_attention_norm(inputs)
    lnx = _maybe_shard_with_logical(lnx, logical_axis_names)

    # 2. Attention
    attention_lnx, kv_cache = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    attention_lnx = _maybe_shard_with_logical(attention_lnx, logical_axis_names)

    # 3. MLP
    mlp_lnx = self.mlp(lnx, deterministic=deterministic)
    mlp_lnx = _maybe_shard_with_logical(mlp_lnx, logical_axis_names)

    # 4. Residuals & Dropout
    next_layer_addition = mlp_lnx + attention_lnx
    next_layer_addition_dropped_out = self.dropout(
        next_layer_addition, deterministic=deterministic
    )

    layer_output = next_layer_addition_dropped_out + inputs
    layer_output = _maybe_shard_with_logical(layer_output, logical_axis_names)

    # Metrics Logging
    metrics = {}
    if cfg.record_internal_nn_metrics:
        metrics['activation_mean'] = jnp.mean(layer_output)
        metrics['activation_stdev'] = jnp.std(layer_output)
        metrics['activation_fraction_zero'] = jnp.sum(layer_output == 0) / jnp.size(layer_output)

    return layer_output, kv_cache, metrics


class PipelineStageBlock(nnx.Module):
  """A self-contained block of layers representing a single pipeline stage."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: Optional[Quant],
      model_mode: str,
      num_layers: int,
      layer_class: Type[nnx.Module],
      scan_axis_name: str = "layers_per_stage",
      remat_policy: Any = None, 
      *,
      rngs: nnx.Rngs
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.num_layers = num_layers
    self.remat_policy = remat_policy # Store policy
    
    self.captured_intermediates = nnx.Intermediate([])
    
    #  We must split params and dropout manually.
    layer_param_keys = jax.random.split(rngs.params(), num_layers)
    layer_dropout_keys = jax.random.split(rngs.dropout(), num_layers)

    def create_layer(p_key, d_key):
        # Reconstruct Rngs for the individual layer
        layer_rngs = nnx.Rngs(params=p_key, dropout=d_key)
        return layer_class(
            config=config, mesh=mesh, quant=quant, model_mode=model_mode, rngs=layer_rngs
        )

    if config.scan_layers_per_stage:
        # Pass split keys to vmap
        self.layers = nnx.vmap(
            create_layer,
            transform_metadata={nnx.PARTITION_NAME: scan_axis_name}
        )(layer_param_keys, layer_dropout_keys)
        self.is_scanned = True
    else:
        # Iterate over keys for sequential list
        self.layers = nnx.List([
            create_layer(layer_param_keys[i], layer_dropout_keys[i]) 
            for i in range(num_layers)
        ])
        self.is_scanned = False

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot=None,
      page_state=None,
      kv_caches=None,
      attention_metadata=None,
  ):
    y = inputs
    
    stage_kv_stack = None
    if kv_caches is not None and self.is_scanned:
         stage_kv_stack = jnp.stack(kv_caches)

    def call_layer(layer_module, y_curr, kv_curr):
        return layer_module(
            y_curr,
            decoder_segment_ids=decoder_segment_ids,
            decoder_positions=decoder_positions,
            deterministic=deterministic,
            model_mode=model_mode,
            previous_chunk=previous_chunk,
            slot=slot,
            page_state=page_state,
            kv_cache=kv_curr, 
            attention_metadata=attention_metadata
        )

    all_metrics = []

    if self.is_scanned:
        # === SCANNED EXECUTION ===
        layer_graph, layer_params = nnx.split(self.layers)
        
        scan_xs = (layer_params, stage_kv_stack) if stage_kv_stack is not None else (layer_params, None)

        def scan_fn(carry, x):
            y_in = carry
            params_slice, kv_slice = x
            
            current_layer = nnx.merge(layer_graph, params_slice)
            
            def layer_forward(y_c, kv_c):
                return call_layer(current_layer, y_c, kv_c)
            
            # FIX: Robust Remat Condition
            if self.config.remat_policy != "none":
                 prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)
                 layer_forward = jax.checkpoint(
                     layer_forward, 
                     policy=self.remat_policy,
                     prevent_cse=prevent_cse
                 )

            # FIX: Robust Unpacking
            result = layer_forward(y_in, kv_slice)
            if isinstance(result, tuple) and len(result) == 2:
                y_out, kv_out = result
                mets = None
            else:
                y_out, kv_out, mets = result
            
            # FIX: Dict return
            return y_out, {"kv": kv_out, "metrics": mets}

        if stage_kv_stack is None:
             def scan_fn_no_kv(carry, params_slice):
                 return scan_fn(carry, (params_slice, None))
             y, (_, stacked_metrics) = jax.lax.scan(scan_fn_no_kv, y, layer_params)
        else:
             y, (_, stacked_metrics) = jax.lax.scan(scan_fn, y, scan_xs)

        if stacked_metrics is not None and stacked_metrics.get("metrics") is not None:
             # Depending on structure, stacked_metrics might be None or dict of Nones
             # We accumulate if it looks valid
             all_metrics.append(stacked_metrics)

    else:
        # === SEQUENTIAL EXECUTION ===
        for i, layer in enumerate(self.layers):
            kv_in = kv_caches[i] if kv_caches else None
            
            def layer_run(y_c, kv_c):
                return call_layer(layer, y_c, kv_c)
            
            # FIX: Robust Remat Condition
            if self.config.remat_policy != "none":
                prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)
                result = jax.checkpoint(layer_run, policy=self.remat_policy,prevent_cse=prevent_cse)(y, kv_in)
            else:
                result = layer_run(y, kv_in)
            
            # FIX: Robust Unpacking
            if isinstance(result, tuple) and len(result) == 2:
                y, _, mets = result[0], result[1], None
            else:
                y, _, mets = result
            
            if mets:
                all_metrics.append(mets)

    if self.config.record_internal_nn_metrics and all_metrics:
        self.captured_intermediates.value = all_metrics

    return y
class Decoder(nnx.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: Optional[Quant] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode
    
    # Metrics Container
    self.captured_intermediates = nnx.Intermediate([])

    # 1. Norm & Embeddings
    self.norm_layer = self.get_norm_layer(num_features=config.emb_dim, rngs=rngs)(
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
            embedding_init=nnx.initializers.normal(stddev=1.0),
            config=config,
            mesh=self.mesh,
            rngs=rngs
        )
    else:
        self.position_embedder = None

    self.dropout = linears.Dropout(rate=config.dropout_rate, rngs=rngs, broadcast_dims=(-2,))
    self.positional_embedding = PositionalEmbedding(embedding_dims=config.base_emb_dim)

    # 2. Strategy Flags
    self.using_pipeline = config.using_pipeline_parallelism
    self.is_gemma3 = (config.decoder_block == DecoderBlockType.GEMMA3)
    self.is_scanned = config.scan_layers

    # 3. Layer Initialization
    if self.using_pipeline:
        self._init_pipeline_layers(rngs)
    elif self.is_gemma3:
        self._init_gemma3_layers(rngs)
    else:
        self._init_standard_layers(rngs)

  def _init_pipeline_layers(self, rngs):
    cfg = self.config
    block_classes = self.get_decoder_layer_class()
    
    if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
        self.first_dense_layers = cfg.first_num_dense_layers
        DenseCls = block_classes[0]
        
        # Generate keys
        dense_p_keys = jax.random.split(rngs.params(), self.first_dense_layers)
        dense_d_keys = jax.random.split(rngs.dropout(), self.first_dense_layers)
        
        # Use helper
        self.local_dense_stack = self._create_layer_stack(DenseCls, dense_p_keys, dense_d_keys,scan_axis_name="dense_layers")
    else:
        self.first_dense_layers = 0
        self.local_dense_stack = None

    stage_factory = self.get_pipeline_stage_module(block_classes)
    self.pipeline_module = pipeline.Pipeline(
        config=cfg,
        mesh=self.mesh,
        layers=stage_factory,
        remat_policy=self.get_remat_policy(),
        rngs=rngs 
    )

    # 3. Remaining Layers
    pp_layers = cfg.pipeline_parallel_layers
    remaining_count = cfg.num_decoder_layers - pp_layers - self.first_dense_layers
    
    if remaining_count > 0:
        RemCls = block_classes[-1]
        
        rem_p_keys = jax.random.split(rngs.params(), remaining_count)
        rem_d_keys = jax.random.split(rngs.dropout(), remaining_count)
        
        self.remaining_stack = self._create_layer_stack(RemCls, rem_p_keys, rem_d_keys,scan_axis_name="layers_outside_pipeline")
    else:
        self.remaining_stack = None

  def _init_gemma3_layers(self, rngs):
      from MaxText.layers import gemma3 
      ScannableBlock = gemma3.Gemma3ScannableBlock
      pattern_len = len(gemma3.GEMMA3_ATTENTION_PATTERN)
      
      num_full_blocks = self.config.num_decoder_layers // pattern_len
      num_remaining = self.config.num_decoder_layers % pattern_len
      
      if num_full_blocks > 0:
        # Split streams
        block_param_keys = jax.random.split(rngs.params(), num_full_blocks)
        block_dropout_keys = jax.random.split(rngs.dropout(), num_full_blocks)

        def create_block(p_key, d_key):
                block_rngs = nnx.Rngs(params=p_key, dropout=d_key)
                return ScannableBlock(
                    config=self.config, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode, 
                    num_of_layers=pattern_len, rngs=block_rngs
                )

        if self.config.scan_layers:
                # vmap over keys
                self.gemma_main_stack = nnx.vmap(
                    create_block,
                    transform_metadata={nnx.PARTITION_NAME: "layers"}
                )(block_param_keys, block_dropout_keys)
        else:
                self.gemma_main_stack = [
                    create_block(block_param_keys[i], block_dropout_keys[i]) 
                    for i in range(num_full_blocks)
                ]
      else:
        self.gemma_main_stack = None

      if num_remaining > 0:
        # For a single block, we don't need vmap splitting, just new keys
        rem_rngs = nnx.Rngs(
            params=jax.random.split(rngs.params(), 1)[0],
            dropout=jax.random.split(rngs.dropout(), 1)[0]
        )
        self.gemma_remainder = ScannableBlock(
                config=self.config, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode, 
                num_of_layers=num_remaining, rngs=rem_rngs
        )
      else:
        self.gemma_remainder = None

  def _init_standard_layers(self, rngs):
    cfg = self.config
    block_classes = self.get_decoder_layer_class()
    layer_stacks = []

    is_deepseek = (len(block_classes) == 2 and cfg.decoder_block == DecoderBlockType.DEEPSEEK)
    
    if is_deepseek:
        self.layer_counts = [cfg.first_num_dense_layers, cfg.num_decoder_layers - cfg.first_num_dense_layers]
    else:
        self.layer_counts = [cfg.num_decoder_layers]
        if len(block_classes) > 1: block_classes = [block_classes[0]]

    # 2. Split RNGs for *all* layers upfront (Implementation A's stability fix)
    total_layers = sum(self.layer_counts)
    if total_layers > 0:
        all_param_keys = jax.random.split(rngs.params(), total_layers)
        all_dropout_keys = jax.random.split(rngs.dropout(), total_layers)
    
    key_idx = 0
    
    # 3. Create Stacks
    for i, count in enumerate(self.layer_counts):
        if count == 0:
            layer_stacks.append(None)
            continue
            
        BlockClass = block_classes[i] if i < len(block_classes) else block_classes[0]
        
        # Slice keys for this stack
        stack_param_keys = all_param_keys[key_idx : key_idx + count]
        stack_dropout_keys = all_dropout_keys[key_idx : key_idx + count]
        
        # === LOGIC FOR AXIS NAME ===
        if is_deepseek:
            # Match original Linen logic: 0 -> dense, 1 -> moe
            scan_axis_name = "dense_layers" if i == 0 else "moe_layers"
        else:
            scan_axis_name = "layers"
        # Use the new helper to create the stack/list
        # This integrates the "is_scanned" check logic from Implementation B
        stack = self._create_layer_stack(BlockClass, stack_param_keys, stack_dropout_keys,scan_axis_name)
        
        layer_stacks.append(stack)
        
        key_idx += count

    self.layer_stacks = nnx.List(layer_stacks)


  def _create_layer_stack(self, BlockClass, param_keys, dropout_keys,scan_axis_name="layers"):
        """
        Factory to create either a vmapped stack (for scan) or a list of layers (for loop),
        depending on self.is_scanned.
        
        Args:
            BlockClass: The layer class to instantiate.
            param_keys: A list/array of RNG keys for parameters (one per layer).
            dropout_keys: A list/array of RNG keys for dropout (one per layer).
        """
        cfg = self.config
        
        # Factory function used by both paths
        def create_layer(p_key, d_key):
            layer_rngs = nnx.Rngs(params=p_key, dropout=d_key)
            return BlockClass(
                config=cfg,
                mesh=self.mesh,
                quant=self.quant,
                model_mode=self.model_mode,
                rngs=layer_rngs
            )

        if self.is_scanned:
            # === SCANNED PATH ===
            # Use nnx.vmap to create a module where parameters have a leading 'layers' axis.
            # We pass the split keys directly to the vmapped function.
            stack = nnx.vmap(
                create_layer,
                transform_metadata={nnx.PARTITION_NAME: scan_axis_name} 
            )(param_keys, dropout_keys)
            return stack
        else:
            count = len(param_keys)
            return nnx.List([create_layer(param_keys[i], dropout_keys[i]) for i in range(count)])

  def _minimal_policy(self, with_context=False):
    """Helper for creating minimal checkpoint policies."""
    names = [
        "query_proj",
        "value_proj",
        "key_proj",
        "qkv_proj",
        "out_proj",
        "mlpwi_0",
        "mlpwi_1",
        "mlpwi",
        "mlpwo",
    ]
    if with_context:
      names.append("context")
    return jax.checkpoint_policies.save_only_these_names(*names)

  def get_remat_policy(self):
    """Get remat policy compatible with jax.checkpoint."""
    policy = None
    cfg = self.config
    if cfg.remat_policy != "none":
      if cfg.remat_policy in ("minimal_with_context", "minimal_flash"):
        # save all
        if cfg.remat_policy == "minimal_flash":
          max_logging.log("WARNING: 'minimal_flash' will be deprecated soon, please use 'minimal_with_context' instead.")
        policy = self._minimal_policy(with_context=True)
      elif cfg.remat_policy == "minimal":
        # save all except context
        policy = self._minimal_policy(with_context=False)
      elif cfg.remat_policy == "save_dot_with_context_except_mlp":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj",
            "value_proj",
            "key_proj",
            "qkv_proj",
            "context",
            "out_proj",
        )
      elif cfg.remat_policy == "save_dot_except_mlpwi":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj",
            "value_proj",
            "key_proj",
            "qkv_proj",
            "out_proj",
            "mlpwo",
        )
      elif cfg.remat_policy == "save_dot_except_mlp":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj",
            "value_proj",
            "key_proj",
            "qkv_proj",
            "out_proj",
        )
      elif cfg.remat_policy == "save_qkv_proj":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj",
            "value_proj",
            "key_proj",
            "qkv_proj",
        )
      elif cfg.remat_policy == "qkv_proj_offloaded":
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=[],
            names_which_can_be_offloaded=["query_proj", "value_proj", "key_proj"],
            offload_src="device",
            offload_dst="pinned_host",
        )
      elif cfg.remat_policy == "minimal_offloaded":
        # offload all except context
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=[],
            names_which_can_be_offloaded=[
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
            offload_src="device",
            offload_dst="pinned_host",
        )
      elif cfg.remat_policy == "custom":
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=cfg.tensors_on_device,
            names_which_can_be_offloaded=cfg.tensors_to_offload,
            offload_src="device",
            offload_dst="pinned_host",
        )
      elif cfg.remat_policy == "save_out_proj":
        policy = jax.checkpoint_policies.save_only_these_names(
            "out_proj",
        )
      else:
        assert cfg.remat_policy == "full", "Remat policy needs to be on list of remat policies"
        policy = None
    return policy


  def get_decoder_layer_class(self):
    """Retrieves decoder layer classes based on config using a dictionary lookup."""
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
        DecoderBlockType.GPT_OSS: get_scannable(
            gpt_oss.GptOssDecoderLayer, gpt_oss.GptOssScannableBlock
        ),
        DecoderBlockType.QWEN3_NEXT: get_scannable(
            qwen3.Qwen3NextDecoderLayer, qwen3.Qwen3NextScannableBlock
        ),
        DecoderBlockType.LLAMA4: get_scannable(
            llama4.Llama4DecoderLayer, llama4.Llama4ScannableBlock
        ),
    }

    if cfg.decoder_block not in layer_map:
        raise ValueError(f"Incorrect decoder_block name {cfg.decoder_block.value=}")

    return layer_map[cfg.decoder_block]

  def get_norm_layer(self, num_features: int, rngs: nnx.Rngs):
    if self.config.decoder_block == DecoderBlockType.GPT3:
        return functools.partial(gpt3.Gpt3LayerNorm, num_features=num_features, reductions_in_fp32=False, use_bias=True, rngs=rngs)
    return functools.partial(RMSNorm, num_features=num_features, shard_mode=self.config.shard_mode, rngs=rngs)

  def get_pipeline_stage_module(self, decoder_block_classes):
    cfg = self.config
    base_stage_cls = decoder_block_classes[1] if cfg.decoder_block == DecoderBlockType.DEEPSEEK else decoder_block_classes[0]
    
    # Pre-fetch policy to pass to stage
    policy = self.get_remat_policy()

    def stage_factory(rngs_key):
        return PipelineStageBlock(
            config=cfg,
            mesh=self.mesh,
            quant=self.quant,
            model_mode=self.model_mode,
            num_layers=cfg.num_layers_per_pipeline_stage,
            layer_class=base_stage_cls,
            remat_policy=policy,
            scan_axis_name="layers_per_stage",
            rngs=rngs_key
        )
    return stage_factory

  def _scan_single_block(self, layer_stack, init_y, init_kv_stack, broadcast_args):
        # Split the NNX module into static graph and parameters
        layer_graph, layer_params = nnx.split(layer_stack)
        
        # Prepare the input sequence for scan
        scan_xs = (layer_params, init_kv_stack) if init_kv_stack is not None else (layer_params, None)

        def scan_fn(carry, x):
            y_in = carry
            params_slice, kv_cache_slice = x
            
            # Merge the layer state for this step
            
            def layer_forward(y_curr, kv_curr):
                current_layer = nnx.merge(layer_graph, params_slice)
                
                (decoder_segment_ids, decoder_positions, deterministic, model_mode, 
                 previous_chunk, slot, page_state, attention_metadata) = broadcast_args
                
                return current_layer(
                    y_curr,
                    decoder_segment_ids, decoder_positions, deterministic, model_mode,
                    previous_chunk, slot, page_state, 
                    kv_cache=kv_curr, attention_metadata=attention_metadata
                )

            if self.config.remat_policy != "none":
                policy = self.get_remat_policy()
                # Check if we should prevent CSE (crucial for TPU performance in MaxText)
                prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)
                
                layer_forward = jax.checkpoint(
                    layer_forward, 
                    policy=policy, 
                    prevent_cse=prevent_cse # <--- KEY MISSING ARG
                )

            result = layer_forward(y_in, kv_cache_slice)
            
            # Attempt to unpack based on inspection
            if isinstance(result, tuple) and len(result) == 2:
                y_out, kv_out = result
                mets = None
            elif isinstance(result, tuple) and len(result) == 3:
                y_out, kv_out, mets = result
            else:
                # Let it crash naturally or raise specific error
                raise ValueError(f"Unexpected layer return structure: {result}")
            # --- DEBUG LOGGING END ---
            
            # Return a DICTIONARY for the accumulated values.
            accumulate = {"kv": kv_out, "metrics": mets}
            return y_out, accumulate

        # Run Scan
        if init_kv_stack is None:
             # Wrapper to adapt signature when no KV stack is provided
             def scan_fn_no_kv(carry, params_slice):
                 # We explicitly pass None as the 2nd part of 'x'
                 return scan_fn(carry, (params_slice, None))
             
             scan_result = jax.lax.scan(scan_fn_no_kv, init_y, layer_params)
        else:
             scan_result = jax.lax.scan(scan_fn, init_y, scan_xs)

        # Unpack result safely using dictionary keys
        y_final, stacked_accumulate = scan_result
        
        stacked_kv_out = stacked_accumulate["kv"]
        stacked_metrics = stacked_accumulate["metrics"]

        return y_final, stacked_kv_out, stacked_metrics



  def _apply_embedding(
      self,
      shared_embedding: nnx.Module,
      decoder_input_tokens,
      decoder_positions,
      deterministic,
      model_mode,
      image_embeddings=None,
      bidirectional_mask=None,
      image_masks=None,
  ):
    cfg = self.config
    y = shared_embedding(decoder_input_tokens.astype("int32"))

    if image_embeddings is not None and cfg.use_multimodal:
       y = multimodal_utils.merge_mm_embeddings(
            text_embeddings=y,
            vision_embeddings=image_embeddings,
            mask=bidirectional_mask,
            image_masks=image_masks,
        )

    y = self.dropout(y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    if cfg.use_untrainable_positional_embedding:
      y = self.positional_embedding(y, decoder_positions)

    if self.position_embedder is not None:
      y += self.position_embedder(decoder_positions.astype("int32"))
      
    return y

  def apply_output_head(self, shared_embedding: nnx.Module, y, deterministic, model_mode):
    cfg = self.config
    y = self.norm_layer(y)
    y = self.dropout(y, deterministic=deterministic)

    if cfg.logits_via_embedding:
        embedding_table = shared_embedding.embedding.value 
        attend_dtype = jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype
        logits = attend_on_embedding(y, embedding_table, attend_dtype, self.config, None)
        
        if self.config.normalize_embedding_logits:
            logits = logits / jnp.sqrt(y.shape[-1])
        if cfg.final_logits_soft_cap:
            logits = logits / cfg.final_logits_soft_cap
            logits = jnp.tanh(logits) * cfg.final_logits_soft_cap
    else:
        raise NotImplementedError("Separate logits dense layer requires init in NNX")

    if self.config.cast_logits_to_fp32:
        logits = logits.astype(jnp.float32)
    return logits

  def _apply_gemma3_scanned_blocks(
      self,
      y,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      bidirectional_mask,
      previous_chunk,
      page_state,
      slot,
      kv_caches,
      attention_metadata=None
  ):
    """Applies Gemma3 scanned decoder blocks, handling main scan and remainders."""
    # Import locally to avoid circular dependencies if strict
    from MaxText.layers import gemma3
    pattern_len = len(gemma3.GEMMA3_ATTENTION_PATTERN)
    
    all_new_kvs = []
    all_metrics = []

    # -------------------------------------------------------------------------
    # 1. Main Scanned Blocks (vmapped stack)
    # -------------------------------------------------------------------------
    if self.gemma_main_stack is not None:
        # Determine dimensions by inspecting the parameter state
        # self.gemma_main_stack is vmapped over 'NumBlocks'
        # We peek at a leaf parameter to find the leading axis size (NumBlocks)
        params_example = jax.tree_util.tree_leaves(nnx.state(self.gemma_main_stack))[0]
        num_blocks = params_example.shape[0]
        
        num_main_layers = num_blocks * pattern_len

        # Prepare KV Caches for Scan: [NumBlocks, PatternLen, Batch, ...]
        stacked_kvs_for_scan = None
        if kv_caches is not None:
            # Take the slice corresponding to main layers
            main_kvs_list = kv_caches[:num_main_layers]
            if main_kvs_list:
                full_stack = jnp.stack(main_kvs_list) # [TotalMainLayers, Batch, ...]
                
                # Reshape to [NumBlocks, PatternLen, Batch, ...]
                # jax.lax.scan will iterate over NumBlocks, passing [PatternLen, Batch...] to each step
                new_shape = (num_blocks, pattern_len, *full_stack.shape[1:])
                stacked_kvs_for_scan = full_stack.reshape(new_shape)

        # Split NNX Module into Graph and Parameters
        layer_graph, layer_params = nnx.split(self.gemma_main_stack)
        
        # Prepare Scan Inputs
        scan_xs = (layer_params, stacked_kvs_for_scan) if stacked_kvs_for_scan is not None else (layer_params, None)

        def scan_fn(carry, x):
            y_in = carry
            params_slice, kv_slice = x
            
            # Reconstruct the Block (which contains 'pattern_len' layers)
            current_block = nnx.merge(layer_graph, params_slice)
            
            def block_forward(y_c, kv_c):
                # Gemma3ScannableBlock expects specific args including bidirectional_mask
                return current_block(
                    y_c,
                    decoder_segment_ids=decoder_segment_ids,
                    decoder_positions=decoder_positions,
                    deterministic=deterministic,
                    model_mode=model_mode,
                    previous_chunk=previous_chunk,
                    slot=slot,
                    page_state=page_state,
                    kv_cache=kv_c, # Expects [PatternLen, Batch...]
                    bidirectional_mask=bidirectional_mask,
                    attention_metadata=attention_metadata
                )

            # Apply Checkpointing (Remat)
            policy = self.get_remat_policy()
            if policy is not None:
                block_forward = jax.checkpoint(block_forward, policy=policy)

            # Execution
            # Expecting block to return (y, kv_updates, metrics)
            y_out, kv_out, mets = block_forward(y_in, kv_slice)
            
            # Return (Carry, Accumulate)
            return y_out, (kv_out, mets)

        # Run JAX Scan
        if stacked_kvs_for_scan is None:
             def scan_fn_no_kv(carry, params_slice):
                 return scan_fn(carry, (params_slice, None))
             y, (stacked_kv_out, stacked_metrics) = jax.lax.scan(scan_fn_no_kv, y, layer_params)
        else:
             y, (stacked_kv_out, stacked_metrics) = jax.lax.scan(scan_fn, y, scan_xs)

        # Process Outputs
        if stacked_kv_out is not None:
            # Flatten [NumBlocks, PatternLen, Batch...] -> [TotalMainLayers, Batch...]
            flat_shape = (num_main_layers, *stacked_kv_out.shape[2:])
            flattened_kvs = stacked_kv_out.reshape(flat_shape)
            # Unstack to list
            all_new_kvs.extend([flattened_kvs[i] for i in range(flattened_kvs.shape[0])])
            
        if stacked_metrics:
            # Metrics are also stacked. We append the whole dictionary/structure.
            all_metrics.append(stacked_metrics)

    # -------------------------------------------------------------------------
    # 2. Remainder Block
    # -------------------------------------------------------------------------
    if self.gemma_remainder is not None:
        # Prepare KV for remainder
        rem_kvs_in = None
        if kv_caches is not None:
            # Offset by the number of main layers processed
            offset = 0
            if self.gemma_main_stack is not None:
                params_example = jax.tree_util.tree_leaves(nnx.state(self.gemma_main_stack))[0]
                offset = params_example.shape[0] * pattern_len
            
            rem_list = kv_caches[offset:]
            if rem_list:
                rem_kvs_in = jnp.stack(rem_list)

        def remainder_forward(y_c, kv_c):
            return self.gemma_remainder(
                y_c,
                decoder_segment_ids=decoder_segment_ids,
                decoder_positions=decoder_positions,
                deterministic=deterministic,
                model_mode=model_mode,
                previous_chunk=previous_chunk,
                slot=slot,
                page_state=page_state,
                kv_cache=kv_c,
                bidirectional_mask=bidirectional_mask,
                attention_metadata=attention_metadata
            )
        
        # Apply Checkpointing (Remat)
        policy = self.get_remat_policy()
        if policy is not None:
            y, rem_kv_out, rem_mets = jax.checkpoint(remainder_forward, policy=policy)(y, rem_kvs_in)
        else:
            y, rem_kv_out, rem_mets = remainder_forward(y, rem_kvs_in)

        if rem_kv_out is not None:
             all_new_kvs.extend([rem_kv_out[i] for i in range(rem_kv_out.shape[0])])
        if rem_mets:
             all_metrics.append(rem_mets)

    return y, all_new_kvs, all_metrics
  
  def _consolidate_metrics(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Consolidates a list of metric dicts (chunks) into a single dict of stacked arrays.
    Handles mixing of single-layer results (scalars) and scanned blocks (arrays).
    """
    if not all_metrics:
        return {}
    
    # Identify all metric keys present (e.g., 'activation_mean', 'activation_stdev')
    keys = set()
    for m in all_metrics:
        keys.update(m.keys())
    
    consolidated = {}
    for k in keys:
        chunks = []
        for m in all_metrics:
            if k in m:
                val = m[k]
                # Normalize shapes:
                # If val is a scalar (ndim=0), it comes from a sequential layer -> reshape to (1,)
                # If val is an array (ndim>=1), it comes from a scanned block -> keep as is
                if val.ndim == 0:
                    val = jnp.expand_dims(val, 0)
                chunks.append(val)
        
        if chunks:
            # Concatenate all chunks along the layer axis (axis 0)
            consolidated[k] = jnp.concatenate(chunks, axis=0)
            
    return consolidated


  def __call__(
      self,
      shared_embedding: nnx.Module,
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
    assert decoder_input_tokens.ndim == 2 
    
    with self.mesh:
        y = self._apply_embedding(
            shared_embedding, decoder_input_tokens, decoder_positions, deterministic, 
            model_mode, image_embeddings, bidirectional_mask, image_masks,
        )

        all_new_kvs = []
        all_metrics = []

        def collect_kvs(kvs):
            if kvs is not None:
                if isinstance(kvs, list): all_new_kvs.extend(kvs)
                else: all_new_kvs.append(kvs)

        def collect_metrics(mets):
            if mets and cfg.record_internal_nn_metrics:
                all_metrics.append(mets)

        broadcast_args = (
            decoder_segment_ids, decoder_positions, deterministic, model_mode,
            previous_chunk, slot, page_state, attention_metadata
        )

        if self.using_pipeline:
            logical_axis_rules_pp_as_dp = sharding.logical_axis_rules_pp_act_as_dp(cfg.logical_axis_rules)
            with linen_partitioning.axis_rules(logical_axis_rules_pp_as_dp):
                if self.local_dense_stack is not None:
                    dense_kvs_in = None
                    if kv_caches: 
                        dense_kvs_in = jnp.stack(kv_caches[:self.first_dense_layers]) if self.is_scanned else kv_caches[:self.first_dense_layers]

                    if self.is_scanned:
                        y, dense_kvs_out, dense_mets = self._scan_single_block(
                            self.local_dense_stack, y, dense_kvs_in, broadcast_args
                        )
                        if dense_kvs_out is not None:
                            collect_kvs([dense_kvs_out[i] for i in range(dense_kvs_out.shape[0])])
                        collect_metrics(dense_mets)
                    else:
                        for i, layer in enumerate(self.local_dense_stack):
                            kv_in = dense_kvs_in[i] if dense_kvs_in else None
                            y, kv_out, mets = layer(
                                y, 
                                decoder_segment_ids, decoder_positions, deterministic, model_mode,
                                previous_chunk, slot, page_state, 
                                kv_cache=kv_in, attention_metadata=attention_metadata
                            )
                            collect_kvs(kv_out)
                            collect_metrics(mets)

            logical_partition_spec = None
            if cfg.pipeline_fsdp_ag_once:
                logical_partition_spec = self.pipeline_module.get_weight_sharding(
                    y, decoder_segment_ids, decoder_positions, deterministic, model_mode
                )
            
            y = self.pipeline_module(
                y, decoder_segment_ids, decoder_positions, deterministic, model_mode,
                logical_partition_spec=logical_partition_spec
            )
            
            with linen_partitioning.axis_rules(logical_axis_rules_pp_as_dp):
                if self.remaining_stack is not None:
                    rem_start = self.first_dense_layers + cfg.pipeline_parallel_layers
                    rem_kvs_in = None 
                    if kv_caches:
                        rem_data = kv_caches[rem_start:]
                        rem_kvs_in = jnp.stack(rem_data) if self.is_scanned and rem_data else rem_data

                    if self.is_scanned:
                        y, rem_kvs_out, rem_mets = self._scan_single_block(
                            self.remaining_stack, y, rem_kvs_in, broadcast_args
                        )
                        if rem_kvs_out is not None:
                            collect_kvs([rem_kvs_out[i] for i in range(rem_kvs_out.shape[0])])
                        collect_metrics(rem_mets)
                    else:
                        for i, layer in enumerate(self.remaining_stack):
                            kv_in = rem_kvs_in[i] if rem_kvs_in else None
                            y, kv_out, mets = layer(
                                y, 
                                decoder_segment_ids, decoder_positions, deterministic, model_mode,
                                previous_chunk, slot, page_state, 
                                kv_cache=kv_in, attention_metadata=attention_metadata
                            )
                            collect_kvs(kv_out)
                            collect_metrics(mets)

        elif self.is_gemma3:
            # Call the new helper method
            y, new_kv_caches, gemma_mets = self._apply_gemma3_scanned_blocks(
                y,
                decoder_segment_ids,
                decoder_positions,
                deterministic,
                model_mode,
                bidirectional_mask,
                previous_chunk,
                page_state,
                slot,
                kv_caches,
                attention_metadata=attention_metadata
            )
            
            # Collect outputs
            if gemma_mets:
                # gemma_mets is likely a list of dicts (one for main, one for remainder)
                # Our collect_metrics helper appends single items, so we extend
                if isinstance(gemma_mets, list):
                    all_metrics.extend(gemma_mets)
                else:
                    collect_metrics(gemma_mets)
                
            if new_kv_caches:
                all_new_kvs = new_kv_caches
        else:
            kv_start_idx = 0
            for stack_idx, stack in enumerate(self.layer_stacks):
                if stack is None: continue
                count = self.layer_counts[stack_idx]
                
                current_kvs_list = None
                current_kvs_stack = None
                if kv_caches is not None:
                    current_kvs_list = kv_caches[kv_start_idx : kv_start_idx + count]
                    if self.is_scanned and len(current_kvs_list) > 0:
                        current_kvs_stack = jnp.stack(current_kvs_list)

                if self.is_scanned:
                    y, new_kv_stack, stack_mets = self._scan_single_block(
                        stack, y, current_kvs_stack, broadcast_args
                    )
                    if new_kv_stack is not None:
                        collect_kvs([new_kv_stack[i] for i in range(new_kv_stack.shape[0])])
                    collect_metrics(stack_mets)
                else:
                    for i, layer in enumerate(stack):
                        kv_in = current_kvs_list[i] if current_kvs_list else None
                        def layer_run_explicit(y_c, kv_c):
                            return layer(
                                y_c, 
                                decoder_segment_ids, decoder_positions, deterministic, 
                                model_mode, previous_chunk, slot, page_state, 
                                kv_cache=kv_c, attention_metadata=attention_metadata
                            )
                        

                        if cfg.remat_policy != "none":
                            policy = self.get_remat_policy()
                            prevent_cse = maxtext_utils.should_prevent_cse_in_remat(cfg)
                            
                            result = jax.checkpoint(
                                layer_run_explicit, 
                                policy=policy,
                                prevent_cse=prevent_cse
                            )(y, kv_in)
                        else:
                            result = layer_run_explicit(y, kv_in)

                        if isinstance(result, tuple) and len(result) == 2:
                            y, kv_out = result
                            mets = None
                        else:
                            y, kv_out, mets = result
                        collect_kvs(kv_out)
                        collect_metrics(mets)
                
                kv_start_idx += count

        if len(all_new_kvs) > 0:
            kv_caches = all_new_kvs

        # Consolidate and Store Metrics
        if cfg.record_internal_nn_metrics and all_metrics:
            # Convert list of potentially jagged chunks into unified stacked arrays
            # e.g., [{'mean': [L1_val]}, {'mean': [L2..L32_vals]}] -> {'mean': [L1..L32_vals]}
            consolidated_metrics = self._consolidate_metrics(all_metrics)
            
            # Store in NNX Intermediate variable
            # Downstream code can access this via: model.captured_intermediates.value
            self.captured_intermediates.value = consolidated_metrics


        assert isinstance(y, jax.Array)
        hidden_state = y
        logits = None

        if cfg.attention == "vllm_rpa":
            logits = None
        elif cfg.num_vocab_tiling > 1 and self.model_mode == MODEL_MODE_TRAIN:
            logits = None
        else:
            logits = self.apply_output_head(shared_embedding, hidden_state, deterministic, model_mode)

        return logits, hidden_state, kv_caches