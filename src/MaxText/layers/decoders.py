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

from typing import Any, List, Optional, Tuple, Type
import functools

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh

from flax import nnx

from MaxText.common_types import DecoderBlockType, ShardMode, Config, EP_AS_CONTEXT
from MaxText.common_types import MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE
from MaxText import max_logging
from MaxText import max_utils
from MaxText.sharding import create_sharding
from MaxText.inference import page_manager
from MaxText.layers import linears
from MaxText.layers import normalizations

from MaxText.layers import quantizations
from MaxText.layers import pipeline
from MaxText import maxtext_utils
from MaxText import multimodal_utils
from MaxText import sharding
# Assumes these modules are adapted for NNX or compatible wrappers
from MaxText.layers.attentions import Attention 
from flax.linen import partitioning as linen_partitioning
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
  This is the core, reusable building block for both the main model's
  decoder stack and the auxiliary MTP layers.
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

    # Sharding Logic (Kept identical)
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
   
    # === METRICS LOGGING ===
    metrics = {}
    if self.config.record_internal_nn_metrics:
        # Calculate statistics
        metrics['activation_mean'] = jnp.mean(layer_output)
        metrics['activation_stdev'] = jnp.std(layer_output)
        metrics['activation_fraction_zero'] = jnp.sum(layer_output == 0) / jnp.size(layer_output)

    if self.config.scan_layers:
      return layer_output, kv_cache, metrics
    return layer_output, kv_cache, metrics


class SequentialBlockDecoderLayers(nnx.Module):
  """Sequential unscanned series of decoder layers."""

  def __init__(
      self, 
      layers: List[nnx.Module],
      config: Config, 
      mesh: Mesh,
      quant: Quant,
      model_mode: str
  ):
    self.layers = layers
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids,
      decoder_positions,
      deterministic: bool,
      model_mode,
      slot: None | int = None,
      page_state: None | page_manager.PageState = None,
      kv_caches: List[jax.Array] | None = None,
      attention_metadata=None,
  ) -> Tuple[jnp.ndarray, Optional[List[jax.Array]]]:
    
    new_kv_caches = []
    
    for i, layer in enumerate(self.layers):
      kv_cache = kv_caches[i] if kv_caches is not None else None
      
      inputs, new_kv = layer(
          inputs,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          slot=slot,
          page_state=page_state,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata
      )
      
      if new_kv is not None:
        new_kv_caches.append(new_kv)
    
    # Return inputs and list of KV caches if they were updated
    final_kv = new_kv_caches if new_kv_caches else None
    return inputs, final_kv

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
      *,
      rngs: nnx.Rngs
  ):
    self.config = config
    self.num_layers = num_layers
    
    # Generate keys for this stage's layers
    layer_keys = rngs.split(num_layers)

    def create_layer(key):
        return layer_class(
            config=config, mesh=mesh, quant=quant, model_mode=model_mode, rngs=key
        )

    if config.scan_layers_per_stage:
        self.layers = nnx.vmap(create_layer)(layer_keys)
        self.is_scanned = True
    else:
        self.layers = [create_layer(k) for k in layer_keys]
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

    # Handle KV Cache Slicing
    # If kv_caches is provided, we stack it for the scan.
    stage_kv_stack = None
    if kv_caches is not None and self.is_scanned:
         # Stack List[Array] -> Array[Layers, Batch, ...]
         stage_kv_stack = jnp.stack(kv_caches)

    # --- Helper to Call Layer safely ---
    def call_layer(layer_module, y_curr, kv_curr):
        y_out, kv_out, metrics = layer_module(
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
        return y_out, kv_out, metrics

    if self.is_scanned:
        # === SCANNED EXECUTION ===
        layer_graph, layer_params = nnx.split(self.layers)
        
        # Prepare Scan Inputs (Parameters + KV Cache)
        if stage_kv_stack is not None:
             scan_xs = (layer_params, stage_kv_stack)
        else:
             scan_xs = (layer_params, None)

        def scan_fn(carry, x):
            y_in = carry
            params_slice, kv_slice = x # kv_slice is None if scan_xs[1] is None
            
            # Reconstruct Layer
            current_layer = nnx.merge(layer_graph, params_slice)
            
            # Define Forward for Checkpointing
            def layer_forward(y_c, kv_c):
                return call_layer(current_layer, y_c, kv_c)
            
            # Apply Remat
            if self.config.remat_policy != "none":
                 layer_forward = jax.checkpoint(layer_forward)

            y_out, kv_out, metrics = layer_forward(y_in, kv_slice)
            return y_out, (kv_out, metrics)

        # Execute Scan
        if stage_kv_stack is None:
             # Adapter for when we don't have KVs to scan over
             def scan_fn_no_kv(carry, params_slice):
                 return scan_fn(carry, (params_slice, None))
             y, (_, stacked_metrics) = jax.lax.scan(scan_fn_no_kv, y, layer_params)
        else:
             y, (_, stacked_metrics) = jax.lax.scan(scan_fn, y, scan_xs)

    else:
        # === SEQUENTIAL EXECUTION ===
        for i, layer in enumerate(self.layers):
            kv_in = kv_caches[i] if kv_caches else None
            
            def layer_run(y_c, kv_c):
                return call_layer(layer, y_c, kv_c)
            
            if self.config.remat_policy != "none":
                y, _, _ = jax.checkpoint(layer_run)(y, kv_in)
            else:
                y, _, _ = layer_run(y, kv_in)

    # We return ONLY the hidden state 'y' to match the SequentialBlockDecoderLayers 
    # interface expected by the Pipeline module.
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

    self.captured_intermediates = nnx.Intermediate({})

    # 1. Output Norm
    self.norm_layer = self.get_norm_layer(num_features=config.emb_dim, rngs=rngs)(
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        parameter_memory_host_offload=config.parameter_memory_host_offload,
    )       

    # 2. Positional Embeddings
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
    
    # Flags to determine execution path in __call__
    self.using_pipeline = config.using_pipeline_parallelism
    self.is_gemma3 = (config.decoder_block == DecoderBlockType.GEMMA3)
    self.is_scanned = config.scan_layers

    # --- STRATEGY A: PIPELINE PARALLELISM ---
    if self.using_pipeline:
        block_classes = self.get_decoder_layer_class()
        
        # A1. DeepSeek Special Case: Dense Layers run locally (outside pipeline)
        if config.decoder_block == DecoderBlockType.DEEPSEEK:
            self.first_dense_layers = config.first_num_dense_layers
            DenseCls = block_classes[0] # DeepSeekDenseLayer
            
            dense_keys = rngs.split(self.first_dense_layers)
            def create_dense(k):
                 return DenseCls(config=config, mesh=mesh, quant=quant, model_mode=model_mode, rngs=k)
            
            if config.scan_layers:
                self.local_dense_stack = nnx.vmap(create_dense)(dense_keys)
            else:
                self.local_dense_stack = [create_dense(k) for k in dense_keys]
        else:
            self.first_dense_layers = 0
            self.local_dense_stack = None

        # A2. Pipeline Module
        # We use a factory helper to create the stages inside the pipeline
        stage_factory = self.get_pipeline_stage_module(block_classes)
        
        # We assume MaxText.layers.pipeline.Pipeline is NNX-compatible or wrapped
        self.pipeline_module = pipeline.Pipeline(
            config=config,
            mesh=mesh,
            layers=stage_factory,
            remat_policy=self.get_remat_policy(),
            rngs=rngs 
        )

        # A3. Remaining Layers (Post-Pipeline)
        # Calculate layers not covered by Dense stack or Pipeline stages
        pp_layers = config.pipeline_parallel_layers
        remaining_count = config.num_decoder_layers - pp_layers - self.first_dense_layers
        
        if remaining_count > 0:
            # For DeepSeek, these would be MoE layers, or standard for others.
            # Using the last class in the list is a safe heuristic for "main" layers.
            RemCls = block_classes[-1] 
            rem_keys = rngs.split(remaining_count)
            def create_rem(k):
                return RemCls(config=config, mesh=mesh, quant=quant, model_mode=model_mode, rngs=k)
            
            if config.scan_layers:
                self.remaining_stack = nnx.vmap(create_rem)(rem_keys)
            else:
                self.remaining_stack = [create_rem(k) for k in rem_keys]
        else:
            self.remaining_stack = None

    # --- STRATEGY B: GEMMA 3 (Complex Blocking) ---
    elif self.is_gemma3:
        # Import internally to avoid circular dep if needed, or rely on global
        from MaxText.layers import gemma3 
        ScannableBlock = gemma3.Gemma3ScannableBlock
        pattern_len = len(gemma3.GEMMA3_ATTENTION_PATTERN)
        
        num_full_blocks = config.num_decoder_layers // pattern_len
        num_remaining = config.num_decoder_layers % pattern_len
        
        # B1. Main Scanned Blocks
        if num_full_blocks > 0:
            block_keys = rngs.split(num_full_blocks)
            def create_block(k):
                 return ScannableBlock(
                     config=config, mesh=mesh, quant=quant, model_mode=model_mode, 
                     num_of_layers=pattern_len, rngs=k
                 )
            if config.scan_layers:
                 self.gemma_main_stack = nnx.vmap(create_block)(block_keys)
            else:
                 self.gemma_main_stack = [create_block(k) for k in block_keys]
        else:
            self.gemma_main_stack = None

        # B2. Remainder Block
        if num_remaining > 0:
            rem_key = rngs.split(1)[0]
            self.gemma_remainder = ScannableBlock(
                 config=config, mesh=mesh, quant=quant, model_mode=model_mode, 
                 num_of_layers=num_remaining, rngs=rem_key
            )
        else:
            self.gemma_remainder = None

    # --- STRATEGY C: STANDARD & DEEPSEEK (No Pipeline) ---
    else:
        block_classes = self.get_decoder_layer_class()
        self.layer_stacks = []
        self.layer_counts = []

        # Determine counts for heterogeneous layers
        if len(block_classes) == 2 and config.decoder_block == DecoderBlockType.DEEPSEEK:
            # [Dense Layers, MoE Layers]
            self.layer_counts = [
                config.first_num_dense_layers, 
                config.num_decoder_layers - config.first_num_dense_layers
            ]
        else:
            # Homogeneous
            self.layer_counts = [config.num_decoder_layers]
            if len(block_classes) > 1:
                 # Fallback if multiple classes returned (e.g. Scannable wrappers)
                 block_classes = [block_classes[0]]

        # Initialize Stacks
        total_layers = sum(self.layer_counts)
        all_layer_keys = rngs.split(total_layers)
        key_idx = 0
        
        for i, count in enumerate(self.layer_counts):
            if count == 0:
                self.layer_stacks.append(None)
                continue
                
            BlockClass = block_classes[i] if i < len(block_classes) else block_classes[0]
            stack_keys = all_layer_keys[key_idx : key_idx + count]
            key_idx += count

            def create_layer_factory(k):
                 return BlockClass(config=config, mesh=mesh, quant=quant, model_mode=model_mode, rngs=k)

            if config.scan_layers:
                stack = nnx.vmap(create_layer_factory)(stack_keys)
                self.layer_stacks.append(stack)
            else:
                layer_list = [create_layer_factory(stack_keys[k]) for k in range(count)]
                self.layer_stacks.append(layer_list)




  def _scan_single_block(self, layer_stack, init_y, init_kv_stack, broadcast_args):
    """Helper to run jax.lax.scan on a single homogeneous stack of layers."""
    (decoder_segment_ids, decoder_positions, deterministic, model_mode, 
        previous_chunk, slot, page_state, attention_metadata) = broadcast_args

    # 1. Split NNX module
    layer_graph, layer_params = nnx.split(layer_stack)

    # 2. Prepare Scan Inputs
    # If KV cache exists, scan over it. Otherwise scan over None (implicitly handled by zip in python or manual handling)
    # scan_xs: (params, kv_cache)
    scan_xs = (layer_params, init_kv_stack) if init_kv_stack is not None else (layer_params, None)

    # 3. Define Scan Function
    def scan_fn(carry, x):
        y_in = carry
        params_slice, kv_cache_slice = x

        # Reconstruct layer
        current_layer = nnx.merge(layer_graph, params_slice)

        # Define forward pass for checkpointing
        def layer_forward(y_curr, kv_curr):
            out, new_kv, metrics = current_layer(
                y_curr,
                decoder_segment_ids,
                decoder_positions,
                deterministic,
                model_mode,
                previous_chunk=previous_chunk,
                slot=slot,
                page_state=page_state,
                kv_cache=kv_curr,
                attention_metadata=attention_metadata
            )
            return out, new_kv, metrics

        # Apply Checkpoint
        policy = self.get_remat_policy()
        if policy is not None:
            layer_forward = jax.checkpoint(layer_forward, policy=policy)

        # Execute
        y_out, kv_out, metrics = layer_forward(y_in, kv_cache_slice)
        return y_out, kv_out, metrics

    # 4. Run JAX Scan
    # Note: We need to handle the case where scan_xs[1] is None inside scan_fn more gracefully if needed,
    # but JAX scan handles None in xs if structure matches.
    # Actually, JAX scan iterates over leading axis. passing None is problematic if it expects an array.
    # We handle this by conditionally defining scan_xs and scan_fn signature or using a wrapper.
    # Simplest way for this snippet:
    
    if init_kv_stack is None:
            # Specialized scan for no KV
            def scan_fn_no_kv(carry, params_slice):
                return scan_fn(carry, (params_slice, None))
            y_final, (stacked_kv_out, stacked_metrics) = jax.lax.scan(scan_fn_no_kv, init_y, layer_params)
    else:
            y_final, (stacked_kv_out, stacked_metrics) = jax.lax.scan(scan_fn, init_y, scan_xs)

    return y_final, stacked_kv_out, stacked_metrics

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
      kv_caches
  ):
    """Applies Gemma3 scanned decoder blocks, handling main scan and remainders."""
    from MaxText.layers import gemma3
    pattern_len = len(gemma3.GEMMA3_ATTENTION_PATTERN)
    
    # Broadcast args specific to Gemma (includes bidirectional_mask)
    broadcast_args = (
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        previous_chunk,
        slot,
        page_state,
        bidirectional_mask # Note: Gemma3 specifically needs this in the layer call
    )

    all_new_kvs = []

    # === 1. Main Scanned Blocks ===
    if self.gemma_main_stack is not None:
        # Prepare KV Caches: We need to group them by block
        # Input kv_caches is List[Array] of length TotalLayers (or subset)
        # We need to extract the portion for main blocks
        num_main_layers = self.gemma_main_stack.split_sizes[0] * pattern_len # approximation of logic
        # Better: use config
        num_main_layers = (self.config.num_decoder_layers // pattern_len) * pattern_len
        
        stacked_kvs_for_scan = None
        if kv_caches is not None:
            # Take relevant KVs
            main_kvs = kv_caches[:num_main_layers]
            # Stack: [NumMainLayers, Batch, ...]
            full_stack = jnp.stack(main_kvs)
            # Reshape for Blocking: [NumBlocks, PatternLen, Batch, ...]
            # jax.lax.scan will iterate over NumBlocks, passing [PatternLen, Batch...] to each block
            new_shape = (
                num_main_layers // pattern_len, 
                pattern_len, 
                *full_stack.shape[1:]
            )
            stacked_kvs_for_scan = full_stack.reshape(new_shape)

        # Reuse generic scan helper
        # The helper will iterate over self.gemma_main_stack (NumBlocks items)
        # and stacked_kvs_for_scan (NumBlocks items)
        y, new_kv_blocks = self._scan_single_block(
            self.gemma_main_stack, y, stacked_kvs_for_scan, broadcast_args
        )

        if new_kv_blocks is not None:
            # Output is [NumBlocks, PatternLen, Batch, ...]
            # Flatten back to [NumMainLayers, Batch, ...]
            flat_shape = (num_main_layers, *new_kv_blocks.shape[2:])
            flattened_kvs = new_kv_blocks.reshape(flat_shape)
            # Unstack to list
            all_new_kvs.extend([flattened_kvs[i] for i in range(flattened_kvs.shape[0])])

    # === 2. Remainder Block ===
    if self.gemma_remainder is not None:
        # Remainder layers KV
        rem_kvs = None
        if kv_caches is not None:
            # Take the rest
            rem_kvs_list = kv_caches[num_main_layers:]
            if rem_kvs_list:
                # The remainder block expects a stack of KVs for its internal layers
                rem_kvs = jnp.stack(rem_kvs_list)

        # Run the remainder block (It's a single module, not vmapped, so no scan needed over it)
        # However, it might internally scan over its layers? 
        # Typically Gemma3ScannableBlock is a scanned loop itself or a unrolled block.
        # If it's an NNX module representing the block, we just call it.
        
        # Note: If Gemma3ScannableBlock expects specific args, ensure they match.
        # Assuming consistency with _scan_single_block broadcast args unpacking
        
        # We need to adapt the call signature because _scan_single_block unpacks a tuple
        # Here we call directly.
        (seg_ids, pos, det, mode, prev, slt, pg_st, bi_mask) = broadcast_args
        
        y, rem_kv_out = self.gemma_remainder(
            y, seg_ids, pos, det, mode, prev, slt, pg_st, rem_kvs, None # metadata
        )
        # Note: Depending on Gemma3 impl, we might need to pass bi_mask differently
        # For now, assuming standard signature + mask handling
        
        if rem_kv_out is not None:
             # rem_kv_out should be [RemainderLayers, Batch, ...]
             all_new_kvs.extend([rem_kv_out[i] for i in range(rem_kv_out.shape[0])])

    return y, all_new_kvs


  def get_remat_policy(self):
    """Get remat policy compatible with jax.checkpoint."""
    cfg = self.config
    policy = None
    if cfg.remat_policy != "none":
        # NNX is compatible with jax.checkpoint_policies
        if cfg.remat_policy in ("minimal_with_context", "minimal_flash"):
             policy = jax.checkpoint_policies.save_only_these_names(
                "query_proj", "value_proj", "key_proj", "qkv_proj", "out_proj", 
                "mlpwi_0", "mlpwi_1", "mlpwi", "mlpwo", "context"
             )
        elif cfg.remat_policy == "minimal":
             policy = jax.checkpoint_policies.save_only_these_names(
                "query_proj", "value_proj", "key_proj", "qkv_proj", "out_proj", 
                "mlpwi_0", "mlpwi_1", "mlpwi", "mlpwo"
             )
        # ... (Include other policies from original code) ...
        # Defaulting to full checkpointing (None usually implies default in jax.checkpoint)
        # But if explicit policy is needed:
        elif cfg.remat_policy == "full":
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
        DecoderBlockType.QWEN3_MOE,
        DecoderBlockType.QWEN3_NEXT,
        DecoderBlockType.GPT_OSS,
        DecoderBlockType.SIMPLE,
        DecoderBlockType.SIMPLE_MLP,
        DecoderBlockType.LLAMA4,
    ):
      return functools.partial(RMSNorm, num_features=num_features, shard_mode=self.config.shard_mode, rngs=rngs)
    elif self.config.decoder_block == DecoderBlockType.GPT3:
        return functools.partial(gpt3.Gpt3LayerNorm, num_features=num_features, reductions_in_fp32=False, use_bias=True, rngs=rngs)
    raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block.value=}")

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
    # shared_embedding is an NNX module, we call it directly
    y = shared_embedding(decoder_input_tokens.astype("int32"))

    if image_embeddings is not None and cfg.use_multimodal:
       # Multimodal logic identical to original
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
    
    # 1. Final Norm
    y = self.norm_layer(y)
    y = self.dropout(y, deterministic=deterministic)

    # 2. Logits
    if cfg.logits_via_embedding:
        # Access the embedding parameter directly from the NNX module
        # Assuming shared_embedding has a 'embedding' Param or Variable
        embedding_table = shared_embedding.embedding.value 
        
        attend_dtype = jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype
        # Re-use existing util, assuming it handles array inputs
        logits = attend_on_embedding(y, embedding_table, attend_dtype, self.config, None)
        
        if self.config.normalize_embedding_logits:
            logits = logits / jnp.sqrt(y.shape[-1])
        if cfg.final_logits_soft_cap:
            logits = logits / cfg.final_logits_soft_cap
            logits = jnp.tanh(logits) * cfg.final_logits_soft_cap
    else:
        # Separate dense head would need to be initialized in __init__ if used
        # For this example, assuming logits_via_embedding is standard or handled elsewhere
        raise NotImplementedError("Separate logits dense layer requires init in NNX")

    if self.config.cast_logits_to_fp32:
        logits = logits.astype(jnp.float32)
    return logits


  def get_pipeline_stage_module(self, decoder_block_classes):
    """Creates a factory for the pipeline stage module."""
    cfg = self.config

    # Determine which layer class creates the pipeline stages
    if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
      # DeepSeek uses the MoE block (index 1) for the pipeline
      base_stage_cls = decoder_block_classes[1]
    else:
      base_stage_cls = decoder_block_classes[0]

    # Define the factory function
    def stage_factory(rngs_key):
        return PipelineStageBlock(
            config=cfg,
            mesh=self.mesh,
            quant=self.quant,
            model_mode=self.model_mode,
            num_layers=cfg.num_layers_per_pipeline_stage,
            layer_class=base_stage_cls,
            rngs=rngs_key
        )
    return stage_factory


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
    assert decoder_input_tokens.ndim == 2  # [batch, len]

    # Ensure all operations run under the mesh context
    with self.mesh:
        # -------------------------------------------------------------------------
        # 1. Apply Embeddings
        # -------------------------------------------------------------------------
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

        # -------------------------------------------------------------------------
        # 2. Layer Execution Setup
        # -------------------------------------------------------------------------
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

        # -------------------------------------------------------------------------
        # 3. Execution Branching
        # -------------------------------------------------------------------------
        
        # --- BRANCH A: PIPELINE PARALLELISM ---
        if self.using_pipeline:
            # Prepare special axis rules for PP: Treat PP axis as Data Parallel for local layers
            logical_axis_rules_pp_as_dp = sharding.logical_axis_rules_pp_act_as_dp(
                cfg.logical_axis_rules
            )
            
            # Apply context for local layers
            # We use linen_partitioning because 'sharding.maybe_shard_with_logical' reads from it
            with linen_partitioning.axis_rules(logical_axis_rules_pp_as_dp):
                
                # A1. Local Dense Stack (DeepSeek)
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

            # A2. Pipeline Module Execution
            # The Pipeline module handles its own internal partitioning/axis rules
            logical_partition_spec = None
            if cfg.pipeline_fsdp_ag_once:
                logical_partition_spec = self.pipeline_module.get_weight_sharding(
                    y, decoder_segment_ids, decoder_positions, deterministic, model_mode
                )
            
            y = self.pipeline_module(
                y,
                decoder_segment_ids,
                decoder_positions,
                deterministic,
                model_mode,
                logical_partition_spec=logical_partition_spec
            )
            
            # A3. Remaining Layers (Post-Pipeline)
            # Apply context again for remaining local layers
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

        # --- BRANCH B: GEMMA 3 ---
        elif self.is_gemma3:
            res = self._apply_gemma3_scanned_blocks(
                y, decoder_segment_ids, decoder_positions, deterministic, model_mode,
                bidirectional_mask, previous_chunk, page_state, slot, kv_caches
            )
            if len(res) == 3:
                y, new_kv_caches, gemma_mets = res
                collect_metrics(gemma_mets)
            else:
                y, new_kv_caches = res
                
            if new_kv_caches:
                all_new_kvs = new_kv_caches

        # --- BRANCH C: STANDARD / DEEPSEEK ---
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
                        
                        policy = self.get_remat_policy()
                        if policy is not None:
                            y, kv_out, mets = jax.checkpoint(layer_run_explicit, policy=policy)(y, kv_in)
                        else:
                            y, kv_out, mets = layer_run_explicit(y, kv_in)
                        
                        collect_kvs(kv_out)
                        collect_metrics(mets)
                
                kv_start_idx += count

        # -------------------------------------------------------------------------
        # 4. Finalize
        # -------------------------------------------------------------------------
        if len(all_new_kvs) > 0:
            kv_caches = all_new_kvs

        if cfg.record_internal_nn_metrics and all_metrics:
            self.captured_intermediates.value = all_metrics

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


 