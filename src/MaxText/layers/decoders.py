"""Module for decoder layers - NNX Version"""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Any, Callable, List, Optional
import functools
import inspect

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh

from flax import linen as nn
from flax import nnx

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
from MaxText.layers.attentions import Attention
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.embeddings import Embed, attend_on_embedding, PositionalEmbedding
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.layers import nnx_wrappers

# Import specific layer definitions
from MaxText.layers import (
    gpt3,
    llama2,
    mistral,
    mixtral,
    gemma,
    gemma2,
    gemma3,
    deepseek,
    qwen3,
    simple_layer,
    llama4,
    gpt_oss,
)

class DecoderLayer(nnx.Module):
    """
    Transformer decoder layer that attends to the encoder (NNX).
    """
    def __init__(
        self,
        config: Config,
        mesh: Mesh,
        model_mode: str,
        rngs: nnx.Rngs,
        quant: None | Quant = None,
        name: str = "decoder_layer",
        **kwargs
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
            rngs=rngs
        )

        # Handle specific layer arguments (Llama4, Qwen3, etc.) passed via kwargs
        attn_kwargs = {}
        if "is_nope_layer" in kwargs:
             attn_kwargs["is_nope_layer"] = kwargs["is_nope_layer"]
        if "layer_idx" in kwargs: 
             pass 

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
            **attn_kwargs
        )

        mlp_kwargs = {}
        if "is_moe_layer" in kwargs:
             mlp_kwargs["is_moe_layer"] = kwargs["is_moe_layer"]

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
            **mlp_kwargs
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
        **kwargs
    ):
        cfg = self.config
        mesh = self.mesh
        _maybe_shard_with_logical = functools.partial(
            sharding.maybe_shard_with_logical,
            mesh=mesh,
            shard_mode=cfg.shard_mode,
            debug_sharding=cfg.debug_sharding,
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

        attn_call_kwargs = {}
        if "bidirectional_mask" in kwargs:
             attn_call_kwargs["bidirectional_mask"] = kwargs["bidirectional_mask"]

        attention_lnx, kv_cache = self.self_attention(
            lnx,
            lnx,
            decoder_positions,
            decoder_segment_ids=decoder_segment_ids,
            deterministic=deterministic,
            model_mode=model_mode,
            kv_cache=kv_cache,
            attention_metadata=attention_metadata,
            **attn_call_kwargs
        )
        attention_lnx = _maybe_shard_with_logical(attention_lnx, logical_axis_names)

        mlp_lnx = self.mlp(lnx, deterministic=deterministic)
        mlp_lnx = _maybe_shard_with_logical(mlp_lnx, logical_axis_names)

        next_layer_addition = mlp_lnx + attention_lnx
        next_layer_addition_dropped_out = self.dropout(
            next_layer_addition, deterministic=deterministic
        )

        layer_output = next_layer_addition_dropped_out + inputs
        layer_output = _maybe_shard_with_logical(layer_output, logical_axis_names)

        if cfg.record_internal_nn_metrics:
           self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
           self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
           self.sow(
               "intermediates",
               "activation_fraction_zero",
               jnp.sum(layer_output == 0) / jnp.size(layer_output),
           ) 

        if cfg.scan_layers:
            return layer_output, None
        else:
            return layer_output, kv_cache


class SequentialDecoderLayers(nnx.Module):
    """Sequential unscanned series of decoder layers (NNX)."""
    def __init__(self, layer_class, num_layers, config, mesh, model_mode, rngs, **kwargs):
        layers_list = []
        for i in range(num_layers):
            layers_list.append(
                layer_class(config=config, mesh=mesh, model_mode=model_mode, rngs=rngs, **kwargs)
            )
        self.layers = nnx.List(layers_list)

    def __call__(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.layers:
            out = layer(x, *args, **kwargs)
            if isinstance(out, tuple):
                x = out[0]
            else:
                x = out
        return x, None


class ScannedDecoderLayers(nnx.Module):
    """Scanned series of decoder layers (NNX)."""
    def __init__(self, layer_class, num_layers, config, mesh, model_mode, rngs, **kwargs):
        self.config = config
        
        if hasattr(rngs, 'params'):
            nnx.split_rngs(rngs, splits=num_layers)
        
        def create_layer(rng):
            return layer_class(config=config, mesh=mesh, model_mode=model_mode, rngs=rng, **kwargs)
        
        self.layers_stack = nnx.vmap(
            create_layer, in_axes=0, out_axes=0, axis_name="layers"
        )(rngs)

    def __call__(self, inputs, *args, **kwargs):
        graphdef, state = nnx.split(self.layers_stack)
        
        def scan_fn(carry, layer_state):
            layer = nnx.merge(graphdef, layer_state)
            out = layer(carry, *args, **kwargs)
            if isinstance(out, tuple):
                new_carry, _ = out
            else:
                new_carry = out
            return new_carry, nnx.state(layer)

        final_carry, new_stack_state = jax.lax.scan(
            scan_fn, inputs, state, length=len(state.params['pre_self_attention_norm']['scale'])
        )
        
        nnx.update(self.layers_stack, new_stack_state)
        return final_carry, None


class Decoder(nnx.Module):
    """A stack of decoder layers (NNX)."""

    def __init__(
        self,
        config: Config,
        mesh: Mesh,
        rngs: nnx.Rngs,
        quant: None | Quant = None,
        model_mode: str = MODEL_MODE_TRAIN,
    ):
        self.config = config
        self.mesh = mesh
        self.quant = quant
        self.model_mode = model_mode
        self.rngs = rngs
        
        self.decoder_norm = self.get_norm_layer(num_features=config.emb_dim, rngs=rngs)(
            dtype=config.dtype,
            weight_dtype=config.weight_dtype,
            epsilon=config.normalization_layer_epsilon,
            kernel_axes=("norm",),
            parameter_memory_host_offload=config.parameter_memory_host_offload,
            rngs=rngs,
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
        else:
          self.position_embedder = None
           
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
              rngs=rngs
            )

        self.scanned_layers = None
        self.is_deepseek = (self.config.decoder_block == DecoderBlockType.DEEPSEEK)
        self.decoder_block_classes = self.get_decoder_layers()

        if self.config.using_pipeline_parallelism:
            self.pipeline_module = self.get_pipeline_stage_module(self.decoder_block_classes)
            
        elif self.config.scan_layers:
            if self.is_deepseek:
                dense_cls, moe_cls = self.decoder_block_classes
                self.dense_stack = self._create_scanned_layers(
                    dense_cls, length=config.first_num_dense_layers, rngs=rngs
                )
                self.moe_stack = self._create_scanned_layers(
                    moe_cls, length=(config.num_decoder_layers - config.first_num_dense_layers), rngs=rngs
                )
            elif self.config.decoder_block == DecoderBlockType.GEMMA3:
                 self.gemma3_block_cls = self.decoder_block_classes[0]
                 pass
            else:
                self.layers = self._create_scanned_layers(
                    self.decoder_block_classes[0], length=config.num_decoder_layers, rngs=rngs
                )
        else:
            layers_list = []
            if self.is_deepseek:
                dense_cls, moe_cls = self.decoder_block_classes
                for i in range(config.first_num_dense_layers):
                    layers_list.append(self._create_single_layer(dense_cls, rngs, name=f"dense_layer_{i}"))
                for i in range(config.num_decoder_layers - config.first_num_dense_layers):
                    layers_list.append(self._create_single_layer(moe_cls, rngs, name=f"moe_layer_{i}"))
            else:
                layer_cls = self.decoder_block_classes[0]
                for i in range(config.num_decoder_layers):
                    kwargs = {}
                    if self.config.decoder_block == DecoderBlockType.LLAMA4:
                        kwargs = {
                            "is_nope_layer": llama4.determine_is_nope_layer(i, self.config.nope_layer_interval),
                            "is_moe_layer": llama4.determine_is_moe_layer(i, self.config.interleave_moe_layer_step),
                        }
                    if self.config.decoder_block == DecoderBlockType.QWEN3_NEXT:
                        kwargs = {"layer_idx": i}
                    if self.config.decoder_block == DecoderBlockType.GPT_OSS:
                         kwargs = {"attention_type": gpt_oss.get_attention_type(layer_id=i)}

                    layers_list.append(self._create_single_layer(layer_cls, rngs, name=f"layers_{i}", **kwargs))
            
            self.layers = nnx.List(layers_list)

    def get_pipeline_stage_module(self, decoder_blocks):
        """Creates the Pipeline module with the correct stage configuration."""
        cfg = self.config
        
        def get_layer_to_pipeline(blocks, cfg):
            if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
                return blocks[1]  
            else:
                return blocks[0]

        base_stage_cls = get_layer_to_pipeline(decoder_blocks, cfg)
        
        # FIX: Fork RNGs before passing to stage module to avoid sharing with Pipeline
        stage_rngs = self.rngs.fork() if hasattr(self.rngs, 'fork') else nnx.Rngs(params=self.rngs.params())

        if cfg.num_layers_per_pipeline_stage == 1:
            stage_module = self._create_single_layer(base_stage_cls, stage_rngs)
        elif cfg.scan_layers_per_stage:
            stage_module = ScannedDecoderLayers(
                base_stage_cls, 
                num_layers=cfg.num_layers_per_pipeline_stage,
                config=cfg,
                mesh=self.mesh,
                model_mode=self.model_mode,
                rngs=stage_rngs
            )
        else:
            stage_module = SequentialDecoderLayers(
                base_stage_cls,
                num_layers=cfg.num_layers_per_pipeline_stage,
                config=cfg,
                mesh=self.mesh,
                model_mode=self.model_mode,
                rngs=stage_rngs
            )
        
        return pipeline.Pipeline(
            config=cfg,
            layers=stage_module,
            mesh=self.mesh,
            remat_policy=self.get_remat_policy(),
            rngs=self.rngs # Pipeline keeps original RNGs
        )

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
        """Creates a VMapped stack of layers."""
        if hasattr(rngs, 'params'):
            nnx.split_rngs(rngs, splits=length)
        
        def create_layer_fn(rng):
            return decoder_layer_class(
                config=self.config,
                mesh=self.mesh,
                quant=self.quant,
                model_mode=self.model_mode,
                rngs=rng,
                **layer_kwargs
            )
            
        return nnx.vmap(
          create_layer_fn,
          in_axes=0, out_axes=0, axis_name="layers",
        )(rngs)
    
    def _apply_layers_sequentially(self, layers_stack, x_in, *args, length: int, **kwargs):
      """Runs the layer stack using jax.lax.scan."""
      policy = self.get_remat_policy()
      prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)
      graphdef, state = nnx.split(layers_stack)
      
      def scan_fn(carry, layer_state_slice):
          layer = nnx.merge(graphdef, layer_state_slice)
          
          if self.config.parameter_memory_host_offload:
              pass

          out = layer(carry, *args, **kwargs)
          if isinstance(out, tuple):
              new_carry, _ = out
          else:
              new_carry = out
          return new_carry, nnx.state(layer)

      if policy:
          scan_fn = jax.checkpoint(scan_fn, policy=policy, prevent_cse=prevent_cse)
      
      final_carry, scanned_state = jax.lax.scan(
          scan_fn, x_in, state, length=length
      )
      nnx.update(layers_stack, scanned_state)
      return final_carry, None

    def get_decoder_layers(self):
      """Retrieves decoder layer classes based on config."""
      cfg = self.config
      layer_map = {
          DecoderBlockType.DEFAULT: [DecoderLayer],
          DecoderBlockType.GPT3: [gpt3.Gpt3DecoderLayer], 
      }
      if cfg.decoder_block not in layer_map:
           return [DecoderLayer]
      return layer_map[cfg.decoder_block]

    def get_remat_policy(self):
      policy = None
      cfg = self.config
      if cfg.remat_policy == "minimal":
          return jax.checkpoint_policies.save_only_these_names("query_proj", "value_proj", "key_proj", "qkv_proj", "out_proj")
      return policy 
    
    def get_norm_layer(self, num_features: int, rngs: nnx.Rngs):
      if self.config.decoder_block == DecoderBlockType.GPT3:
          return functools.partial(gpt3.Gpt3LayerNorm, num_features=num_features, reductions_in_fp32=False, use_bias=True, rngs=rngs)
      return functools.partial(RMSNorm, num_features=num_features, shard_mode=self.config.shard_mode, rngs=rngs)

    def _apply_embedding(self, shared_embedding, decoder_input_tokens, decoder_positions, deterministic, model_mode, image_embeddings=None, bidirectional_mask=None, image_masks=None):
        cfg = self.config
        y = shared_embedding(decoder_input_tokens.astype("int32"), model_mode=model_mode)
        
        if image_embeddings is not None and cfg.use_multimodal:
            y = multimodal_utils.merge_mm_embeddings(y, image_embeddings, bidirectional_mask, image_masks)

        y = self.dropout(y, deterministic=deterministic)
        y = y.astype(cfg.dtype)

        if cfg.use_untrainable_positional_embedding:
            y = self.positional_embedding(y, decoder_positions)

        if cfg.trainable_position_size > 0 and self.position_embedder:
            y += self.position_embedder(decoder_positions.astype("int32"), model_mode=model_mode)
        return y

    def apply_output_head(self, shared_embedding, y, deterministic, model_mode):
        cfg = self.config
        norm_out_sharding = None
        if cfg.shard_mode == ShardMode.EXPLICIT:
            norm_out_sharding = create_sharding(self.mesh, ("activation_batch", "activation_length_no_exp", "activation_embed"))

        y = self.decoder_norm(y, out_sharding=norm_out_sharding)
        y = self.dropout(y, deterministic=deterministic)

        out_sharding = None
        if cfg.shard_mode == ShardMode.EXPLICIT:
             if model_mode in (MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE):
                out_sharding = create_sharding(self.mesh, (None, None, "activation_vocab"))
             else:
                out_sharding = create_sharding(self.mesh, ("activation_embed_and_logits_batch", "activation_length_no_exp", "activation_vocab"))

        if cfg.logits_via_embedding:
            if isinstance(shared_embedding, nnx.Module):
                embedding_table = shared_embedding.embedding.value
            elif hasattr(shared_embedding, 'variables'): 
                embedding_table = shared_embedding.variables["params"]["embedding"]
            else:
                embedding_table = shared_embedding.embedding.value 
            
            if isinstance(embedding_table, nn.spmd.LogicallyPartitioned):
                embedding_table = embedding_table.unbox()
            
            attend_dtype = jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype
            logits = attend_on_embedding(y, embedding_table, attend_dtype, self.config, out_sharding)

            if self.config.normalize_embedding_logits:
                logits = logits / jnp.sqrt(y.shape[-1])
            if cfg.final_logits_soft_cap:
                logits = logits / cfg.final_logits_soft_cap
                logits = jnp.tanh(logits) * cfg.final_logits_soft_cap
        else:
            logits = self.logits_dense(y) 
        
        if self.config.cast_logits_to_fp32:
            logits = logits.astype(jnp.float32)
        return logits
    
    def _apply_gemma3_scanned_blocks(self, y, decoder_segment_ids, decoder_positions, deterministic, model_mode, bidirectional_mask, previous_chunk, page_state, slot):
        """Applies Gemma3 scanned decoder blocks (NNX Port)."""
        cfg = self.config
        attention_pattern_length = len(gemma3.GEMMA3_ATTENTION_PATTERN)
        scan_length = cfg.num_decoder_layers // attention_pattern_length
        Gemma3Block = self.gemma3_block_cls 
        
        if scan_length > 0:
            if not self.scanned_layers:
                 if hasattr(self.rngs, 'params'):
                     nnx.split_rngs(self.rngs, splits=scan_length)
                     
                 def create_block(rng):
                     return Gemma3Block(
                         config=cfg, mesh=self.mesh, quant=self.quant, model_mode=model_mode, rngs=rng,
                         num_of_layers=attention_pattern_length
                     )
                 
                 self.scanned_layers = nnx.vmap(create_block, in_axes=0, out_axes=0)(self.rngs)

            layer_kwargs = {"bidirectional_mask": bidirectional_mask}
            y, _ = self._apply_layers_sequentially(
                 self.scanned_layers, 
                 y, 
                 decoder_segment_ids, 
                 decoder_positions, 
                 deterministic, 
                 model_mode, 
                 length=scan_length,
                 **layer_kwargs
            )
            
        num_remaining = cfg.num_decoder_layers % attention_pattern_length
        if num_remaining > 0:
             remainder_block = Gemma3Block(
                  config=cfg, mesh=self.mesh, quant=self.quant, model_mode=model_mode, rngs=self.rngs,
                  num_of_layers=num_remaining, name="layers_remainder"
             )
             y, _ = remainder_block(
                  y,
                  decoder_segment_ids,
                  decoder_positions,
                  deterministic,
                  model_mode,
                  previous_chunk,
                  page_state,
                  slot,
                  bidirectional_mask=bidirectional_mask
             )
        return y

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

        layer_args = (
             decoder_segment_ids,
             decoder_positions,
             deterministic,
             model_mode
        )
        layer_kwargs = {
            "previous_chunk": previous_chunk,
            "page_state": page_state,
            "slot": slot,
            "attention_metadata": attention_metadata,
        }
        
        if cfg.using_pipeline_parallelism:
            if cfg.pipeline_fsdp_ag_once:
                 logical_partition_spec = None 
            else:
                 logical_partition_spec = None
            
            y = self.pipeline_module(
                y, 
                *layer_args, 
                logical_partition_spec=logical_partition_spec
            )

        elif cfg.scan_layers:
            if self.is_deepseek:
                y, _ = self._apply_layers_sequentially(
                    self.dense_stack, y, *layer_args, length=cfg.first_num_dense_layers, **layer_kwargs
                )
                y, _ = self._apply_layers_sequentially(
                    self.moe_stack, y, *layer_args, length=(cfg.num_decoder_layers - cfg.first_num_dense_layers), **layer_kwargs
                )
            elif cfg.decoder_block == DecoderBlockType.GEMMA3:
                 y = self._apply_gemma3_scanned_blocks(
                      y, decoder_segment_ids, decoder_positions, deterministic, model_mode, bidirectional_mask, previous_chunk, page_state, slot
                 )
            else:
                if cfg.decoder_block == DecoderBlockType.LLAMA4:
                    pass
                
                y, _ = self._apply_layers_sequentially(
                    self.layers, y, *layer_args, length=cfg.num_decoder_layers, **layer_kwargs
                )
                
        else:
            for i, layer in enumerate(self.layers):
                kv_cache = kv_caches[i] if kv_caches is not None else None
                
                call_kwargs = layer_kwargs.copy()
                if cfg.decoder_block == DecoderBlockType.GEMMA3:
                     call_kwargs["bidirectional_mask"] = bidirectional_mask

                out = layer(y, *layer_args, kv_cache=kv_cache, **call_kwargs)
                
                if isinstance(out, tuple):
                    y, kv_cache_out = out
                else:
                    y = out
                    kv_cache_out = None
                if kv_caches is not None:
                    kv_caches[i] = kv_cache_out

        assert isinstance(y, jax.Array)
        hidden_state = y

        if cfg.num_vocab_tiling > 1 and self.model_mode == MODEL_MODE_TRAIN:
            logits = None
        else:
            logits = self.apply_output_head(shared_embedding, hidden_state, deterministic, model_mode)

        return logits, hidden_state, kv_caches

def decoder_as_linen(
    config: Config,
    mesh: Mesh,
    rngs: nnx.Rngs,
    model_mode: str,
    quant: None | Quant = None,
):
  """Creates a Decoder module wrapped as Linen."""
  module = nnx_wrappers.to_linen(
      Decoder,
      config=config,
      mesh=mesh,
      model_mode=model_mode,
      rngs=rngs,
      quant=quant,
      name="decoder",
      abstract_init=False,
      metadata_fn=initializers.variable_to_logically_partitioned,
  )
  return module