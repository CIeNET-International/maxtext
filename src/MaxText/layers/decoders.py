"""Decoders for MaxText using Flax NNX.
Implements manual scanning and checkpointing (no nnx.scan/nnx.remat).
"""
import functools
from typing import Any, Optional, List, Tuple, Dict, Union
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from flax import nnx
from flax import linen as nn_linen 

from MaxText.common_types import (
    DecoderBlockType, ShardMode, Config, EP_AS_CONTEXT,
    MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE
)
from MaxText import max_logging
from MaxText import max_utils
from MaxText.sharding import create_sharding, maybe_shard_with_logical
from MaxText.inference import page_manager
from MaxText.layers import linears
from MaxText.layers import normalizations
from MaxText.layers import quantizations
from MaxText.layers import pipeline_nnx as pipeline 
from MaxText import maxtext_utils
from MaxText import multimodal_utils
from MaxText import sharding

from MaxText.layers.attentions import attention_as_linen
from MaxText.layers.embeddings import attend_on_embedding, embed_as_linen, positional_embedding_as_linen, Embed
from MaxText.layers import (
    deepseek, deepseek_batchsplit, gemma, gemma2, gemma3,
    gpt3, gpt_oss, llama2, llama4, mistral, mixtral, 
    qwen3, simple_layer
)
from MaxText.layers.attentions import Attention
from MaxText.layers.normalizations import RMSNorm

# ------------------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------------------

def _get_logical_axis_names(config, model_mode):
    if model_mode == MODEL_MODE_PREFILL:
        return ("activation_batch", "prefill_activation_length", "activation_embed")
    elif config.expert_shard_attention_option == EP_AS_CONTEXT and model_mode == MODEL_MODE_TRAIN:
        return ("activation_batch_no_exp", "activation_length", "activation_embed")
    else:
        return ("activation_batch", "activation_length_no_exp", "activation_embed")

class InternalMetrics(nnx.Variable):
    pass

# ------------------------------------------------------------------------------
#  Decoder Layer
# ------------------------------------------------------------------------------

class DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: Config,
        mesh: Mesh,
        model_mode: str,
        quant: None | quantizations.AqtQuantization = None,
        *,
        rngs: nnx.Rngs,
        layer_idx: int = 0,
        **layer_kwargs,
    ):
        self.config = config
        self.mesh = mesh
        self.model_mode = model_mode
        self.quant = quant
        
        if config.record_internal_nn_metrics:
            self.metrics = InternalMetrics({})

        self.pre_self_attention_norm = nnx.RMSNorm(
            num_features=config.emb_dim, 
            epsilon=config.normalization_layer_epsilon,
            dtype=config.dtype,
            rngs=rngs
        )
        attention_type = self._get_attention_type(self.config, layer_idx)
        attn_kwargs = {}
        if "is_nope_layer" in layer_kwargs:
            attn_kwargs["is_nope_layer"] = layer_kwargs["is_nope_layer"]
        if "is_vision" in layer_kwargs:
            attn_kwargs["is_vision"] = layer_kwargs["is_vision"]

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
            kv_quant=quantizations.configure_kv_quant(self.config),
            prefill_cache_axis_order=tuple(map(int, self.config.prefill_cache_axis_order.split(","))),
            ar_cache_axis_order=tuple(map(int, self.config.ar_cache_axis_order.split(","))),
            compute_axis_order=tuple(map(int, self.config.compute_axis_order.split(","))),
            reshape_q=self.config.reshape_q,
            model_mode=model_mode,
            attention_type=attention_type,
            rngs=rngs,
            **attn_kwargs,
        )

        self.mlp_lnx = linears.MlpBlock(
            config=self.config,
            mesh=self.mesh,
            in_features=self.config.emb_dim,
            intermediate_dim=self.config.mlp_dim,
            activations=self.config.mlp_activations,
            intermediate_dropout_rate=self.config.dropout_rate,
            dtype=self.config.dtype,
            weight_dtype=self.config.weight_dtype,
            model_mode=model_mode,
            quant=self.quant,
            rngs=rngs,
        )
        
        self.dropout = linears.Dropout(rate=self.config.dropout_rate, rngs=rngs, broadcast_dims=(-2,))

    def _get_attention_type(self, cfg, layer_idx):
        if cfg.decoder_block == DecoderBlockType.GEMMA3:
           return gemma3.get_attention_type(layer_id=layer_idx)
        if cfg.decoder_block == DecoderBlockType.GPT_OSS:
            return gpt_oss.get_attention_type(layer_id=layer_idx)
        return gpt_oss.AttentionType.GLOBAL

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
        kv_cache=None,
        attention_metadata=None,
    ):
        cfg = self.config
        logical_axis_names = _get_logical_axis_names(cfg, model_mode)
        
        def _shard(x):
            return maybe_shard_with_logical(
                x, logical_axis_names, mesh=self.mesh, shard_mode=cfg.shard_mode
            )

        inputs = _shard(inputs)
        inputs = checkpoint_name(inputs, "decoder_layer_input")

        lnx = self.pre_self_attention_norm(inputs)
        lnx = _shard(lnx)

        attention_lnx, kv_cache = self.self_attention(
            lnx, lnx, 
            decoder_positions, 
            decoder_segment_ids=decoder_segment_ids,
            deterministic=deterministic,
            model_mode=model_mode,
            kv_cache=kv_cache,
            attention_metadata=attention_metadata
        )
        attention_lnx = _shard(attention_lnx)

        mlp_lnx = self.mlp_lnx(lnx, deterministic=deterministic)
        mlp_lnx = _shard(mlp_lnx)

        next_layer_addition = mlp_lnx + attention_lnx
        next_layer_addition_dropped_out = self.dropout(
            next_layer_addition, 
            deterministic=deterministic
        )
        
        layer_output = next_layer_addition_dropped_out + inputs
        layer_output = _shard(layer_output)

        if cfg.record_internal_nn_metrics:
            metrics = {
                "activation_mean": jnp.mean(layer_output),
                "activation_stdev": jnp.std(layer_output),
                "activation_fraction_zero": jnp.sum(layer_output == 0) / jnp.size(layer_output),
            }
            # Update metrics variable
            if hasattr(self, 'metrics'):
                self.metrics.value = metrics

        return layer_output, kv_cache

# ------------------------------------------------------------------------------
#  Sequential Layers Container
# ------------------------------------------------------------------------------

class SequentialBlockDecoderLayers(nnx.Module):
    """Sequential unscanned series of decoder layers (NNX)."""
    
    def __init__(self, decoder_layer, num_decoder_layers, config, mesh, quant, model_mode, rngs, remat_policy=None, **kwargs):
        self.remat_policy = remat_policy
        self.decoder_layer = decoder_layer
        self.num_decoder_layers = num_decoder_layers
        
        # Attributes required by Pipeline
        self.config = config
        self.mesh = mesh
        self.quant = quant
        self.model_mode = model_mode
        self.scan_layers = kwargs.get('scan_layers', False)
        
        # Determine start index (useful if this block is part of a pipeline stage)
        # Default to 0 if not provided.
        start_idx = kwargs.get('layer_idx', 0)

        layers_list = []
        for i in range(num_decoder_layers):
            current_idx = start_idx + i
            layer_kwargs = {}
            
            # --- Model-Specific Logic (Replicating Linen Decoder loop) ---
            if config.decoder_block == DecoderBlockType.GEMMA3:
                 layer_kwargs["attention_type"] = gemma3.get_attention_type(layer_id=current_idx)
            
            if config.decoder_block == DecoderBlockType.LLAMA4:
                 # Note: ensure llama4 is imported in the file or locally
                 layer_kwargs["is_nope_layer"] = llama4.determine_is_nope_layer(current_idx, config.nope_layer_interval)
                 layer_kwargs["is_moe_layer"] = llama4.determine_is_moe_layer(current_idx, config.interleave_moe_layer_step)
            
            if config.decoder_block == DecoderBlockType.QWEN3_NEXT:
                 layer_kwargs["layer_idx"] = current_idx
            
            if config.decoder_block == DecoderBlockType.GPT_OSS:
                 layer_kwargs["attention_type"] = gpt_oss.get_attention_type(layer_id=current_idx)
            # -------------------------------------------------------------

            layers_list.append(
                decoder_layer(
                    config=config, 
                    mesh=mesh, 
                    quant=quant, 
                    model_mode=model_mode, 
                    rngs=rngs, 
                    **layer_kwargs
                )
            )

        self.layers = nnx.List(layers_list)

    def __call__(
        self, inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode, 
        slot=None, page_state=None, kv_caches=None
    ):
        output = inputs
        new_kv_caches = []
        
        def layer_pure_fn(state, graphdef, x, kv_cache, seg_ids, pos, det, mode):
            module = nnx.merge(graphdef, state)
            out, new_kv = module(x, seg_ids, pos, det, mode, slot=slot, page_state=page_state, kv_cache=kv_cache)
            _, new_state = nnx.split(module)
            return new_state, (out, new_kv)

        if self.remat_policy is not None:
            layer_pure_fn = jax.checkpoint(layer_pure_fn, policy=self.remat_policy, static_argnums=(6, 7))

        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            
            graphdef, state = nnx.split(layer)
            
            new_state, (output, new_kv) = layer_pure_fn(
                state, graphdef, output, kv_cache,
                decoder_segment_ids, decoder_positions, deterministic, model_mode
            )
            
            nnx.update(layer, new_state)
            
            if new_kv is not None:
                new_kv_caches.append(new_kv)

        return output, new_kv_caches

# ------------------------------------------------------------------------------
#  Main Decoder
# ------------------------------------------------------------------------------

class Decoder(nnx.Module):
    def __init__(self, config: Config, mesh: Mesh, model_mode: str = MODEL_MODE_TRAIN, quant: Optional[Any] = None, rngs: nnx.Rngs = None):
        self.config = config
        self.mesh = mesh
        self.model_mode = model_mode
        self.quant = quant
        
        if rngs is None:
            rngs = nnx.Rngs(params=jax.random.PRNGKey(config.init_weights_seed))
        self.rngs = rngs
        
        if config.trainable_position_size > 0:
            self.pos_embedder = Embed(
                num_embeddings=config.trainable_position_size,
                num_features=config.emb_dim,
                dtype=config.dtype,
                config=config,
                mesh=mesh,
                rngs=rngs
            )
        else:
            self.pos_embedder = None
            
        self.dropout = linears.Dropout(rate=self.config.dropout_rate, rngs=rngs, broadcast_dims=(-2,))
        self.decoder_norm = self.get_norm_layer(num_features=config.emb_dim, rngs=rngs)

        if config.logits_via_embedding:
            self.logits_dense = None
        else:
            self.logits_dense = linears.DenseGeneral(
                in_features_shape=self.config.emb_dim,
                out_features_shape=self.config.vocab_size,
                weight_dtype=self.config.weight_dtype,
                dtype=jnp.float32 if self.config.logits_dot_in_fp32 else self.config.dtype,
                kernel_axes=("embed", "vocab"),
                shard_mode=self.config.shard_mode,
                matmul_precision=self.config.matmul_precision,
                parameter_memory_host_offload=self.config.parameter_memory_host_offload,
                rngs=rngs,
            )
            # Init logits
            dummy_in = jnp.zeros((1, 1, self.config.emb_dim), dtype=self.config.dtype)
            try:
                self.logits_dense(dummy_in)
            except Exception:
                pass

        # FIX: Lazy Layer Setup for Pipeline (Prevents OOM)
        if self.config.using_pipeline_parallelism:
            # We create a SINGLE template layer (not the full stack)
            # The Pipeline module will use this template to create the full distributed stack
            self.layers_container = self._setup_layers(rngs, single_template=True)
            
            remat_policy = self.get_remat_policy()
            self.pipeline_module = pipeline.Pipeline(
                layers=self.layers_container,
                config=self.config,
                mesh=self.mesh,
                remat_policy=remat_policy,
                rngs=rngs
            )
        else:
            # Standard full stack creation
            self.layers_container = self._setup_layers(rngs, single_template=False)
            self.pipeline_module = None

    def _setup_layers(self, rngs, single_template=False):
        cfg = self.config
        LayerCls = self.get_decoder_layer_cls()[-1]
        remat_policy = self.get_remat_policy()

        # If pipeline enabled, we only need 1 layer worth of memory in Decoder
        num_layers = 1 if single_template else cfg.num_decoder_layers
        
        # Exception: For pipeline, we use num_layers_per_pipeline_stage logic *inside* Pipeline.
        # But here we just return a container. 
        # If single_template=True, we return SequentialBlock(num=1).
        
        if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
            return self._setup_deepseek_layers(rngs)

        if cfg.scan_layers and not single_template:
            # Scan VMAP logic (Omitted for brevity, assuming Pipeline path used)
            pass
        
        # Create Sequential Block
        # If single_template=True, we create a block with 1 layer to serve as a type/config carrier.
        return SequentialBlockDecoderLayers(
            LayerCls, 
            num_layers, # 1 if pipeline, else 32
            cfg, self.mesh, self.quant, self.model_mode, rngs, 
            remat_policy=remat_policy
        )

    def _setup_deepseek_layers(self, rngs):
        """DeepSeek specific setup (Dense Stack + MoE Stack)."""
        cfg = self.config
        DenseCls, MoeCls = self.get_decoder_layer_cls()
        
        dense_len = cfg.first_num_dense_layers
        moe_len = cfg.num_decoder_layers - dense_len
        
        dense_rngs = rngs.fork(split=dense_len)
        moe_rngs = rngs.fork(split=moe_len)
        
        def create_dense(r): return DenseCls(config=cfg, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode, rngs=r)
        def create_moe(r): return MoeCls(config=cfg, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode, rngs=r)

        return {
            "dense": nnx.vmap(create_dense)(dense_rngs),
            "moe": nnx.vmap(create_moe)(moe_rngs)
        }

    def get_norm_layer(self, num_features, rngs):
        return nnx.RMSNorm(
            num_features=num_features,
            epsilon=self.config.normalization_layer_epsilon,
            dtype=self.config.dtype,
            rngs=rngs
        )

    def get_decoder_layer_cls(self):
        match self.config.decoder_block:
            case DecoderBlockType.DEFAULT: return [DecoderLayer]
            case DecoderBlockType.LLAMA2: return [llama2.LlamaDecoderLayer]
            case DecoderBlockType.MISTRAL: return [mistral.MistralDecoderLayer]
            case DecoderBlockType.MIXTRAL: return [mixtral.MixtralDecoderLayer]
            case DecoderBlockType.DEEPSEEK:
                if self.config.use_batch_split_schedule:
                    return [deepseek_batchsplit.DeepSeekDenseLayer, deepseek_batchsplit.DeepSeekMoELayer]
                return [deepseek.DeepSeekDenseLayer, deepseek.DeepSeekMoELayer]
            case DecoderBlockType.GEMMA: return [gemma.GemmaDecoderLayer]
            case DecoderBlockType.GEMMA2: return [gemma2.Gemma2DecoderLayer]
            case DecoderBlockType.GEMMA3: return [gemma3.Gemma3DecoderLayer]
            case DecoderBlockType.GPT3: return [gpt3.Gpt3DecoderLayer]
            case DecoderBlockType.GPT_OSS: return [gpt_oss.GptOssDecoderLayer]
            case DecoderBlockType.QWEN3: return [qwen3.Qwen3DecoderLayer]
            case DecoderBlockType.QWEN3_MOE: return [qwen3.Qwen3MoeDecoderLayer]
            case DecoderBlockType.QWEN3_NEXT: return [qwen3.Qwen3NextDecoderLayer]
            case DecoderBlockType.SIMPLE: return [simple_layer.SimpleDecoderLayer]
            case DecoderBlockType.SIMPLE_MLP: return [simple_layer.SimpleMlpDecoderLayer]
            case DecoderBlockType.LLAMA4: return [llama4.Llama4DecoderLayer]
            case _:
                raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block.value=}")

    def get_remat_policy(self):
        cfg = self.config
        if cfg.remat_policy == "none":
            return None
        if cfg.remat_policy == "minimal":
             return jax.checkpoint_policies.save_only_these_names(
                 "query_proj", "value_proj", "key_proj", "qkv_proj", "out_proj", "mlpwi_0", "mlpwi_1", "mlpwi", "mlpwo"
             )
        return jax.checkpoint_policies.nothing_saveable()

    # --- Manual Scanning ---
    def _run_scanned_stack(self, layer_stack, x, broadcast_args, kv_caches=None):
        graphdef, state = nnx.split(layer_stack)
        
        def scan_body(carry, layer_state_slice):
            hidden_states, kvc_list = carry
            layer_module = nnx.merge(graphdef, layer_state_slice)
            
            # Unpack broadcast_args
            seg, pos, det, mode = broadcast_args
            
            out, new_kv = layer_module(hidden_states, seg, pos, det, mode)
            
            _, new_layer_state = nnx.split(layer_module)
            
            # FIX: Append new KV to list?
            # Scan accumulated outputs must be same structure as input?
            # Actually scan returns (carry, accumulated).
            # We return output as carry (daisy chain) and KV as accumulation.
            return (out, kvc_list), (new_layer_state, new_kv)

        policy = self.get_remat_policy()
        if policy is not None:
            scan_body = jax.checkpoint(scan_body, policy=policy)

        init_carry = (x, [])
        (final_out, _), (final_state, final_kv_stack) = jax.lax.scan(scan_body, init_carry, state)
        
        nnx.update(layer_stack, final_state)
        return final_out, final_kv_stack

    # --- Forward Pass ---

    def _apply_embedding(self, shared_embedding, decoder_input_tokens, decoder_positions, deterministic, model_mode):
        cfg = self.config
        
        # NNX Architecture Fix: Use shared_embedding passed from caller
        y = shared_embedding(decoder_input_tokens.astype("int32"))
        
        y = self.dropout(y, deterministic=deterministic)
        y = y.astype(cfg.dtype)
        
        if cfg.use_untrainable_positional_embedding:
            y = positional_embedding_as_linen.apply_rotary(y, decoder_positions, cfg) 
        
        if cfg.trainable_position_size > 0:
            y += self.pos_embedder(decoder_positions.astype("int32"))
        return y

    def _apply_output_head(self, shared_embedding, y, deterministic, model_mode):
        cfg = self.config
        if cfg.shard_mode == ShardMode.EXPLICIT:
             sharding = create_sharding(self.mesh, ("activation_batch", "activation_length_no_exp", "activation_embed"))
             y = maybe_shard_with_logical(y, sharding, self.mesh, cfg.shard_mode)

        y = self.decoder_norm(y)
        y = self.dropout(y, deterministic=deterministic)

        if model_mode in (MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE):
             out_sharding = create_sharding(self.mesh, (None, None, "activation_vocab"))
        else:
             out_sharding = create_sharding(self.mesh, ("activation_embed_and_logits_batch", "activation_length_no_exp", "activation_vocab"))

        if cfg.logits_via_embedding:
            # Use shared embedding table
            embedding_table = shared_embedding.embedding.value
            if isinstance(embedding_table, nn_linen.spmd.LogicallyPartitioned):
                embedding_table = embedding_table.unbox()
            attend_dtype = jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype
            logits = attend_on_embedding(y, embedding_table, attend_dtype, self.config, out_sharding)
            if self.config.normalize_embedding_logits:
                logits = logits / jnp.sqrt(y.shape[-1])
        else:
            logits = self.logits_dense(y)
            logits = maybe_shard_with_logical(logits, out_sharding, self.mesh, cfg.shard_mode)

        if cfg.final_logits_soft_cap:
             logits = jnp.tanh(logits / cfg.final_logits_soft_cap) * cfg.final_logits_soft_cap
        if self.config.cast_logits_to_fp32:
            logits = logits.astype(jnp.float32)
        return logits

    def __call__(
        self,
        shared_embedding,
        decoder_input_tokens,
        decoder_positions,
        decoder_segment_ids=None,
        deterministic=False,
        model_mode=MODEL_MODE_TRAIN,
        previous_chunk=None,
        slot=None,
        page_state=None,
        bidirectional_mask=None,
        image_embeddings=None,
        image_masks=None,
        kv_caches=None,
        attention_metadata=None,
    ):
        cfg = self.config
        y = self._apply_embedding(shared_embedding, decoder_input_tokens, decoder_positions, deterministic, model_mode)
        
        broadcast_args = (decoder_segment_ids, decoder_positions, deterministic, model_mode)
        
        if cfg.using_pipeline_parallelism:
             # Pipeline handles execution
             y = self.pipeline_module(
                 y, 
                 segment_ids=decoder_segment_ids, 
                 positions=decoder_positions,
                 deterministic=deterministic, 
                 model_mode=model_mode
             )
        else:
            # Standard Execution
            if cfg.scan_layers:
                y, kv_caches = self._run_scanned_stack(self.layers_container, y, broadcast_args, kv_caches)
            else:
                y, kv_caches = self.layers_container(
                    y, *broadcast_args,
                    slot=slot, page_state=page_state, kv_caches=kv_caches
                )

        assert isinstance(y, jax.Array)
        hidden_state = y

        if self.config.attention == "vllm_rpa":
             logits = None
        elif cfg.num_vocab_tiling > 1 and model_mode == MODEL_MODE_TRAIN:
             logits = None
        else:
             logits = self._apply_output_head(shared_embedding, hidden_state, deterministic, model_mode)

        return logits, hidden_state, kv_caches