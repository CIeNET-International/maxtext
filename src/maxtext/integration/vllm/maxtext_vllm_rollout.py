import gc
import logging
import time
from typing import Any
import jax
from flax.traverse_util import flatten_dict, unflatten_dict

from pathwaysutils.experimental import reshard as _experimental_reshard
from tunix.generate.vllm_sampler import VllmConfig, VllmSampler
from tunix.rl.rollout import base_rollout, vllm_rollout


def get_vllm_hf_overrides(maxtext_config) -> dict[str, str]:
    del maxtext_config
    return {
        "architectures": ["MaxTextForCausalLM"],
    }


def generate_maxtext_config_for_vllm(maxtext_config):
    """Truncates the config for vLLM payload."""
    # We only need enough config for model architecture, not training config.
    # Converting to dict here allows it to be easily JSON serialized.
    cfg = maxtext_config
    return {
        "model_name": cfg.model_name,
        "base_num_decoder_layers": cfg.base_num_decoder_layers,
        "base_emb_dim": cfg.base_emb_dim,
        "base_num_query_heads": cfg.base_num_query_heads,
        "base_num_kv_heads": cfg.base_num_kv_heads,
        "base_mlp_dim": cfg.base_mlp_dim,
        "vocab_size": cfg.vocab_size,
        "logits_via_embedding": getattr(cfg, "logits_via_embedding", False),
        "use_mrope": getattr(cfg, "use_mrope", False),
        "mrope_section_size": getattr(cfg, "mrope_section_size", None),
        "scan_layers": False,  # vLLM requires flattened layers
        "decoder_block": cfg.decoder_block,
        "emb_dim": cfg.emb_dim,
        "num_query_heads": cfg.num_query_heads,
        "num_kv_heads": cfg.num_kv_heads,
        "mlp_dim": cfg.mlp_dim,
        "num_decoder_layers": cfg.num_decoder_layers,
        "attention": cfg.attention,
    }


def unroll_gemma_scanned_weights(weights):
    """Workaround for tunix unstacking bug with Gemma 3/4 scanned blocks.

    tunix fails to map nested layers like `layers.layers_0` to `layers_X`.
    We manually unroll them here if we detect the structure.
    """
    pure_dict = weights.to_pure_dict() if hasattr(weights, "to_pure_dict") else weights

    if not isinstance(pure_dict, dict):
        return weights

    flat_w = flatten_dict(pure_dict, sep=None)
    new_flat_w = {}

    # Check if this is actually a scanned Gemma 3/4 checkpoint
    is_gemma_scanned = False
    for k in flat_w:
        if "decoder" in k:
            try:
                dec_idx = k.index("decoder")
                if len(k) > dec_idx + 2:
                    if (
                        k[dec_idx + 1] in ("layers", "scanned_blocks")
                        and isinstance(k[dec_idx + 2], str)
                        and k[dec_idx + 2].startswith("layers_")
                    ):
                        is_gemma_scanned = True
                        break
            except ValueError:
                pass

    if not is_gemma_scanned:
        return weights

    logging.info(
        "MaxTextVllmSampler: Detected Gemma scanned weights structure. Unrolling along axis 1..."
    )

    pattern_keys = set()
    scan_length = 0
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

    for k, v in flat_w.items():
        if "decoder" in k:
            try:
                dec_idx = k.index("decoder")
                if (
                    len(k) > dec_idx + 2
                    and k[dec_idx + 1] in ("layers", "scanned_blocks")
                    and isinstance(k[dec_idx + 2], str)
                    and k[dec_idx + 2].startswith("layers_")
                ):
                    layer_sub_idx = int(k[dec_idx + 2].split("layers_")[-1])
                    pattern_keys.add(layer_sub_idx)
                    if hasattr(v, "shape") and len(v.shape) > 1:
                        scan_length = max(scan_length, v.shape[1])
            except ValueError:
                pass

    pattern_length = max(pattern_keys) + 1 if pattern_keys else 0
    logging.info(
        "MaxTextVllmSampler: Discovered scan_length=%s, pattern_length=%s",
        scan_length,
        pattern_length,
    )

    unrolled_count = 0
    for k, v in flat_w.items():
        is_unrolled = False
        if "decoder" in k:
            try:
                dec_idx = k.index("decoder")
                if (
                    len(k) > dec_idx + 2
                    and k[dec_idx + 1] in ("layers", "scanned_blocks")
                    and isinstance(k[dec_idx + 2], str)
                    and k[dec_idx + 2].startswith("layers_")
                ):
                    layer_sub_idx = int(k[dec_idx + 2].split("layers_")[-1])

                    if hasattr(v, "shape") and len(v.shape) > 1:
                        v_swapped = jnp.swapaxes(v, 1, 0)
                        unstacked = [v_swapped[i] for i in range(scan_length)]
                    else:
                        unstacked = [v] * scan_length

                    for i in range(scan_length):
                        global_idx = i * pattern_length + layer_sub_idx
                        new_key = list(k)
                        new_key[dec_idx + 1] = "layers"
                        new_key[dec_idx + 2] = global_idx  # integer key!
                        new_flat_w[tuple(new_key)] = unstacked[i]

                        if unrolled_count < 5:
                            old_shape = v.shape if hasattr(v, "shape") else "scalar"
                            new_shape = (
                                unstacked[i].shape
                                if hasattr(unstacked[i], "shape")
                                else "scalar"
                            )
                            logging.info(
                                "MaxTextVllmSampler Unrolled: %s from src %s. Old shape: %s, New shape: %s",
                                tuple(new_key),
                                k,
                                old_shape,
                                new_shape,
                            )

                        unrolled_count += 1
                    is_unrolled = True
                elif (
                    len(k) > dec_idx + 2
                    and k[dec_idx + 1] == "layers_remainder"
                    and isinstance(k[dec_idx + 2], str)
                    and k[dec_idx + 2].startswith("layers_")
                ):
                    layer_sub_idx = int(k[dec_idx + 2].split("layers_")[-1])
                    global_idx = scan_length * pattern_length + layer_sub_idx
                    new_key = list(k)
                    new_key[dec_idx + 1] = "layers"
                    new_key[dec_idx + 2] = global_idx  # integer key!
                    new_flat_w[tuple(new_key)] = v
                    unrolled_count += 1
                    is_unrolled = True
            except ValueError:
                pass

        if not is_unrolled:
            new_flat_w[k] = v

    logging.info(
        "MaxTextVllmSampler: Successfully unrolled %s scanned tensor components into vLLM-compatible nnx.List format.",
        unrolled_count,
    )
    return unflatten_dict(new_flat_w, sep=None)


class MaxTextVllmSampler(VllmSampler):
    """VllmSampler that delegates weight updates to a MaxText to vLLM converter.

    When a converter is supplied, update_params bypasses transfer_state_with_mappings
    entirely and instead runs converter.convert() followed by a direct device_put
    into the vLLM model-runner state dict.  If no converter is supplied the base-class
    behaviour is preserved, so this class is safe to use as a drop-in replacement.
    """

    def __init__(
        self,
        tokenizer: Any,
        config: VllmConfig,
        converter: Any = None,
    ):
        super().__init__(tokenizer=tokenizer, config=config)
        self._converter = converter

    def update_params(
        self,
        updated_weights: Any,
        filter_types: tuple[Any, ...] | None = None,
    ):
        """Updates parameters in the vLLM instance."""
        if self._converter is None:
            # Apply workaround to unroll scanned blocks (e.g. Gemma 3/4)
            # before handing them off to the base tunix sampler which assumes
            # flat layers or simple homogeneous scans.
            updated_weights = unroll_gemma_scanned_weights(updated_weights)
            super().update_params(updated_weights, filter_types)
            return

        # Delete the old weights first to avoid OOM since we can't reliably
        # perform an in-place transfer if the sharding/shapes differ.
        self.transformer_state = None
        if self.llm is not None:
            self.llm.reset_prefix_cache()
            self.llm.collective_rpc("delete_kv_cache")
            self.llm.collective_rpc("delete_weights")

        # Force synchronous execution of all pending JAX operations.
        jax.block_until_ready(updated_weights)

        # Convert checkpoint directly to huggingface target format, dropping maxtext wrappers
        converted_weights = self._converter.convert(updated_weights)

        # Force synchronous execution of conversion.
        jax.block_until_ready(converted_weights)

        # Attempt to free maxtext weights aggressively before loading vllm weights
        del updated_weights
        gc.collect()

        t0 = time.time()
        num_params = self._device_put_into_vllm_state(converted_weights)
        t1 = time.time()
        logging.info(
            "Directly device_put %d parameters from PyTree to vLLM. Total time = %s",
            num_params,
            t1 - t0,
        )

    def _device_put_into_vllm_state(self, converted_weights):
        if not hasattr(self._model_runner, "state"):
            raise AttributeError("vLLM model runner doesn't have state.")
        target_state = self._model_runner.state
        flat_weights = flatten_dict(converted_weights, sep=".")
        flat_target = flatten_dict(target_state, sep=".")
        count = 0

        import torch  # pylint: disable=import-outside-toplevel

        for key, value in flat_weights.items():
            if key not in flat_target:
                raise ValueError(f"Key {key} not found in vllm target state")
            target_tensor = flat_target[key]

            if isinstance(value, torch.Tensor):
                value = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(value))

            # Use experimental pre-reshard if available in the environment to speed up memory transfer
            target_mesh = (
                target_tensor.sharding.mesh
                if hasattr(target_tensor, "sharding")
                else None
            )
            if target_mesh and hasattr(_experimental_reshard, "pre_reshard"):
                value = _experimental_reshard.pre_reshard(value, target_mesh)

            target_tensor = jax.device_put(value, target_tensor.sharding)
            count += 1
        return count


class MaxTextVllmRollout(vllm_rollout.VllmRollout):
    """Custom VllmRollout that injects the MaxText config payload.

    By passing maxtext_config via `vllm_additional_config`, the underlying
    Tunix adapter will boot the native `MaxTextForCausalLM` engine in vLLM
    instead of falling back to HuggingFace generic pipelines.
    """

    def __init__(
        self,
        rollout_actor: Any,
        tokenizer: Any,
        mesh: jax.sharding.Mesh,
        rollout_config: base_rollout.RolloutConfig,
        maxtext_config: Any,
        cache_config_or_size: base_rollout.CacheConfig | int = None,
    ):  # pylint: disable=super-init-not-called,too-many-positional-arguments
        if cache_config_or_size is None:
            cache_config_or_size = rollout_config.kv_cache_size

        from tunix.generate import mappings

        mapping_config = mappings.MappingConfig.build(
            mapping_obj=rollout_config.rollout_mapping_config,
            model=rollout_actor,
            backend="vllm_jax",
        )

        engine_kwargs = {
            "max_model_len": cache_config_or_size,
            "model": rollout_config.rollout_vllm_model_version,
            "swap_space": getattr(
                rollout_config,
                "rollout_vllm_swap_space_size_gb",
                maxtext_config.swap_space_vllm_gb,
            ),
            "async_scheduling": rollout_config.rollout_vllm_async_scheduling,
        }

        if (
            hasattr(rollout_config, "rollout_vllm_kwargs")
            and rollout_config.rollout_vllm_kwargs
        ):
            engine_kwargs.update(rollout_config.rollout_vllm_kwargs)

        mt_config_payload = generate_maxtext_config_for_vllm(maxtext_config)
        additional_config = (
            getattr(rollout_config, "rollout_vllm_additional_config", {}) or {}
        )
        additional_config["maxtext_config"] = mt_config_payload
        if "vllm_hf_overrides" not in additional_config:
            additional_config["vllm_hf_overrides"] = get_vllm_hf_overrides(
                maxtext_config
            )

        self._sampler = MaxTextVllmSampler(
            tokenizer=tokenizer,
            config=VllmConfig(
                mesh=mesh,
                hbm_utilization=rollout_config.rollout_vllm_hbm_utilization,
                init_with_random_weights=rollout_config.rollout_vllm_init_with_random_weights,
                tpu_backend_type=rollout_config.rollout_vllm_tpu_backend_type,
                mapping_config=mapping_config,
                lora_config=rollout_config.rollout_vllm_lora_config,
                server_mode=rollout_config.rollout_vllm_server_mode,
                tensor_parallel_size=rollout_config.tensor_parallel_size,
                data_parallel_size=rollout_config.data_parallel_size,
                enable_dp_attention=rollout_config.rollout_vllm_enable_dp_attention,
                engine_kwargs=engine_kwargs,
                additional_config=additional_config,
            ),
            converter=None,
        )

        # Initial weight sync: run the converter so vLLM starts with real weights.
        from flax import nnx

        state = nnx.state(rollout_actor)
        self._sampler.load_checkpoint(state)
