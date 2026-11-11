import sys

def _patch():
    try:
        import jax
        import jax.numpy as jnp
        _orig_make_array = jax.make_array_from_single_device_arrays

        def _safe_make_array(shape, sharding, arrays, dtype=None):
            if len(arrays) == 0 and dtype is None and len(getattr(sharding, "addressable_devices", [])) == 0:
                dtype = getattr(sharding, "dtype", jnp.bfloat16)
            return _orig_make_array(shape, sharding, arrays, dtype=dtype)

        jax.make_array_from_single_device_arrays = _safe_make_array
    except Exception:
        pass

    try:
        import jax.distributed
        _orig_dist_init = jax.distributed.initialize
        def _safe_dist_init(*args, **kwargs):
            if "heartbeat_timeout_seconds" not in kwargs:
                kwargs["heartbeat_timeout_seconds"] = 1800
            if "initialization_timeout" not in kwargs:
                kwargs["initialization_timeout"] = 1800
            if "shutdown_timeout_seconds" not in kwargs:
                kwargs["shutdown_timeout_seconds"] = 600
            return _orig_dist_init(*args, **kwargs)
        jax.distributed.initialize = _safe_dist_init

        _orig_dist_shutdown = jax.distributed.shutdown
        def _safe_dist_shutdown(*args, **kwargs):
            try:
                return _orig_dist_shutdown(*args, **kwargs)
            except Exception:
                pass
        jax.distributed.shutdown = _safe_dist_shutdown
    except Exception:
        pass

    try:
        import orbax.checkpoint._src.metadata.array_metadata_store as ams
        from orbax.checkpoint._src.path import async_path
        async def _safe_maybe_create_base_dir(self, base_dir):
            return await async_path.mkdir(base_dir, parents=True, exist_ok=True)
        ams.Store._maybe_create_base_dir = _safe_maybe_create_base_dir
    except Exception:
        pass

    try:
        import tpu_inference.utils as tpu_utils
        def patched_hbm_usage_bytes(devices):
            for d in devices:
                try:
                    hbm_used = d.memory_stats()["bytes_in_use"]
                    hbm_limit = d.memory_stats()["bytes_limit"]
                    return [(hbm_used, hbm_limit)] * len(devices)
                except Exception:
                    continue
            return [(0, 0)] * len(devices)

        def patched_hbm_usage_gb(devices):
            return [(hbm_used / (1024 ** 3), hbm_limit / (1024 ** 3)) for hbm_used, hbm_limit in patched_hbm_usage_bytes(devices)]

        tpu_utils.hbm_usage_bytes = patched_hbm_usage_bytes
        tpu_utils.hbm_usage_gb = patched_hbm_usage_gb
    except Exception:
        pass

    try:
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        if not hasattr(pltpu, "reciprocal") and hasattr(pl, "reciprocal"):
            pltpu.reciprocal = pl.reciprocal
    except Exception:
        pass

_patch()
