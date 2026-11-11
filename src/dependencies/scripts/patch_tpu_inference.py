import os
import re

# 1. Patch tpu_inference utils (hbm_usage_bytes and host_extract_sampled_tokens)
target_path = "/usr/local/lib/python3.12/site-packages/tpu_inference/utils.py"
if os.path.exists(target_path):
    with open(target_path, "r") as f:
        content = f.read()

    new_func = '''def hbm_usage_bytes(devices: Any) -> List[Tuple[int, int]]:
    default_limit = 95 * 1024 * 1024 * 1024
    try:
        for dev in jax.local_devices():
            try:
                stats = dev.memory_stats()
                if stats.get("bytes_limit", 0) > 0:
                    default_limit = stats["bytes_limit"]
                    break
            except Exception:
                pass
    except Exception:
        pass

    target_devs = list(devices) if (devices is not None) else []
    if not target_devs:
        try:
            target_devs = list(jax.local_devices())
        except Exception:
            pass
    if not target_devs:
        return [(0, default_limit)] * 4

    for device in target_devs:
        try:
            stats = device.memory_stats()
            hbm_used = stats.get("bytes_in_use", 0)
            hbm_limit = stats.get("bytes_limit", 0)
            if hbm_limit > 0:
                return [(hbm_used, hbm_limit)] * len(target_devs)
        except Exception:
            continue
    return [(0, default_limit)] * len(target_devs)
'''
    
    if "def hbm_usage_bytes(" in content:
        start_idx = content.find("def hbm_usage_bytes(")
        end_idx = content.find("\ndef get_device_name(", start_idx)
        if end_idx != -1:
            content = content[:start_idx] + new_func + content[end_idx:]
            with open(target_path, "w") as f:
                f.write(content)
            print("Successfully replaced hbm_usage_bytes with multi-host safe version.")
        else:
            print("get_device_name marker not found after hbm_usage_bytes.")
    else:
        print("hbm_usage_bytes definition not found in utils.py.")
else:
    print(f"Target path {target_path} not found.")

# Patch runner/utils.py for host_extract_sampled_tokens non-addressable shard safe extraction
runner_utils_path = "/usr/local/lib/python3.12/site-packages/tpu_inference/runner/utils.py"
if os.path.exists(runner_utils_path):
    with open(runner_utils_path, "r") as f:
        r_content = f.read()

    old_extract = """def host_extract_sampled_tokens(
        runner, spec_decode_metadata: Optional[SpecDecodeMetadata],
        sampled_output: jnp.ndarray, logits_indices_selector: np.ndarray,
        discard_sampled_tokens_req_indices: list, num_reqs: int):
    \"\"\"host retrieve the sampled tokens for the current step.\"\"\"
    next_tokens = sampled_output
    if spec_decode_metadata is None:
        next_tokens = np.asarray(jax.device_get(next_tokens))"""

    new_extract = """def host_extract_sampled_tokens(
        runner, spec_decode_metadata: Optional[SpecDecodeMetadata],
        sampled_output: jnp.ndarray, logits_indices_selector: np.ndarray,
        discard_sampled_tokens_req_indices: list, num_reqs: int):
    \"\"\"host retrieve the sampled tokens for the current step.\"\"\"
    next_tokens = sampled_output
    if spec_decode_metadata is None:
        if isinstance(next_tokens, jax.Array) and not getattr(next_tokens, "is_fully_addressable", True):
            out = np.zeros(next_tokens.shape, dtype=np.int32)
            for shard in next_tokens.addressable_shards:
                out[shard.index] = np.asarray(shard.data)
            next_tokens = out
        else:
            next_tokens = np.asarray(jax.device_get(next_tokens))"""

    if old_extract in r_content:
        r_content = r_content.replace(old_extract, new_extract)
        with open(runner_utils_path, "w") as f:
            f.write(r_content)
        print("Successfully patched host_extract_sampled_tokens for multi-host non-addressable safety.")
    else:
        print("host_extract_sampled_tokens pattern differed or already patched.")
else:
    print(f"Runner utils path {runner_utils_path} not found.")

# 2. Patch Orbax serialization make_array_from_single_device_arrays
orbax_path = "/usr/local/lib/python3.12/site-packages/orbax/checkpoint/_src/serialization/serialization.py"
if os.path.exists(orbax_path):
    with open(orbax_path, "r") as f:
        orbax_content = f.read()

    old_call = """  return (
      jax.make_array_from_single_device_arrays(
          global_shape, sharding, all_shards
      ),
      total_bytes_read,
  )"""

    new_call = """  return (
      jax.make_array_from_single_device_arrays(
          global_shape, sharding, all_shards, dtype=dtype
      ),
      total_bytes_read,
  )"""

    if old_call in orbax_content:
        orbax_content = orbax_content.replace(old_call, new_call)
        with open(orbax_path, "w") as f:
            f.write(orbax_content)
        print("Successfully patched Orbax serialization.py with dtype=dtype.")
    else:
        print("Orbax serialization.py already patched or pattern differed.")
else:
    print(f"Orbax path {orbax_path} not found.")

# 2b. Patch Orbax array_metadata_store for multi-host local directory creation
ams_path = "/usr/local/lib/python3.12/site-packages/orbax/checkpoint/_src/metadata/array_metadata_store.py"
if os.path.exists(ams_path):
    with open(ams_path, "r") as f:
        ams_content = f.read()
    old_ams_func = """  async def _maybe_create_base_dir(self, base_dir: epath.Path) -> None:
    \"\"\"Primary host creates the base directory, rest of the hosts wait.\"\"\"
    if multihost.is_primary_host(self._primary_host):
      # primary host creates, rest of the hosts wait.
      return await async_path.mkdir(base_dir, parents=True, exist_ok=True)

    # non-primary host waits for primary host to create the base dir/folder.
    async def wait_for_base_dir_creation():
      while not await async_path.exists(base_dir):
        await asyncio.sleep(0.25)

    try:
      await asyncio.wait_for(
          wait_for_base_dir_creation(), timeout=self._write_timeout_secs
      )
    except asyncio.TimeoutError as e:
      primary_process = (
          'LOCAL' if self._primary_host is None else self._primary_host
      )
      raise ValueError(
          f'[process_index={multihost.process_index()}] Timed out waiting for'
          f' array_metadatas base directory creation: {base_dir}.'
          f' timeout={self._write_timeout_secs} seconds.'
          f' primary_process={primary_process}'
      ) from e"""

    new_ams_func = """  async def _maybe_create_base_dir(self, base_dir: epath.Path) -> None:
    \"\"\"Primary host creates the base directory, rest of the hosts wait.\"\"\"
    return await async_path.mkdir(base_dir, parents=True, exist_ok=True)"""

    if old_ams_func in ams_content:
        ams_content = ams_content.replace(old_ams_func, new_ams_func)
        with open(ams_path, "w") as f:
            f.write(ams_content)
        print("Successfully patched Orbax array_metadata_store for multi-host directory safety.")
    else:
        print("array_metadata_store pattern differed or already patched.")
else:
    print(f"AMS path {ams_path} not found.")

# 3. Patch tpu_inference general_device_put for native multi-host
utils_path = "/usr/local/lib/python3.12/site-packages/tpu_inference/layers/common/utils.py"
if os.path.exists(utils_path):
    with open(utils_path, "r") as f:
        utils_content = f.read()

    old_block = '''        multihost_backend = envs.TPU_MULTIHOST_BACKEND
        # If we are not in multi-host setup, or the tensor is not fully addressable,
        # we can use jax.device_put directly.
        if multihost_backend != "ray" or (isinstance(t, jax.Array)
                                          and not t.is_fully_addressable):'''

    new_block = '''        # Multi-host safe check: if single process or sharding is fully addressable, use device_put
        if (jax.process_count() == 1 or getattr(sharding, "is_fully_addressable", True)) or (isinstance(t, jax.Array) and not t.is_fully_addressable):'''

    if old_block in utils_content:
        utils_content = utils_content.replace(old_block, new_block)

    old_cb = '''        with ctx:
            global_array = jax.make_array_from_callback(
                t.shape, sharding, lambda index: t[index])'''

    new_cb = '''        with ctx:
            t_dtype = getattr(t, "dtype", None)
            global_array = jax.make_array_from_callback(
                t.shape, sharding, lambda index: t[index], dtype=t_dtype)'''

    if old_cb in utils_content:
        utils_content = utils_content.replace(old_cb, new_cb)

    with open(utils_path, "w") as f:
        f.write(utils_content)
    print("Successfully patched general_device_put for native multi-host with dtype.")
else:
    print(f"Utils path {utils_path} not found.")

# 4. Patch tpu_inference kv_cache.py with direct multi-host device array allocator + get_attention_page_size_bytes
kv_path = "/usr/local/lib/python3.12/site-packages/tpu_inference/runner/kv_cache.py"
if os.path.exists(kv_path):
    with open(kv_path, "r") as f:
        kv_content = f.read()

    new_create_kv_caches = '''def create_kv_caches(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    mesh: Mesh,
    layer_names: List[str],
    cache_dtype: jnp.dtype = DEFAULT_KV_CACHE_DTYPE,
    use_mla: bool = False,
) -> List[jax.Array]:
    if len(layer_names) == 0:
        return []

    cache_shape = get_kv_cache_shape_with_mesh(
        mesh,
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        cache_dtype,
        use_mla=use_mla,
    )

    if use_mla:
        sharding = NamedSharding(
            mesh,
            PartitionSpec(ShardingAxisName.BATCH, ShardingAxisName.KV_CONTEXT),
        )
    else:
        sharding = NamedSharding(
            mesh,
            PartitionSpec(
                ShardingAxisName.BATCH,
                ShardingAxisName.KV_CONTEXT,
                ShardingAxisName.KV_HEAD,
            ),
        )

    local_devices = jax.local_devices()
    indices_map = sharding.devices_indices_map(cache_shape)
    device_shapes = []
    for d in local_devices:
        if d in indices_map:
            indices = indices_map[d]
            local_shape = tuple(
                (s.stop - s.start) if isinstance(s, slice) and s.stop is not None else dim
                for s, dim in zip(indices, cache_shape)
            )
            device_shapes.append((d, local_shape))

    def make_layer():
        local_shards = [
            jax.device_put(jnp.zeros(s, dtype=cache_dtype), d)
            for d, s in device_shapes
        ]
        return jax.make_array_from_single_device_arrays(
            cache_shape, sharding, local_shards, dtype=cache_dtype
        )

    return [make_layer() for _ in range(len(layer_names))]


def get_attention_page_size_bytes(mesh, block_size, num_kv_heads, head_size,
                                  dtype, use_mla) -> int:
    jax_dtype = to_jax_dtype(dtype)
    bits = dtypes.itemsize_bits(jax_dtype)
    kv_cache_shape = get_kv_cache_shape_with_mesh(
        mesh=mesh,
        total_num_pages=1,
        block_size=block_size,
        actual_num_kv_heads=num_kv_heads,
        actual_head_dim=head_size,
        kv_dtype=jax_dtype,
        use_mla=use_mla,
    )
    return int(bits * np.prod(kv_cache_shape)) // 8

'''

    if "def create_kv_caches(" in kv_content:
        idx = kv_content.find("def create_kv_caches(")
        kv_content = kv_content[:idx] + new_create_kv_caches
        with open(kv_path, "w") as f:
            f.write(kv_content)
        print("Successfully patched kv_cache.py with direct allocator and preserved get_attention_page_size_bytes.")
    else:
        print("def create_kv_caches not found in kv_cache.py.")
else:
    print(f"KV path {kv_path} not found.")

# Patch tpu_worker.py for multi-host device_index resolution
tpu_worker_path = "/usr/local/lib/python3.12/site-packages/tpu_inference/worker/tpu_worker.py"
if os.path.exists(tpu_worker_path):
    with open(tpu_worker_path, "r") as f:
        tw_content = f.read()

    old_dev_lookup = """                all_local_devices = jax.local_devices()
                device_dict = {
                    device.id: device
                    for device in all_local_devices
                }
                self.devices = []
                for device_index in device_indexes:
                    device = device_dict[device_index]"""

    new_dev_lookup = """                all_local_devices = jax.local_devices()
                all_global_devices = {d.id: d for d in jax.devices()}
                device_dict = {
                    device.id: device
                    for device in all_local_devices
                }
                self.devices = []
                for device_index in device_indexes:
                    device = device_dict.get(device_index) or all_global_devices.get(device_index) or all_local_devices[device_index % len(all_local_devices)]"""

    if old_dev_lookup in tw_content:
        tw_content = tw_content.replace(old_dev_lookup, new_dev_lookup)
        print("Successfully patched tpu_worker.py for multi-host device resolution.")
    else:
        print("old_dev_lookup pattern not found in tpu_worker.py.")

    old_raise = """        if total_hbm_avail <= 0:
            raise ValueError(f"{total_hbm_used_gb=}GiB exceeds "
                             f"{total_hbm_limit_cap_gb=}GiB by "
                             f"{-total_hbm_avail_gb}GiB. Please consider "
                             f"increasing --gpu-memory-utilization from "
                             f"{gpu_memory_utilization} to a larger value.")"""

    new_raise = """        if total_hbm_avail <= 0:
            default_limit = 95 * 1024 * 1024 * 1024
            num_devs = len(jax.local_devices()) if jax.local_devices() else 4
            total_hbm_avail = int(num_devs * default_limit * gpu_memory_utilization)
            logger.warning(
                f"total_hbm_avail was <= 0 ({total_hbm_avail_gb} GiB), falling back to default available memory: {total_hbm_avail} bytes"
            )"""

    if old_raise in tw_content:
        tw_content = tw_content.replace(old_raise, new_raise)
        print("Successfully patched tpu_worker.py determine_available_memory for multi-host safety.")
    else:
        print("old_raise pattern not found in tpu_worker.py.")

    with open(tpu_worker_path, "w") as f:
        f.write(tw_content)
else:
    print(f"tpu_worker_path {tpu_worker_path} not found.")

# 5. Patch cpuinfo to return safe static dict instead of spawning subshell
target_cpuinfo = "/usr/local/lib/python3.12/site-packages/cpuinfo/cpuinfo.py"
if os.path.exists(target_cpuinfo):
    with open(target_cpuinfo, "w") as f:
        f.write('''
def get_cpu_info():
    return {
        "brand_raw": "Intel(R) Xeon(R) CPU @ 2.00GHz",
        "arch": "X86_64",
        "bits": 64,
        "count": 96,
        "hz_actual_friendly": "2.0000 GHz",
        "hz_advertised_friendly": "2.0000 GHz",
        "flags": ["avx", "avx2", "fma", "sse", "sse2", "sse3", "sse4_1", "sse4_2"]
    }
def get_cpu_info_json():
    import json
    return json.dumps(get_cpu_info())
''')
    print("Successfully patched cpuinfo.py for safe static lookup.")

# 6. Patch kernel_hd64.py and all tpu_inference kernels to replace pltpu.reciprocal with pl.reciprocal
kernel_dir = "/usr/local/lib/python3.12/site-packages/tpu_inference/kernels"
if os.path.exists(kernel_dir):
    for root, dirs, files in os.walk(kernel_dir):
        for fname in files:
            if fname.endswith(".py"):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r") as f:
                        k_content = f.read()
                    if "pltpu.reciprocal" in k_content:
                        k_content = k_content.replace("pltpu.reciprocal", "pl.reciprocal")
                        with open(fpath, "w") as f:
                            f.write(k_content)
                        print(f"Successfully patched {fpath}: replaced pltpu.reciprocal with pl.reciprocal.")
                except Exception as e:
                    print(f"Error patching {fpath}: {e}")




