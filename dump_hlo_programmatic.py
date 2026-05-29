"""Dump train_step HLO + buffer assignment via compiled APIs. No XLA_FLAGS needed.

Outputs:
  - train_step.after_opt.txt    Post-optimization HLO (while loops, computation structure)
  - train_step.before_opt.txt   Pre-optimization StableHLO (JAX lowered IR)
  - buffer_assignment.txt       Buffer sizes, lifetimes, and aliasing decisions
  - memory_summary.txt          Memory breakdown (total, output, temp, argument)

Usage (on TPU VM):
  python dump_hlo_programmatic.py /tmp/hlo_output maxtext/configs/base.yml \
    model_name=llama2-7b dataset_type=synthetic steps=1 \
    max_target_length=32 per_device_batch_size=2 \
    ici_pipeline_parallelism=2 num_pipeline_microbatches=4 \
    num_layers_per_pipeline_stage=1 pipeline_fsdp_ag_per_repeat=true \
    scan_pipeline_iterations=true enable_nnx=false pure_nnx_decoder=false
"""

import os
import sys

OUTPUT_DIR = sys.argv[1]
os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.argv = [sys.argv[0]] + sys.argv[2:]

from maxtext.trainers.pre_train import train as train_module

_original_train_loop = train_module.train_loop

def patched_train_loop(config, recorder, state=None):
  from maxtext.utils import train_utils
  from maxtext.utils import maxtext_utils
  from maxtext.utils import max_utils
  from maxtext.utils import sharding
  from flax.linen import partitioning as nn_partitioning
  import jax

  (
      init_rng,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      data_loader,
      rampup_manager,
      eval_data_iterator,
      state,
  ) = train_utils.setup_train_loop(config, recorder)

  params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(
      config, state_mesh_shardings
  )

  with jax.set_mesh(mesh), mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
        config, model, mesh, state, state_mesh_shardings,
        train_module.train_step, train_module.eval_step,
        eval_data_iterator, params_shardings,
    )
    shaped_batch = maxtext_utils.get_shaped_batch(config)
    if config.shard_optimizer_over_data:
      state = sharding.maybe_shard_with_name(state, state_mesh_shardings, config.shard_mode)

    print("Compiling train_step...")
    lowered = p_train_step.lower(state, shaped_batch, init_rng)
    compiled = lowered.compile()

    stats = compiled.memory_analysis()
    max_utils.print_compiled_memory_stats(stats)

    hlo_text = compiled.as_text()
    path1 = os.path.join(OUTPUT_DIR, "train_step.after_opt.txt")
    with open(path1, "w") as f:
      f.write(hlo_text)
    print(f"Post-opt HLO: {path1} ({len(hlo_text)//1024} KB, {hlo_text.count('while(')} while loops)")

    pre_text = lowered.as_text()
    path2 = os.path.join(OUTPUT_DIR, "train_step.before_opt.txt")
    with open(path2, "w") as f:
      f.write(pre_text)
    print(f"Pre-opt HLO: {path2} ({len(pre_text)//1024} KB, {pre_text.count('while(')} while loops)")

    # Buffer assignment: shows buffer sizes, lifetimes, aliasing decisions
    try:
      cost_analysis = compiled.cost_analysis()
      path3 = os.path.join(OUTPUT_DIR, "cost_analysis.txt")
      with open(path3, "w") as f:
        for key, val in sorted(cost_analysis[0].items()):
          f.write(f"{key}: {val}\n")
      print(f"Cost analysis: {path3}")
    except Exception as e:
      print(f"Cost analysis unavailable: {e}")

    # Memory breakdown saved to file
    path4 = os.path.join(OUTPUT_DIR, "memory_summary.txt")
    with open(path4, "w") as f:
      f.write(f"Memory Analysis:\n")
      if stats:
        for attr in ['generated_code_size_in_bytes', 'argument_size_in_bytes',
                     'output_size_in_bytes', 'alias_size_in_bytes',
                     'temp_size_in_bytes', 'host_temp_size_in_bytes',
                     'host_argument_size_in_bytes', 'host_output_size_in_bytes']:
          val = getattr(stats, attr, None)
          if val is not None:
            size_gb = val / (1024**3)
            f.write(f"  {attr}: {val} ({size_gb:.2f} GB)\n")
    print(f"Memory summary: {path4}")

    # Extract buffer info from HLO text: find large allocations
    path5 = os.path.join(OUTPUT_DIR, "buffer_assignment.txt")
    with open(path5, "w") as f:
      f.write("Large buffers (>100 MB) in post-optimization HLO:\n")
      f.write("=" * 80 + "\n\n")
      import re
      # Find allocation sizes in HLO
      alloc_pattern = re.compile(r'(\w+)\s*=\s*\w+\[([^\]]*)\]\s*(\w+\[[^\]]*\])')
      # Find buffer sizes from shape annotations
      shape_pattern = re.compile(r'(f32|bf16|f16|s32|u32)\[([^\]]+)\]')
      dtype_sizes = {'f32': 4, 'bf16': 2, 'f16': 2, 's32': 4, 'u32': 4}
      large_buffers = []
      for match in shape_pattern.finditer(hlo_text):
        dtype, dims = match.group(1), match.group(2)
        try:
          size_bytes = dtype_sizes.get(dtype, 4)
          for dim in dims.split(','):
            dim = dim.strip()
            if dim.isdigit():
              size_bytes *= int(dim)
          size_mb = size_bytes / (1024 * 1024)
          if size_mb > 100:
            large_buffers.append((size_mb, dtype, dims, match.start()))
        except (ValueError, OverflowError):
          pass
      large_buffers.sort(reverse=True)
      seen_sizes = set()
      for size_mb, dtype, dims, pos in large_buffers[:50]:
        key = f"{dtype}[{dims}]"
        if key not in seen_sizes:
          seen_sizes.add(key)
          size_gb = size_mb / 1024
          if size_gb >= 1.0:
            f.write(f"  {size_gb:.2f} GB  {key}\n")
          else:
            f.write(f"  {size_mb:.0f} MB  {key}\n")
      f.write(f"\nTotal unique large buffer shapes: {len(seen_sizes)}\n")
      # Count while loops
      f.write(f"\nWhile loops in post-opt HLO: {hlo_text.count('while(')}\n")
      # Count closed_call
      f.write(f"closed_call occurrences: {hlo_text.count('closed_call')}\n")
    print(f"Buffer assignment: {path5} ({len(seen_sizes)} large buffers found)")

    print("\nDone.")
    sys.exit(0)

train_module.train_loop = patched_train_loop

from absl import app
app.run(train_module.main)
