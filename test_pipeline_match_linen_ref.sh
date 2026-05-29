#!/bin/bash
# =============================================================================
# test_pipeline_match_linen_ref.sh
#   The Linen BASELINE reference for the xplane-match study.
#   RUN THIS ON MAIN:   git checkout main && bash test_pipeline_match_linen_ref.sh
#
#   Produces the SAME artifacts as test_pipeline_match.sh (identical config block)
#   so the NNX variants can be diffed against it apples-to-apples:
#     - mem.log            (Total memory size + throughput)
#     - dump_hlo/          (after_opt, before_opt, buffer_assignment, memory_summary, cost_analysis)
#     - xplane (.xplane.pb under <out>/tensorboard/.../plugins/profile/)
#     - structural.txt     (before_opt while-loops = expected 8 = the nn.remat fingerprint)
#
#   Config is byte-identical to pipeline_cfg() in test_pipeline_match.sh EXCEPT
#   enable_nnx=false (main is pure Linen). Everything else matches.
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$SCRIPT_DIR" || exit 1
OUT="${SCRIPT_DIR}/variant_test_results_match/_linen_ref_main"
mkdir -p "$OUT"

echo ">>> branch: $(git branch --show-current)  commit: $(git rev-parse --short HEAD)"
if [ "$(git branch --show-current)" != "main" ]; then
  echo "WARNING: not on main. This script is meant for the Linen baseline. Continue? (Ctrl-C to abort)"; sleep 4
fi

# IDENTICAL config block to test_pipeline_match.sh pipeline_cfg() — keep in sync.
CFG() {  # $1 run_name  $2 out  $3 steps  $4 extra
  echo "maxtext/configs/base.yml \
    run_name=$1 base_output_directory=$2 \
    model_name=llama2-7b dataset_type=synthetic \
    steps=$3 max_target_length=32 per_device_batch_size=2 \
    num_pipeline_microbatches=4 ici_pipeline_parallelism=2 num_layers_per_pipeline_stage=1 \
    pipeline_fsdp_ag_per_repeat=true scan_pipeline_iterations=true \
    enable_checkpointing=false async_checkpointing=false upload_all_profiler_results=false \
    managed_mldiagnostics=false enable_nnx=false pure_nnx_decoder=false $4"
}

echo "--- [1] train Linen + xplane (steps=10) ---"
python -m maxtext.trainers.pre_train.train $(CFG "linenref_train" "${OUT}/train_out" 10 "profiler=xplane skip_first_n_steps_for_profiler=2 profiler_steps=4") \
    2>&1 | tee "${OUT}/mem.log"

echo "--- [2] HLO dump (compile-only) ---"
python dump_hlo_programmatic.py "${OUT}/dump_hlo" $(CFG "linenref_hlo" "${OUT}/dump_hlo" 1 "") \
    2>&1 | tee "${OUT}/hlo.log"

echo "--- [3] structural counts (expect before_opt while ~8) ---"
{
  AO="${OUT}/dump_hlo/train_step.after_opt.txt"; BO="${OUT}/dump_hlo/train_step.before_opt.txt"; BA="${OUT}/dump_hlo/buffer_assignment.txt"
  echo "# LINEN MAIN reference"
  [ -f "$BO" ] && echo "before_opt while(:  $(grep -c 'while(' "$BO")"
  [ -f "$BA" ] && grep -iE 'While loops in post-opt|peak' "$BA"
  [ -f "$AO" ] && echo "after_opt  while(:        $(grep -c 'while(' "$AO")"
  [ -f "$AO" ] && echo "after_opt  closed_call:   $(grep -c 'closed_call\|@closed_call' "$AO")"
  [ -f "$AO" ] && echo "after_opt  all-gather:    $(grep -c 'all-gather' "$AO")"
  [ -f "$AO" ] && echo "after_opt  reduce-scatter:$(grep -c 'reduce-scatter' "$AO")"
  [ -f "${OUT}/dump_hlo/memory_summary.txt" ] && grep -iE 'temp_size|peak' "${OUT}/dump_hlo/memory_summary.txt"
} | tee "${OUT}/structural.txt"

echo ""; echo "=== LINEN REFERENCE DONE ==="
echo "Memory: $(grep -a 'Total memory size:' "${OUT}/mem.log" | tail -1)"
echo "Throughput: $(grep -aiE 'Tokens/s|tokens_per_second' "${OUT}/mem.log" | tail -1)"
echo "Artifacts: ${OUT}/  (mem.log, dump_hlo/, structural.txt, xplane under train_out/tensorboard/)"
echo "Now compare against NNX variants in variant_test_results_match/<variant>/"
