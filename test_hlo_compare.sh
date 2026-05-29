#!/bin/bash
# Compare HLO between Linen (main branch) and NNX (test/pipeline-scan-nnx-v2)
# with scan_pipeline_iterations=true. Keeps local HLO for analysis.
#
# Usage:
#   1. Run Linen:  git checkout main && bash test_hlo_compare.sh linen
#   2. Run NNX:    git checkout test/pipeline-scan-nnx-v2 && bash test_hlo_compare.sh nnx
#   3. Compare:    bash test_hlo_compare.sh compare

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE="${SCRIPT_DIR}/variant_test_results_hlo_compare_3"

VARIANT="${1:-}"
if [[ -z "$VARIANT" ]]; then
  echo "Usage: $0 {linen|nnx|compare}"
  echo "  linen   — run on main branch (Linen pipeline)"
  echo "  nnx     — run on test/pipeline-scan-nnx-v2 branch (NNX pipeline)"
  echo "  compare — diff the HLO outputs"
  exit 1
fi

export JAX_LOG_COMPILES=1

run_pipeline() {
  local NAME="$1"
  local OUTPUT_DIR="${OUTPUT_BASE}/${NAME}"
  local HLO_DIR="${OUTPUT_DIR}/hlo"
  local LOG_FILE="${OUTPUT_DIR}/logs/${NAME}.log"
  mkdir -p "${HLO_DIR}" "${OUTPUT_DIR}/logs"

  # NNX branch types.py:2395 only sets os.environ["XLA_FLAGS"] if NOT already set.
  # On TPU VMs, libtpu may pre-set XLA_FLAGS, causing dump flags to be silently
  # dropped. Fix: pass flags via dump_hlo_xla_flags config param AND set
  # XLA_FLAGS directly so both old (pyconfig_deprecated) and new (types.py)
  # code paths work.
  local XLA_DUMP_FLAGS="--xla_dump_to=${HLO_DIR} --xla_dump_large_constants --xla_dump_hlo_module_re=jit_train_step"
  export XLA_FLAGS="${XLA_DUMP_FLAGS}"

  echo "=== Running ${NAME} pipeline (branch: $(git branch --show-current)) ==="
  echo "XLA_FLAGS=${XLA_FLAGS}"
  python -m maxtext.trainers.pre_train.train \
    maxtext/configs/base.yml \
    run_name="hlo_${NAME}_scan_true" \
    base_output_directory="${OUTPUT_DIR}" \
    model_name=llama2-7b \
    dataset_type=synthetic \
    steps=3 \
    debug_sharding=false \
    max_target_length=32 \
    async_checkpointing=true \
    enable_checkpointing=false \
    per_device_batch_size=2 \
    scan_layers_per_stage=false \
    num_pipeline_microbatches=4 \
    ici_pipeline_parallelism=2 \
    num_layers_per_pipeline_stage=1 \
    pipeline_fsdp_ag_per_repeat=true \
    upload_all_profiler_results=false \
    scan_pipeline_iterations=true \
    enable_nnx=false \
    pure_nnx_decoder=false \
    dump_hlo=false \
    dump_hlo_local_dir="${HLO_DIR}" \
    dump_hlo_delete_local_after=false \
    dump_hlo_gcs_dir="gs://mesa-maxtext/variant_test_results_hlo_compare_3/hlo_${NAME}_scan_true/" \
    > "${LOG_FILE}" 2>&1

  local EXIT_CODE=$?
  echo "${NAME} Exit Code: ${EXIT_CODE}"
  grep "Total memory size:" "${LOG_FILE}" | tail -1

  # Check for XLA-dumped train_step files
  local TRAIN_STEP_COUNT
  TRAIN_STEP_COUNT=$(find "${HLO_DIR}" -name "*train_step*" -type f 2>/dev/null | wc -l)
  echo "XLA dump train_step files: ${TRAIN_STEP_COUNT}"

  # If XLA dump missed train_step, check TPU VM for large files
  if [[ "${TRAIN_STEP_COUNT}" -eq 0 ]]; then
    echo "WARNING: No train_step HLO from XLA dump. Checking for large module files..."
    find "${HLO_DIR}" -type f -size +1M 2>/dev/null | head -5
    echo "Largest files:"
    find "${HLO_DIR}" -type f -exec du -k {} \; 2>/dev/null | sort -rn | head -5
  fi
  echo "---"
}

compare_hlo() {
  echo "=== Memory comparison ==="
  for name in linen nnx; do
    echo -n "${name}: "
    grep "Total memory size:" "${OUTPUT_BASE}/${name}/logs/${name}.log" 2>/dev/null | tail -1 || echo "NOT RUN"
  done

  echo ""
  echo "=== While loop count ==="
  for name in linen nnx; do
    local hlo_file
    hlo_file=$(find "${OUTPUT_BASE}/${name}/hlo" -name "*train_step*before_optimizations*" 2>/dev/null | head -1)
    if [[ -n "$hlo_file" ]]; then
      echo "${name}: $(grep -c 'while(' "$hlo_file") while loops"
    else
      echo "${name}: NO HLO FOUND"
    fi
  done

  echo ""
  echo "=== Scan body region comparison ==="
  LINEN_HLO=$(find "${OUTPUT_BASE}/linen/hlo" -name "*train_step*before_optimizations*" 2>/dev/null | head -1)
  NNX_HLO=$(find "${OUTPUT_BASE}/nnx/hlo" -name "*train_step*before_optimizations*" 2>/dev/null | head -1)
  if [[ -n "$LINEN_HLO" && -n "$NNX_HLO" ]]; then
    echo "Linen HLO: $(wc -l < "$LINEN_HLO") lines"
    echo "NNX HLO:   $(wc -l < "$NNX_HLO") lines"
    echo ""
    echo "Linen while loops:"
    grep "while(" "$LINEN_HLO"
    echo ""
    echo "NNX while loops:"
    grep "while(" "$NNX_HLO"
  else
    echo "Missing HLO files. Run both linen and nnx first."
  fi
}

case "$VARIANT" in
  linen) run_pipeline "linen" ;;
  nnx)   run_pipeline "nnx" ;;
  compare) compare_hlo ;;
  *)
    echo "Unknown variant: $VARIANT"
    echo "Usage: $0 {linen|nnx|compare}"
    exit 1
    ;;
esac
