#!/bin/bash
# Run DeepSeek MoE pipeline test matching test_circular_deepseek_megablox_same_output_and_grad config.
# Dumps HLO + memory analysis for the DeepSeek MoE circular pipeline.
#
# Usage:
#   bash test_deepseek_pipeline.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE="${SCRIPT_DIR}/variant_test_results_deepseek"
mkdir -p "${OUTPUT_BASE}"

export JAX_LOG_COMPILES=1

run_deepseek_pipeline() {
  local NAME="$1"
  local OUTPUT_DIR="${OUTPUT_BASE}/${NAME}"
  local HLO_DIR="${OUTPUT_DIR}/hlo"
  local LOG_FILE="${OUTPUT_DIR}/logs/${NAME}.log"
  mkdir -p "${HLO_DIR}" "${OUTPUT_DIR}/logs"

  local XLA_DUMP_FLAGS="--xla_dump_to=${HLO_DIR} --xla_dump_hlo_as_text=true --xla_dump_hlo_pass_re=spmd|layout --xla_dump_hlo_module_re=jit_train_step"
  export XLA_FLAGS="${XLA_DUMP_FLAGS}"

  echo "=== Running DeepSeek MoE pipeline: ${NAME} ==="
  echo "XLA_FLAGS=${XLA_FLAGS}"
  python -m maxtext.trainers.pre_train.train \
    maxtext/configs/base.yml \
    run_name="deepseek_${NAME}" \
    base_output_directory="${OUTPUT_DIR}" \
    dataset_type=synthetic \
    steps=3 \
    debug_sharding=false \
    async_checkpointing=false \
    enable_checkpointing=false \
    upload_all_profiler_results=false \
    scan_pipeline_iterations=true \
    max_target_length=128 \
    base_emb_dim=28 \
    ici_pipeline_parallelism=4 \
    base_num_decoder_layers=8 \
    num_pipeline_microbatches=8 \
    per_device_batch_size=4 \
    num_experts=4 \
    num_experts_per_tok=2 \
    megablox=false \
    sparse_matmul=false \
    capacity_factor=1 \
    decoder_block=deepseek \
    base_moe_mlp_dim=1024 \
    base_mlp_dim=1024 \
    attention_type=mla \
    shared_experts=1 \
    > "${LOG_FILE}" 2>&1

  local EXIT_CODE=$?
  echo "${NAME} Exit Code: ${EXIT_CODE}"
  grep "Total memory size:" "${LOG_FILE}" | tail -1 || echo "No memory info found"
  grep "Tokens/s/device:" "${LOG_FILE}" | tail -1 || echo "No throughput info found"

  local TRAIN_STEP_COUNT
  TRAIN_STEP_COUNT=$(find "${HLO_DIR}" -name "*train_step*" -type f 2>/dev/null | wc -l)
  echo "XLA dump train_step files: ${TRAIN_STEP_COUNT}"
  echo "---"

  # Run dump_hlo_programmatic.py for compile-time memory analysis + buffer info
  local PROG_DIR="${OUTPUT_DIR}/dump_hlo_programatically"
  mkdir -p "${PROG_DIR}"
  echo "=== Running dump_hlo_programmatic.py for ${NAME} ==="
  unset XLA_FLAGS
  python dump_hlo_programmatic.py "${PROG_DIR}" \
    maxtext/configs/base.yml \
    run_name="deepseek_prog_${NAME}" \
    base_output_directory="${PROG_DIR}" \
    dataset_type=synthetic \
    steps=1 \
    debug_sharding=false \
    async_checkpointing=false \
    enable_checkpointing=false \
    upload_all_profiler_results=false \
    scan_pipeline_iterations=true \
    max_target_length=128 \
    base_emb_dim=28 \
    ici_pipeline_parallelism=4 \
    base_num_decoder_layers=8 \
    num_pipeline_microbatches=8 \
    per_device_batch_size=4 \
    num_experts=4 \
    num_experts_per_tok=2 \
    megablox=false \
    sparse_matmul=false \
    capacity_factor=1 \
    decoder_block=deepseek \
    base_moe_mlp_dim=1024 \
    base_mlp_dim=1024 \
    attention_type=mla \
    shared_experts=1 \
    2>&1 | tee "${PROG_DIR}/programmatic.log"
  echo "Programmatic dump: ${PROG_DIR}"
  echo "==="
}

run_deepseek_pipeline "nnx_baseline"
