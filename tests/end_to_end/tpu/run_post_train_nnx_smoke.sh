#!/usr/bin/env bash
#
# Runs the post-training NNX migration smoke suite on a TPU VM.
#
# Direct TPU VM usage:
#   HF_TOKEN=<token> \
#   BASE_OUTPUT_DIRECTORY=gs://<bucket>/post_train_smoke \
#   bash tests/end_to_end/tpu/run_post_train_nnx_smoke.sh
#
# Remote TPU VM usage from a workstation with gcloud:
#   MODE=remote TPU_NAME=<tpu-vm-name> TPU_ZONE=<zone> REMOTE_REPO_DIR=~/maxtext \
#   BASE_OUTPUT_DIRECTORY=gs://<bucket>/post_train_smoke \
#   bash tests/end_to_end/tpu/run_post_train_nnx_smoke.sh
#
# Case selection:
#   CASES=all                         # default
#   CASES=quick                       # PT-01, PT-04, PT-05, PT-07, PT-09
#   CASES=pt01_sft_nnx_smoke,pt09_lora_sft
#
# Run dump:
#   DUMP_LOGS=1                        # default; writes ${LOG_DIR}/run_dump.md
#   PRINT_LOG_DUMP=1                   # default; prints the dump at the end
#   LOG_TAIL_LINES=40                  # default; per-case log tail length
#
# The cases mirror "MaxText NNX Migration - Smoke Test Status.md". A few
# defaults are intentionally adjusted for an 8-chip TPU VM:
#   - ICI_DATA_PARALLELISM defaults to 1, not the doc's 4, because
#     ici_fsdp_parallelism=8 and ici_data_parallelism=4 require 32 chips.
#   - CHIPS_PER_VM defaults to 8 for V6e-8.
#   - RL_ROLLOUT_TENSOR_PARALLELISM defaults to -1 so MaxText derives a value
#     that matches the sampler device count.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"

ALL_CASES=(
  pt01_sft_nnx_smoke
  pt02_sft_linen_ckpt
  pt02_sft_nnx_ckpt
  pt03_sft_linen_to_nnx
  pt04_rl_grpo_smoke
  pt05_rl_gspo_smoke
  pt06_rl_llama31_functional
  pt07_distill_smoke
  pt09_lora_sft
)

QUICK_CASES=(
  pt01_sft_nnx_smoke
  pt04_rl_grpo_smoke
  pt05_rl_gspo_smoke
  pt07_distill_smoke
  pt09_lora_sft
)

MODE="${MODE:-local}"
CASES="${CASES:-all}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_PREFIX="${RUN_PREFIX:-pt_nnx_smoke_${RUN_ID}}"
BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-gs://lance-maxtext/pt_ckpt_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/post_train_nnx_smoke/${RUN_ID}}"
SUMMARY_FILE="${SUMMARY_FILE:-${LOG_DIR}/summary.tsv}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
SKIP_ENV_CHECKS="${SKIP_ENV_CHECKS:-0}"
SKIP_MISSING_CASES="${SKIP_MISSING_CASES:-0}"
STRICT_DOC_PROGRESS="${STRICT_DOC_PROGRESS:-0}"
PRINT_ENV="${PRINT_ENV:-1}"
DUMP_LOGS="${DUMP_LOGS:-1}"
LOG_DUMP_FILE="${LOG_DUMP_FILE:-${LOG_DIR}/run_dump.md}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-40}"
PRINT_LOG_DUMP="${PRINT_LOG_DUMP:-1}"

# Common training knobs.
SFT_TRAINER_MODULE="${SFT_TRAINER_MODULE:-maxtext.trainers.post_train.sft.train_sft}"
DISTILL_TRAINER_MODULE="${DISTILL_TRAINER_MODULE:-maxtext.trainers.post_train.distillation.train_distill}"
RL_TRAINER_MODULE="${RL_TRAINER_MODULE:-maxtext.trainers.post_train.rl.train_rl}"
SFT_STEPS="${SFT_STEPS:-5}"
DISTILL_STEPS="${DISTILL_STEPS:-5}"
RL_NUM_BATCHES="${RL_NUM_BATCHES:-2}"
RL_FUNCTIONAL_NUM_BATCHES="${RL_FUNCTIONAL_NUM_BATCHES:-2}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-1024}"
WEIGHT_DTYPE="${WEIGHT_DTYPE:-float32}"
PROFILER="${PROFILER:-xplane}"
ICI_FSDP_PARALLELISM="${ICI_FSDP_PARALLELISM:-8}"
ICI_DATA_PARALLELISM="${ICI_DATA_PARALLELISM:-1}"

# RL topology knobs for V6e-8 TPU VMs. Override for other hardware.
CHIPS_PER_VM="${CHIPS_PER_VM:-8}"
TRAINER_DEVICES_FRACTION="${TRAINER_DEVICES_FRACTION:-1.0}"
SAMPLER_DEVICES_FRACTION="${SAMPLER_DEVICES_FRACTION:-1.0}"
RL_ROLLOUT_DATA_PARALLELISM="${RL_ROLLOUT_DATA_PARALLELISM:-2}"
RL_ROLLOUT_TENSOR_PARALLELISM="${RL_ROLLOUT_TENSOR_PARALLELISM:--1}"
RL_ROLLOUT_EXPERT_PARALLELISM="${RL_ROLLOUT_EXPERT_PARALLELISM:-1}"
RL_NUM_TEST_BATCHES="${RL_NUM_TEST_BATCHES:-0}"
RL_FUNCTIONAL_NUM_TEST_BATCHES="${RL_FUNCTIONAL_NUM_TEST_BATCHES:-1}"
RL_MAX_PREFILL_PREDICT_LENGTH="${RL_MAX_PREFILL_PREDICT_LENGTH:-256}"
RL_MAX_TARGET_LENGTH="${RL_MAX_TARGET_LENGTH:-1024}"

# Tokenizers and checkpoint paths from the smoke status doc.
SFT_TOKENIZER_PATH="${SFT_TOKENIZER_PATH:-meta-llama/Llama-2-7b-chat-hf}"
QWEN_TOKENIZER_PATH="${QWEN_TOKENIZER_PATH:-Qwen/Qwen3-0.6B}"
LLAMA31_TOKENIZER_PATH="${LLAMA31_TOKENIZER_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
LLAMA31_MODEL_NAME="${LLAMA31_MODEL_NAME:-llama3.1-8b}"

LINEN_GPT3_52K_CKPT="${LINEN_GPT3_52K_CKPT:-gs://lance-maxtext/pt_seed_ckpts/pt_seed_ckpts/pt_seed_ckpt_gpt352k_linen/checkpoints/9/items}"
NNX_GPT3_52K_CKPT="${NNX_GPT3_52K_CKPT:-}"
DISTILL_TEACHER_CKPT="${DISTILL_TEACHER_CKPT:-gs://lance-maxtext/pt_seed_ckpts/pt_seed_ckpts/pt_seed_ckpt_gpt352k_v32k_linen/checkpoints/4/items}"

REMOTE_COPY_SCRIPT="${REMOTE_COPY_SCRIPT:-1}"
REMOTE_SCRIPT_PATH="${REMOTE_SCRIPT_PATH:-}"
TPU_WORKER="${TPU_WORKER:-}"
PASS_HF_TOKEN_TO_REMOTE="${PASS_HF_TOKEN_TO_REMOTE:-0}"

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

quote() {
  printf '%q' "$1"
}

join_path() {
  local base="${1%/}"
  local child="${2#/}"
  printf '%s/%s' "${base}" "${child}"
}

contains_case() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [ "${item}" = "${needle}" ]; then
      return 0
    fi
  done
  return 1
}

selected_cases() {
  local normalized="${CASES//,/ }"
  local item

  if [ "${normalized}" = "all" ]; then
    printf '%s\n' "${ALL_CASES[@]}"
    return
  fi

  if [ "${normalized}" = "quick" ]; then
    printf '%s\n' "${QUICK_CASES[@]}"
    return
  fi

  for item in ${normalized}; do
    contains_case "${item}" "${ALL_CASES[@]}" || die "Unknown case '${item}'. Use MODE=list to see valid cases."
    printf '%s\n' "${item}"
  done
}

case_description() {
  case "$1" in
    pt01_sft_nnx_smoke) printf 'PT-01 SFT random-init NNX smoke' ;;
    pt02_sft_linen_ckpt) printf 'PT-02 SFT restore from Linen checkpoint' ;;
    pt02_sft_nnx_ckpt) printf 'PT-02 SFT restore from NNX checkpoint' ;;
    pt03_sft_linen_to_nnx) printf 'PT-03 cross-format Linen checkpoint into NNX model' ;;
    pt04_rl_grpo_smoke) printf 'PT-04 RL GRPO Qwen3 smoke' ;;
    pt05_rl_gspo_smoke) printf 'PT-05 RL GSPO Qwen3 smoke' ;;
    pt06_rl_llama31_functional) printf 'PT-06 RL GRPO Llama3.1 functional smoke' ;;
    pt07_distill_smoke) printf 'PT-07 GPT-3 52k distillation smoke' ;;
    pt09_lora_sft) printf 'PT-09 LoRA SFT smoke' ;;
    *) printf 'Unknown case' ;;
  esac
}

list_cases() {
  local case_id
  for case_id in "${ALL_CASES[@]}"; do
    printf '%-28s %s\n' "${case_id}" "$(case_description "${case_id}")"
  done
}

requires_hf_token() {
  local case_id
  for case_id in "$@"; do
    case "${case_id}" in
      pt01_sft_nnx_smoke|pt02_sft_linen_ckpt|pt02_sft_nnx_ckpt|pt03_sft_linen_to_nnx|pt06_rl_llama31_functional|pt07_distill_smoke|pt09_lora_sft)
        return 0
        ;;
    esac
  done
  return 1
}

preflight() {
  local cases=("$@")

  [ "${SKIP_ENV_CHECKS}" = "1" ] && return

  command -v python3 >/dev/null 2>&1 || die "python3 is not available."

  if requires_hf_token "${cases[@]}"; then
    [ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN is required for the selected Llama tokenizer/checkpoint cases."
  fi

  if contains_case pt02_sft_nnx_ckpt "${cases[@]}"; then
    if [ -z "${NNX_GPT3_52K_CKPT}" ]; then
      if [ "${SKIP_MISSING_CASES}" = "1" ]; then
        warn "NNX_GPT3_52K_CKPT is unset; pt02_sft_nnx_ckpt will be skipped."
      else
        die "NNX_GPT3_52K_CKPT is required for pt02_sft_nnx_ckpt. The smoke-status doc currently points that case at the Linen checkpoint, so this script refuses to guess."
      fi
    fi
  fi

  if [ "${ICI_DATA_PARALLELISM}" != "1" ]; then
    warn "ICI_DATA_PARALLELISM=${ICI_DATA_PARALLELISM}; with ICI_FSDP_PARALLELISM=${ICI_FSDP_PARALLELISM}, make sure the product fits your TPU topology."
  fi
}

print_startup() {
  [ "${PRINT_ENV}" = "1" ] || return
  printf 'Run id: %s\n' "${RUN_ID}"
  printf 'Run prefix: %s\n' "${RUN_PREFIX}"
  printf 'Repo root: %s\n' "${REPO_ROOT}"
  printf 'Base output: %s\n' "${BASE_OUTPUT_DIRECTORY}"
  printf 'Log dir: %s\n' "${LOG_DIR}"
  printf 'Log dump: %s (enabled=%s, print=%s, tail_lines=%s)\n' \
    "${LOG_DUMP_FILE}" "${DUMP_LOGS}" "${PRINT_LOG_DUMP}" "${LOG_TAIL_LINES}"
  printf 'SFT/distill ICI: fsdp=%s data=%s\n' "${ICI_FSDP_PARALLELISM}" "${ICI_DATA_PARALLELISM}"
  printf 'RL topology: chips_per_vm=%s trainer_fraction=%s sampler_fraction=%s rollout_dp=%s rollout_tp=%s rollout_ep=%s\n' \
    "${CHIPS_PER_VM}" "${TRAINER_DEVICES_FRACTION}" "${SAMPLER_DEVICES_FRACTION}" \
    "${RL_ROLLOUT_DATA_PARALLELISM}" "${RL_ROLLOUT_TENSOR_PARALLELISM}" "${RL_ROLLOUT_EXPERT_PARALLELISM}"
}

write_log_dump() {
  local exit_status="${1:-0}"
  [ "${DUMP_LOGS}" = "1" ] || return 0

  set +e
  mkdir -p "$(dirname "${LOG_DUMP_FILE}")"
  mkdir -p "${LOG_DIR}"

  local tmp_file="${LOG_DUMP_FILE}.tmp.$$"
  {
    printf '# Post-train NNX smoke run dump\n\n'
    printf 'Generated UTC: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'Exit status: %s\n' "${exit_status}"
    printf 'Run id: %s\n' "${RUN_ID}"
    printf 'Run prefix: %s\n' "${RUN_PREFIX}"
    printf 'Mode: %s\n' "${MODE}"
    printf 'Cases: %s\n' "${CASES}"
    printf 'Repo root: %s\n' "${REPO_ROOT}"
    printf 'Base output: %s\n' "${BASE_OUTPUT_DIRECTORY}"
    printf 'Log dir: %s\n' "${LOG_DIR}"
    printf 'Summary file: %s\n' "${SUMMARY_FILE}"
    if [ -n "${HF_TOKEN:-}" ]; then
      printf 'HF token: set\n'
    else
      printf 'HF token: unset\n'
    fi
    printf 'DRY_RUN: %s\n' "${DRY_RUN}"
    printf 'CONTINUE_ON_FAILURE: %s\n' "${CONTINUE_ON_FAILURE}"
    printf 'SKIP_MISSING_CASES: %s\n' "${SKIP_MISSING_CASES}"
    printf 'STRICT_DOC_PROGRESS: %s\n' "${STRICT_DOC_PROGRESS}"

    printf '\n## Summary\n\n'
    if [ -f "${SUMMARY_FILE}" ]; then
      column -t -s $'\t' "${SUMMARY_FILE}" 2>/dev/null || cat "${SUMMARY_FILE}"
    else
      printf '(summary file not found)\n'
    fi

    printf '\n## Log file tails\n'
    local found_logs=0
    local log_file
    while IFS= read -r log_file; do
      found_logs=1
      printf '\n### %s\n\n' "${log_file}"
      tail -n "${LOG_TAIL_LINES}" "${log_file}" 2>&1
    done < <(find "${LOG_DIR}" -maxdepth 1 -type f -name '*.log' | sort)

    if [ "${found_logs}" -eq 0 ]; then
      printf '\n(no .log files found; dry runs only write the summary)\n'
    fi
  } > "${tmp_file}"

  mv "${tmp_file}" "${LOG_DUMP_FILE}"
  printf '\nLog dump: %s\n' "${LOG_DUMP_FILE}"
  if [ "${PRINT_LOG_DUMP}" = "1" ]; then
    printf '\n'
    cat "${LOG_DUMP_FILE}"
  fi
}

common_sft_args() {
  printf '%s\n' \
    "src/maxtext/configs/post_train/sft.yml" \
    "model_name=gpt3-52k" \
    "per_device_batch_size=${PER_DEVICE_BATCH_SIZE}" \
    "ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM}" \
    "ici_data_parallelism=${ICI_DATA_PARALLELISM}" \
    "max_target_length=${MAX_TARGET_LENGTH}" \
    "steps=${SFT_STEPS}" \
    "eval_interval=-1" \
    "gradient_accumulation_steps=1" \
    "weight_dtype=${WEIGHT_DTYPE}" \
    "log_config=False" \
    "enable_goodput_recording=False" \
    "profiler=${PROFILER}" \
    "tokenizer_path=${SFT_TOKENIZER_PATH}" \
    "tokenizer_type=huggingface" \
    "hf_access_token=${HF_TOKEN:-}" \
    "base_output_directory=${BASE_OUTPUT_DIRECTORY}"
}

common_rl_args() {
  printf '%s\n' \
    "src/maxtext/configs/post_train/rl.yml" \
    "chips_per_vm=${CHIPS_PER_VM}" \
    "num_batches=${RL_NUM_BATCHES}" \
    "num_test_batches=${RL_NUM_TEST_BATCHES}" \
    "rollout_data_parallelism=${RL_ROLLOUT_DATA_PARALLELISM}" \
    "rollout_tensor_parallelism=${RL_ROLLOUT_TENSOR_PARALLELISM}" \
    "rollout_expert_parallelism=${RL_ROLLOUT_EXPERT_PARALLELISM}" \
    "trainer_devices_fraction=${TRAINER_DEVICES_FRACTION}" \
    "sampler_devices_fraction=${SAMPLER_DEVICES_FRACTION}" \
    "async_scheduling=False" \
    "log_config=False" \
    "enable_goodput_recording=False" \
    "profiler=${PROFILER}" \
    "debug.rl=False" \
    "pure_nnx=True" \
    "enable_nnx=True" \
    "pure_nnx_decoder=True" \
    "hf_access_token=${HF_TOKEN:-}" \
    "max_prefill_predict_length=${RL_MAX_PREFILL_PREDICT_LENGTH}" \
    "max_target_length=${RL_MAX_TARGET_LENGTH}"
}

validate_log() {
  local case_id="$1"
  local log_file="$2"
  local exit_code="$3"
  local progress_regex="$4"
  local status="PASS"
  local reason=""

  if [ "${exit_code}" -ne 0 ]; then
    status="FAIL"
    reason="exit_code=${exit_code}"
  elif grep -Eqi 'Traceback|(^|[^[:alpha:]])(loss|total_loss|lm_loss):?[[:space:]]*(nan|inf)([^[:alpha:]]|$)|Aborting training due to (NaN|Inf)' "${log_file}"; then
    status="FAIL"
    reason="traceback_or_nan_inf"
  elif ! grep -Eq "${progress_regex}" "${log_file}"; then
    status="FAIL"
    reason="missing_progress_marker"
  elif [ "${STRICT_DOC_PROGRESS}" = "1" ] && ! grep -q 'completed step:' "${log_file}"; then
    status="FAIL"
    reason="missing_doc_completed_step_marker"
  fi

  printf '%s\t%s\t%s\t%s\n' "${case_id}" "${status}" "${reason:-ok}" "${log_file}" >> "${SUMMARY_FILE}"

  if [ "${status}" = "FAIL" ]; then
    printf 'Case %s failed: %s\n' "${case_id}" "${reason}" >&2
    return 1
  fi

  printf 'Case %s passed smoke validation.\n' "${case_id}"
}

run_logged() {
  local case_id="$1"
  local description="$2"
  local progress_regex="$3"
  shift 3

  mkdir -p "${LOG_DIR}"
  local log_file="${LOG_DIR}/${case_id}.log"
  local exit_code=0

  printf '\n=== %s: %s ===\n' "${case_id}" "${description}"
  printf 'Log: %s\n' "${log_file}"
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'

  if [ "${DRY_RUN}" = "1" ]; then
    printf '%s\t%s\t%s\t%s\n' "${case_id}" "DRY_RUN" "not_executed" "${log_file}" >> "${SUMMARY_FILE}"
    return 0
  fi

  set +e
  "$@" 2>&1 | tee "${log_file}"
  exit_code="${PIPESTATUS[0]}"
  set -e

  validate_log "${case_id}" "${log_file}" "${exit_code}" "${progress_regex}"
}

run_case() {
  local case_id="$1"
  local args=()
  local sft_args=()
  local rl_args=()

  case "${case_id}" in
    pt01_sft_nnx_smoke)
      mapfile -t sft_args < <(common_sft_args)
      args=(python3 -m "${SFT_TRAINER_MODULE}" "${sft_args[@]}"
        "pure_nnx=True" "enable_nnx=True" "pure_nnx_decoder=True"
        "run_name=${RUN_PREFIX}_01_sft_smoke")
      run_logged "${case_id}" "$(case_description "${case_id}")" 'completed step:' "${args[@]}"
      ;;

    pt02_sft_linen_ckpt)
      mapfile -t sft_args < <(common_sft_args)
      args=(python3 -m "${SFT_TRAINER_MODULE}" "${sft_args[@]}"
        "pure_nnx=False" "enable_nnx=False" "pure_nnx_decoder=False"
        "load_parameters_path=${LINEN_GPT3_52K_CKPT}"
        "run_name=${RUN_PREFIX}_02_sft_linen_ckpt")
      run_logged "${case_id}" "$(case_description "${case_id}")" 'completed step:' "${args[@]}"
      ;;

    pt02_sft_nnx_ckpt)
      if [ -z "${NNX_GPT3_52K_CKPT}" ] && [ "${SKIP_MISSING_CASES}" = "1" ]; then
        printf '%s\t%s\t%s\t%s\n' "${case_id}" "SKIP" "NNX_GPT3_52K_CKPT unset" "" >> "${SUMMARY_FILE}"
        return 0
      fi
      mapfile -t sft_args < <(common_sft_args)
      args=(python3 -m "${SFT_TRAINER_MODULE}" "${sft_args[@]}"
        "pure_nnx=True" "enable_nnx=True" "pure_nnx_decoder=True"
        "load_parameters_path=${NNX_GPT3_52K_CKPT}"
        "run_name=${RUN_PREFIX}_02_sft_nnx_ckpt")
      run_logged "${case_id}" "$(case_description "${case_id}")" 'completed step:' "${args[@]}"
      ;;

    pt03_sft_linen_to_nnx)
      mapfile -t sft_args < <(common_sft_args)
      args=(python3 -m "${SFT_TRAINER_MODULE}" "${sft_args[@]}"
        "pure_nnx=True" "enable_nnx=True" "pure_nnx_decoder=True"
        "load_parameters_path=${LINEN_GPT3_52K_CKPT}"
        "run_name=${RUN_PREFIX}_03_sft_linen_ckpt")
      run_logged "${case_id}" "$(case_description "${case_id}")" 'completed step:' "${args[@]}"
      ;;

    pt04_rl_grpo_smoke)
      mapfile -t rl_args < <(common_rl_args)
      args=(python3 -m "${RL_TRAINER_MODULE}" "${rl_args[@]}"
        "model_name=qwen3-0.6b"
        "tokenizer_path=${QWEN_TOKENIZER_PATH}"
        "base_output_directory=$(join_path "${BASE_OUTPUT_DIRECTORY}" "04_rl_grpo_smoke")"
        "run_name=${RUN_PREFIX}_04_rl_grpo_smoke"
        "rl.loss_algo=grpo")
      run_logged "${case_id}" "$(case_description "${case_id}")" 'completed step:|RL Training Completed Successfully|Starting RL training' "${args[@]}"
      ;;

    pt05_rl_gspo_smoke)
      mapfile -t rl_args < <(common_rl_args)
      args=(python3 -m "${RL_TRAINER_MODULE}" "${rl_args[@]}"
        "model_name=qwen3-0.6b"
        "tokenizer_path=${QWEN_TOKENIZER_PATH}"
        "base_output_directory=$(join_path "${BASE_OUTPUT_DIRECTORY}" "05_rl_gspo_smoke")"
        "run_name=${RUN_PREFIX}_05_rl_gspo_smoke"
        "rl.loss_algo=gspo-token")
      run_logged "${case_id}" "$(case_description "${case_id}")" 'completed step:|RL Training Completed Successfully|Starting RL training' "${args[@]}"
      ;;

    pt06_rl_llama31_functional)
      mapfile -t rl_args < <(common_rl_args)
      args=(python3 -m "${RL_TRAINER_MODULE}" "${rl_args[@]}"
        "model_name=${LLAMA31_MODEL_NAME}"
        "tokenizer_path=${LLAMA31_TOKENIZER_PATH}"
        "num_batches=${RL_FUNCTIONAL_NUM_BATCHES}"
        "num_test_batches=${RL_FUNCTIONAL_NUM_TEST_BATCHES}"
        "convert_checkpoint_if_possible=False"
        "enable_checkpointing=False"
        "base_output_directory=$(join_path "${BASE_OUTPUT_DIRECTORY}" "06_rl_grpo_functional")"
        "run_name=${RUN_PREFIX}_06_rl_grpo_llama31_functional"
        "rl.loss_algo=grpo")
      run_logged "${case_id}" "$(case_description "${case_id}")" 'completed step:|RL Training Completed Successfully|Post RL Training' "${args[@]}"
      ;;

    pt07_distill_smoke)
      args=(python3 -m "${DISTILL_TRAINER_MODULE}"
        "src/maxtext/configs/post_train/distillation.yml"
        "student_overrides.model_name=gpt3-52k"
        "student_overrides.vocab_size=32000"
        "student_overrides.override_model_config=True"
        "teacher_overrides.model_name=gpt3-52k"
        "teacher_overrides.vocab_size=32000"
        "teacher_overrides.override_model_config=True"
        "teacher_overrides.load_parameters_path=${DISTILL_TEACHER_CKPT}"
        "tokenizer_path=${SFT_TOKENIZER_PATH}"
        "tokenizer_type=huggingface"
        "hf_access_token=${HF_TOKEN:-}"
        "steps=${DISTILL_STEPS}"
        "learning_rate_schedule_steps=${DISTILL_STEPS}"
        "per_device_batch_size=${PER_DEVICE_BATCH_SIZE}"
        "teacher_overrides.per_device_batch_size=${PER_DEVICE_BATCH_SIZE}"
        "ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM}"
        "ici_data_parallelism=${ICI_DATA_PARALLELISM}"
        "weight_dtype=${WEIGHT_DTYPE}"
        "gradient_accumulation_steps=1"
        "profiler=${PROFILER}"
        "log_config=False"
        "enable_goodput_recording=False"
        "pure_nnx=True"
        "enable_nnx=True"
        "pure_nnx_decoder=True"
        "base_output_directory=${BASE_OUTPUT_DIRECTORY}"
        "run_name=${RUN_PREFIX}_07_distill_smoke")
      run_logged "${case_id}" "$(case_description "${case_id}")" 'completed step:|step [0-9]+ \||Distillation metrics' "${args[@]}"
      ;;

    pt09_lora_sft)
      mapfile -t sft_args < <(common_sft_args)
      args=(python3 -m "${SFT_TRAINER_MODULE}" "${sft_args[@]}"
        "pure_nnx=True" "enable_nnx=True" "pure_nnx_decoder=True"
        "lora.enable_lora=True" "lora.lora_rank=8" "lora.lora_alpha=16"
        "lora.lora_module_path=decoder/layers/(self_attention/(qkv_proj|out)|mlp/(wi|wo))"
        "run_name=${RUN_PREFIX}_09_lora_sft")
      run_logged "${case_id}" "$(case_description "${case_id}")" 'completed step:' "${args[@]}"
      ;;

    *)
      die "Unhandled case ${case_id}"
      ;;
  esac
}

run_local() {
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_DIR}"
  printf 'case\tstatus\treason\tlog\n' > "${SUMMARY_FILE}"
  trap 'write_log_dump "$?"' EXIT

  mapfile -t cases_to_run < <(selected_cases)
  preflight "${cases_to_run[@]}"
  print_startup

  local case_id
  local failures=0
  for case_id in "${cases_to_run[@]}"; do
    if ! run_case "${case_id}"; then
      failures=$((failures + 1))
      [ "${CONTINUE_ON_FAILURE}" = "1" ] || break
    fi
  done

  printf '\nSummary: %s\n' "${SUMMARY_FILE}"
  column -t -s $'\t' "${SUMMARY_FILE}" 2>/dev/null || cat "${SUMMARY_FILE}"

  if [ "${failures}" -gt 0 ]; then
    die "${failures} case(s) failed."
  fi
}

remote_env_assignment() {
  local name="$1"
  local value="$2"
  printf '%s=%s ' "${name}" "$(quote "${value}")"
}

run_remote() {
  [ -n "${TPU_NAME:-}" ] || die "TPU_NAME is required for MODE=remote."
  [ -n "${TPU_ZONE:-}" ] || die "TPU_ZONE is required for MODE=remote."
  [ -n "${REMOTE_REPO_DIR:-}" ] || die "REMOTE_REPO_DIR is required for MODE=remote."
  command -v gcloud >/dev/null 2>&1 || die "gcloud is not available."

  local gcloud_flags=(--zone="${TPU_ZONE}")
  if [ -n "${TPU_PROJECT:-}" ]; then
    gcloud_flags+=(--project="${TPU_PROJECT}")
  fi
  if [ -n "${TPU_WORKER}" ]; then
    gcloud_flags+=(--worker="${TPU_WORKER}")
  fi

  local remote_script="${REMOTE_SCRIPT_PATH}"
  if [ -z "${remote_script}" ]; then
    remote_script="${REMOTE_REPO_DIR%/}/tests/end_to_end/tpu/run_post_train_nnx_smoke.sh"
  fi

  if [ "${REMOTE_COPY_SCRIPT}" = "1" ]; then
    printf 'Copying runner to %s:%s\n' "${TPU_NAME}" "${remote_script}"
    gcloud compute tpus tpu-vm scp "${BASH_SOURCE[0]}" "${TPU_NAME}:${remote_script}" "${gcloud_flags[@]}"
  fi

  local remote_cmd=""
  remote_cmd+="cd $(quote "${REMOTE_REPO_DIR}") && "
  remote_cmd+="$(remote_env_assignment MODE local)"
  remote_cmd+="$(remote_env_assignment CASES "${CASES}")"
  remote_cmd+="$(remote_env_assignment RUN_ID "${RUN_ID}")"
  remote_cmd+="$(remote_env_assignment RUN_PREFIX "${RUN_PREFIX}")"
  remote_cmd+="$(remote_env_assignment BASE_OUTPUT_DIRECTORY "${BASE_OUTPUT_DIRECTORY}")"
  remote_cmd+="$(remote_env_assignment CONTINUE_ON_FAILURE "${CONTINUE_ON_FAILURE}")"
  remote_cmd+="$(remote_env_assignment SKIP_MISSING_CASES "${SKIP_MISSING_CASES}")"
  remote_cmd+="$(remote_env_assignment STRICT_DOC_PROGRESS "${STRICT_DOC_PROGRESS}")"
  remote_cmd+="$(remote_env_assignment DUMP_LOGS "${DUMP_LOGS}")"
  remote_cmd+="$(remote_env_assignment LOG_TAIL_LINES "${LOG_TAIL_LINES}")"
  remote_cmd+="$(remote_env_assignment PRINT_LOG_DUMP "${PRINT_LOG_DUMP}")"
  remote_cmd+="$(remote_env_assignment ICI_FSDP_PARALLELISM "${ICI_FSDP_PARALLELISM}")"
  remote_cmd+="$(remote_env_assignment ICI_DATA_PARALLELISM "${ICI_DATA_PARALLELISM}")"
  remote_cmd+="$(remote_env_assignment CHIPS_PER_VM "${CHIPS_PER_VM}")"
  remote_cmd+="$(remote_env_assignment RL_ROLLOUT_DATA_PARALLELISM "${RL_ROLLOUT_DATA_PARALLELISM}")"
  remote_cmd+="$(remote_env_assignment RL_ROLLOUT_TENSOR_PARALLELISM "${RL_ROLLOUT_TENSOR_PARALLELISM}")"
  remote_cmd+="$(remote_env_assignment RL_ROLLOUT_EXPERT_PARALLELISM "${RL_ROLLOUT_EXPERT_PARALLELISM}")"
  remote_cmd+="$(remote_env_assignment NNX_GPT3_52K_CKPT "${NNX_GPT3_52K_CKPT}")"
  remote_cmd+="$(remote_env_assignment LINEN_GPT3_52K_CKPT "${LINEN_GPT3_52K_CKPT}")"
  remote_cmd+="$(remote_env_assignment DISTILL_TEACHER_CKPT "${DISTILL_TEACHER_CKPT}")"
  if [ "${PASS_HF_TOKEN_TO_REMOTE}" = "1" ] && [ -n "${HF_TOKEN:-}" ]; then
    remote_cmd+="$(remote_env_assignment HF_TOKEN "${HF_TOKEN}")"
  fi
  remote_cmd+="bash $(quote "${remote_script}")"

  printf 'Running remote smoke suite on %s...\n' "${TPU_NAME}"
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" "${gcloud_flags[@]}" --command="${remote_cmd}"
}

case "${MODE}" in
  local)
    run_local
    ;;
  remote)
    run_remote
    ;;
  list)
    list_cases
    ;;
  validate)
    mapfile -t cases_to_validate < <(selected_cases)
    preflight "${cases_to_validate[@]}"
    printf 'Preflight passed for %s\n' "${CASES}"
    ;;
  *)
    die "Unknown MODE='${MODE}'. Use local, remote, list, or validate."
    ;;
esac
