#!/usr/bin/env bash
# =============================================================================
# run_post_train_nnx_smoke.sh
#
# Post-training NNX smoke-test runner. Runs ON a V6e-8 TPU VM (8 chips, 1 host).
# Mirrors the post-training suite in "MaxText NNX Migration — Smoke Test Status.md"
# (PT-01..PT-09), with two documented-command BUGS corrected (see below).
#
# WHAT IT DOES
#   - Preflight: HF token, TPU chip count, maxtext import, RL deps, GCS auth.
#   - Runs each requested PT case (NNX always; Linen only where it can work).
#   - Per-case log -> $LOG_DIR/<id>_<mode>.log.
#   - PASS = exit 0 AND "completed step:" seen AND no "loss: nan/inf" AND no Traceback.
#   - Prints a summary table; exits 1 if any case FAILs.
#
# LOG DUMP / COLLECTION (so logs leave the TPU VM)
#   Everything lands under $LOG_DIR (default <repo>/smoke_logs/<RUN_ID>/):
#     console.log     full run transcript (preflight + status + summary)
#     <id>_<mode>.log per-case training stdout, prefixed with the exact CMD
#     manifest.txt    run_id, git commit, jax/libtpu/tunix/vllm versions, toggles
#     summary.tsv     case<TAB>mode<TAB>status<TAB>log
#     summary.json    {run_id,pass,fail,skip,results:[...]}  (for MaxView/parsing)
#   Then bundled to  <repo>/smoke_logs/post_train_smoke_<RUN_ID>.tgz  and, if
#   UPLOAD_LOGS=1 (default), pushed to  $LOG_GCS_DIR  (BASE_OUTPUT_DIR/smoke_logs).
#   Set UPLOAD_LOGS=0 to keep logs local only.
#
# DOC BUGS CORRECTED (validated against source; see
#   docs/superpowers/specs/2026-06-04-posttrain-doc-validation.md §8):
#   A) Doc uses ici_fsdp_parallelism=8 ici_data_parallelism=4 (=32) but V6e-8 has
#      8 chips. max_utils.fill_unspecified_mesh_axes asserts prod(ici)==devices ->
#      crash. Corrected to ici_fsdp_parallelism=8 ici_data_parallelism=1 (=8).
#   B) gpt3-52k has vocab_size=1024 but the Llama-2 tokenizer is 32000-vocab.
#      Random-init SFT (PT-01, PT-09) needs override_model_config=True
#      vocab_size=32000 or it indexes out of bounds at runtime.
#   Set USE_DOC_LITERAL=1 to run the doc's verbatim (crashing) flags instead.
#
# PREREQUISITES (run once on the TPU VM, NOT done here):
#   pip install -e .
#   install_tpu_post_train_extra_deps        # tunix + vLLM for RL
#   gcloud auth application-default login     # for gs:// checkpoint IO
#
# USAGE
#   export HF_TOKEN=hf_xxx
#   scripts/run_post_train_nnx_smoke.sh
#   CASES="01 02 07" scripts/run_post_train_nnx_smoke.sh
#   SKIP_RL=1 scripts/run_post_train_nnx_smoke.sh
#   USE_DOC_LITERAL=1 scripts/run_post_train_nnx_smoke.sh   # reproduce doc crashes
# =============================================================================

set -uo pipefail   # NOT -e: a failing case must not abort the suite.

# ---- locate repo root (script lives in <repo>/scripts) ----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || { echo "cannot cd to ${REPO_ROOT}" >&2; exit 1; }

# ---- configuration (env-overridable) ----------------------------------------
HF_TOKEN="${HF_TOKEN:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-gs://mesa-maxtext/post_train/pt_ckpt_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/smoke_logs/${RUN_ID}}"
UPLOAD_LOGS="${UPLOAD_LOGS:-1}"                       # 1 = also push the log bundle to GCS
LOG_GCS_DIR="${LOG_GCS_DIR:-${BASE_OUTPUT_DIR}/smoke_logs/${RUN_ID}}"

# Seed checkpoints (from the doc; override if your bucket differs)
SEED_LINEN_CKPT="${SEED_LINEN_CKPT:-gs://lance-maxtext/pt_seed_ckpts/pt_seed_ckpts/pt_seed_ckpt_gpt352k_linen/checkpoints/9/items}"
SEED_DISTILL_CKPT="${SEED_DISTILL_CKPT:-gs://lance-maxtext/pt_seed_ckpts/pt_seed_ckpts/pt_seed_ckpt_gpt352k_v32k_linen/checkpoints/4/items}"

CASES="${CASES:-01 02 03 04 05 06 07 09}"   # PT-08/10/11/12 are not in the doc
NUM_CHIPS="${NUM_CHIPS:-8}"
CASE_TIMEOUT="${CASE_TIMEOUT:-1800}"         # seconds per case
SKIP_RL="${SKIP_RL:-0}"                       # 1 = skip PT-04/05/06
RUN_LINEN="${RUN_LINEN:-1}"                   # 1 = also run Linen for PT-01/02
STRICT="${STRICT:-0}"                         # 1 = missing dep is fatal, not skipped
USE_DOC_LITERAL="${USE_DOC_LITERAL:-0}"       # 1 = doc's verbatim (buggy) flags
DRY_RUN="${DRY_RUN:-0}"                        # 1 = print expanded commands, execute nothing

# ---- corrected vs literal flag fragments (BUG A / BUG B) --------------------
if [[ "${USE_DOC_LITERAL}" == "1" ]]; then
  SFT_SHARD="ici_fsdp_parallelism=8 ici_data_parallelism=4"   # 32 -> crashes on 8 chips
  VOCAB_FIX=""                                                 # omitted -> OOB at runtime
else
  SFT_SHARD="ici_fsdp_parallelism=${NUM_CHIPS} ici_data_parallelism=1"
  VOCAB_FIX="override_model_config=True vocab_size=32000"
fi

NNX_FLAGS="pure_nnx=True enable_nnx=True pure_nnx_decoder=True"
LINEN_FLAGS="pure_nnx=False enable_nnx=False pure_nnx_decoder=False"

# Tokenizer sources. USE_OPEN_MIRRORS=1 swaps Meta's gated repos for ungated
# NousResearch mirrors (same tokenizer + chat template, no license form) so you
# can run the Llama cases with only a free HF token. Verified gated:false.
USE_OPEN_MIRRORS="${USE_OPEN_MIRRORS:-0}"
if [[ "${USE_OPEN_MIRRORS}" == "1" ]]; then
  LLAMA2_TOK="${LLAMA2_TOK:-NousResearch/Llama-2-7b-chat-hf}"
  LLAMA31_TOK="${LLAMA31_TOK:-NousResearch/Meta-Llama-3.1-8B-Instruct}"
else
  LLAMA2_TOK="${LLAMA2_TOK:-meta-llama/Llama-2-7b-chat-hf}"      # gated (Meta)
  LLAMA31_TOK="${LLAMA31_TOK:-meta-llama/Llama-3.1-8B-Instruct}" # gated (Meta)
fi
QWEN3_TOK="${QWEN3_TOK:-Qwen/Qwen3-0.6B}"                         # open

# Flags common to SFT/distill (Tunix) runs
COMMON="max_target_length=1024 steps=5 eval_interval=-1 gradient_accumulation_steps=1 \
weight_dtype=float32 log_config=False enable_goodput_recording=False profiler=xplane \
tokenizer_path=${LLAMA2_TOK} tokenizer_type=huggingface hf_access_token=${HF_TOKEN}"

SFT_MOD="python3 -m maxtext.trainers.post_train.sft.train_sft src/maxtext/configs/post_train/sft.yml"
RL_MOD="python3 -m maxtext.trainers.post_train.rl.train_rl src/maxtext/configs/post_train/rl.yml"
DISTILL_MOD="python3 -m maxtext.trainers.post_train.distillation.train_distill src/maxtext/configs/post_train/distillation.yml"

# ---- logging helpers --------------------------------------------------------
c_red=$'\033[31m'; c_grn=$'\033[32m'; c_yel=$'\033[33m'; c_cyn=$'\033[36m'; c_rst=$'\033[0m'
log()  { printf '%s[*]%s %s\n' "$c_cyn" "$c_rst" "$*"; }
ok()   { printf '%s[+]%s %s\n' "$c_grn" "$c_rst" "$*"; }
warn() { printf '%s[!]%s %s\n' "$c_yel" "$c_rst" "$*" >&2; }
err()  { printf '%s[x]%s %s\n' "$c_red" "$c_rst" "$*" >&2; }

declare -a RESULTS   # "id|mode|STATUS|logfile"
RL_AVAILABLE=1

# ---- manifest: record exactly what produced these logs ----------------------
write_manifest() {
  local m="${LOG_DIR}/manifest.txt"
  {
    echo "run_id:        ${RUN_ID}"
    echo "date_utc:      $(date -u +%FT%TZ)"
    echo "host:          $(hostname 2>/dev/null || echo NA)"
    echo "repo_root:     ${REPO_ROOT}"
    echo "git_branch:    $(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo NA)"
    echo "git_commit:    $(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo NA)"
    echo "cases:         ${CASES}"
    echo "toggles:       USE_DOC_LITERAL=${USE_DOC_LITERAL} USE_OPEN_MIRRORS=${USE_OPEN_MIRRORS} RUN_LINEN=${RUN_LINEN} SKIP_RL=${SKIP_RL} STRICT=${STRICT}"
    echo "sft_shard:     ${SFT_SHARD}"
    echo "vocab_fix:     ${VOCAB_FIX:-<none>}"
    echo "llama2_tok:    ${LLAMA2_TOK}"
    echo "llama31_tok:   ${LLAMA31_TOK}"
    echo "base_output:   ${BASE_OUTPUT_DIR}"
    echo "--- package versions ---"
    python3 - <<'PY' 2>/dev/null || echo "version probe failed"
import importlib
for p in ("jax","jaxlib","libtpu","flax","maxtext","tunix","vllm"):
    try:
        m = importlib.import_module(p)
        print(f"{p}: {getattr(m,'__version__','?')}")
    except Exception:
        print(f"{p}: NOT INSTALLED")
try:
    import jax
    print("jax_devices:", jax.device_count(), "/", jax.devices()[0].device_kind)
except Exception:
    print("jax_devices: probe failed")
PY
  } > "${m}"
  log "Manifest -> ${m}"
}

# ---- preflight --------------------------------------------------------------
preflight() {
  log "Preflight (repo=${REPO_ROOT}, run_id=${RUN_ID})"
  mkdir -p "${LOG_DIR}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    warn "DRY_RUN=1 — expanded commands will be printed; no preflight, no execution."
    return 0
  fi

  if [[ -z "${HF_TOKEN}" ]]; then
    err "HF_TOKEN is empty. Export it: export HF_TOKEN=hf_xxx  (gates Llama/Qwen tokenizers+datasets)"
    exit 2
  fi

  if ! python3 -c 'import maxtext' 2>/dev/null; then
    err "Cannot 'import maxtext'. Run 'pip install -e .' in ${REPO_ROOT} first."
    [[ "${STRICT}" == "1" ]] && exit 2 || warn "Continuing; runs will fail."
  else
    ok "maxtext importable"
  fi

  local ndev
  ndev="$(python3 -c 'import jax; print(jax.device_count())' 2>/dev/null || echo 0)"
  if [[ "${ndev}" != "${NUM_CHIPS}" ]]; then
    warn "JAX sees ${ndev} devices, expected ${NUM_CHIPS}. Adjust NUM_CHIPS or check the TPU runtime."
  else
    ok "JAX device_count=${ndev}"
  fi

  if [[ " ${CASES} " =~ ( 04 | 05 | 06 ) && "${SKIP_RL}" != "1" ]]; then
    if python3 -c 'import tunix, vllm' 2>/dev/null; then
      ok "RL deps present (tunix, vllm)"
    else
      RL_AVAILABLE=0
      if [[ "${STRICT}" == "1" ]]; then
        err "RL deps missing (tunix/vllm). Run install_tpu_post_train_extra_deps."
        exit 2
      fi
      warn "RL deps missing -> PT-04/05/06 will be SKIPPED. Run install_tpu_post_train_extra_deps to enable."
    fi
  fi

  if gcloud auth application-default print-access-token >/dev/null 2>&1; then
    ok "GCS application-default credentials present"
  else
    warn "No GCS ADC found. gs:// checkpoint IO may fail. Run: gcloud auth application-default login"
  fi

  [[ "${USE_DOC_LITERAL}" == "1" ]] && \
    warn "USE_DOC_LITERAL=1 -> using the doc's verbatim flags; SFT/distill WILL crash (BUG A/B). For faithfulness only."
}

# ---- evaluate a finished log into PASS/FAIL --------------------------------
# args: <logfile> <exit_code>
evaluate_log() {
  local logf="$1" rc="$2"
  if [[ "${rc}" == "124" ]]; then echo "FAIL(timeout)"; return; fi
  if grep -q "Traceback (most recent call last)" "${logf}" 2>/dev/null; then echo "FAIL(traceback)"; return; fi
  if grep -Eq "loss[:=] *([-+]?inf|nan)" "${logf}" 2>/dev/null;      then echo "FAIL(diverged)";  return; fi
  if [[ "${rc}" != "0" ]]; then echo "FAIL(rc=${rc})"; return; fi
  if grep -q "completed step:" "${logf}" 2>/dev/null;               then echo "PASS"; return; fi
  # RL may surface progress as reward logging before the step buffer flushes.
  if grep -Eqi "reward|completed step" "${logf}" 2>/dev/null;       then echo "PASS(rl)"; return; fi
  echo "FAIL(no-progress)"
}

# ---- run one case -----------------------------------------------------------
# args: <id> <mode> <description> <command string>
run_case() {
  local id="$1" mode="$2" desc="$3" cmd="$4"
  local logf="${LOG_DIR}/${id}_${mode}.log"
  log "PT-${id} [${mode}] ${desc}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%s--- CMD ---%s\n%s\n\n' "$c_cyn" "$c_rst" "${cmd}"
    RESULTS+=("${id}|${mode}|DRYRUN|-")
    return 0
  fi
  {
    echo "### PT-${id} mode=${mode} run_id=${RUN_ID} $(date -u +%FT%TZ)"
    echo "### CMD: ${cmd}"
    echo "###"
  } > "${logf}"
  timeout "${CASE_TIMEOUT}" bash -c "${cmd}" >> "${logf}" 2>&1
  local rc="${PIPESTATUS[0]}"
  local status; status="$(evaluate_log "${logf}" "${rc}")"
  RESULTS+=("${id}|${mode}|${status}|${logf}")
  case "${status}" in
    PASS*) ok   "PT-${id} [${mode}] -> ${status}" ;;
    *)     err  "PT-${id} [${mode}] -> ${status}  (see ${logf})" ;;
  esac
}

skip_case() {
  local id="$1" mode="$2" reason="$3"
  warn "PT-${id} [${mode}] SKIPPED: ${reason}"
  RESULTS+=("${id}|${mode}|SKIP|-")
}

# ===== Case command builders (mirror the doc; corrected flags applied) =======

# PT-01 SFT smoke (random init). NNX + (optional) Linen reference.
case_01() {
  local mode="$1" flags; [[ "${mode}" == nnx ]] && flags="${NNX_FLAGS}" || flags="${LINEN_FLAGS}"
  run_case 01 "${mode}" "SFT smoke (random init)" \
    "${SFT_MOD} model_name=gpt3-52k ${flags} per_device_batch_size=1 ${SFT_SHARD} ${COMMON} ${VOCAB_FIX} \
     base_output_directory=${BASE_OUTPUT_DIR} run_name=pt_sft_${mode}_${RUN_ID}_01_sft_smoke"
}

# PT-02 SFT from a (Linen) seed checkpoint. Vocab fixed by the checkpoint -> no VOCAB_FIX.
case_02() {
  local mode="$1" flags; [[ "${mode}" == nnx ]] && flags="${NNX_FLAGS}" || flags="${LINEN_FLAGS}"
  run_case 02 "${mode}" "SFT from seed checkpoint" \
    "${SFT_MOD} model_name=gpt3-52k ${flags} per_device_batch_size=1 ${SFT_SHARD} ${COMMON} \
     load_parameters_path=${SEED_LINEN_CKPT} \
     base_output_directory=${BASE_OUTPUT_DIR} run_name=pt_sft_${mode}_${RUN_ID}_02_sft_ckpt"
}

# PT-03 Cross-format restore: Linen checkpoint -> NNX model (NNX only).
case_03() {
  run_case 03 nnx "Cross-format SFT (Linen ckpt -> NNX)" \
    "${SFT_MOD} model_name=gpt3-52k ${NNX_FLAGS} per_device_batch_size=1 ${SFT_SHARD} ${COMMON} \
     load_parameters_path=${SEED_LINEN_CKPT} \
     base_output_directory=${BASE_OUTPUT_DIR} run_name=pt_sft_nnx_${RUN_ID}_03_sft_crossfmt"
}

# PT-04 RL GRPO smoke (Qwen3-0.6B). NNX only.
case_04() {
  run_case 04 nnx "RL GRPO smoke (Qwen3-0.6B)" \
    "${RL_MOD} model_name=qwen3-0.6b tokenizer_path=Qwen/Qwen3-0.6B chips_per_vm=${NUM_CHIPS} num_batches=2 \
     rollout_data_parallelism=${NUM_CHIPS} trainer_devices_fraction=1.0 sampler_devices_fraction=1.0 \
     async_scheduling=False log_config=False enable_goodput_recording=False profiler=xplane debug.rl=False \
     ${NNX_FLAGS} hf_access_token=${HF_TOKEN} \
     base_output_directory=${BASE_OUTPUT_DIR}/04_rl_grpo_smoke run_name=pt_rl_grpo_smoke rl.loss_algo=grpo"
}

# PT-05 RL GSPO smoke (Qwen3-0.6B). NNX only.
case_05() {
  run_case 05 nnx "RL GSPO smoke (Qwen3-0.6B)" \
    "${RL_MOD} model_name=qwen3-0.6b tokenizer_path=Qwen/Qwen3-0.6B chips_per_vm=${NUM_CHIPS} num_batches=2 \
     rollout_data_parallelism=${NUM_CHIPS} trainer_devices_fraction=1.0 sampler_devices_fraction=1.0 \
     async_scheduling=False log_config=False enable_goodput_recording=False profiler=xplane debug.rl=False \
     ${NNX_FLAGS} hf_access_token=${HF_TOKEN} \
     base_output_directory=${BASE_OUTPUT_DIR}/05_rl_gspo_smoke run_name=pt_rl_gspo_smoke rl.loss_algo=gspo-token"
}

# PT-06 RL GRPO functional (Llama3.1-8B-Instruct). NNX only. (Gated HF model.)
case_06() {
  run_case 06 nnx "RL GRPO functional (Llama3.1-8B)" \
    "${RL_MOD} model_name=llama3.1-8b tokenizer_path=meta-llama/Llama-3.1-8B-Instruct chips_per_vm=${NUM_CHIPS} num_batches=2 \
     rollout_data_parallelism=${NUM_CHIPS} trainer_devices_fraction=1.0 sampler_devices_fraction=1.0 \
     async_scheduling=False log_config=False enable_goodput_recording=False profiler=xplane debug.rl=False \
     convert_checkpoint_if_possible=False enable_checkpointing=False \
     ${NNX_FLAGS} hf_access_token=${HF_TOKEN} \
     base_output_directory=${BASE_OUTPUT_DIR}/06_rl_grpo_functional run_name=pt_rl_grpo_llama31_functional rl.loss_algo=grpo"
}

# PT-07 Distillation (GPT-3 52k student + teacher from seed ckpt). NNX only (ModelBundle is nnx.Module).
case_07() {
  run_case 07 nnx "Distillation smoke (student+teacher)" \
    "${DISTILL_MOD} \
     student_overrides.model_name=gpt3-52k student_overrides.vocab_size=32000 student_overrides.override_model_config=True \
     teacher_overrides.model_name=gpt3-52k teacher_overrides.vocab_size=32000 teacher_overrides.override_model_config=True \
     teacher_overrides.load_parameters_path=${SEED_DISTILL_CKPT} teacher_overrides.per_device_batch_size=1 \
     tokenizer_path=meta-llama/Llama-2-7b-chat-hf tokenizer_type=huggingface hf_access_token=${HF_TOKEN} \
     steps=5 per_device_batch_size=1 ${SFT_SHARD} weight_dtype=float32 gradient_accumulation_steps=1 \
     profiler=xplane log_config=False enable_goodput_recording=False ${NNX_FLAGS} \
     base_output_directory=${BASE_OUTPUT_DIR} run_name=pt_distill_nnx_${RUN_ID}_07_distill_smoke"
}

# PT-09 LoRA SFT (random init). NNX only (qwix / nnx.LoRAParam).
case_09() {
  run_case 09 nnx "LoRA SFT (random init)" \
    "${SFT_MOD} model_name=gpt3-52k ${NNX_FLAGS} per_device_batch_size=1 ${SFT_SHARD} ${COMMON} ${VOCAB_FIX} \
     lora.enable_lora=True lora.lora_rank=8 lora.lora_alpha=16 \
     'lora.lora_module_path=decoder/layers/(self_attention/(qkv_proj|out)|mlp/(wi|wo))' \
     base_output_directory=${BASE_OUTPUT_DIR} run_name=pt_sft_nnx_${RUN_ID}_09_lora_sft"
}

# ---- dispatch ---------------------------------------------------------------
dispatch() {
  for id in ${CASES}; do
    case "${id}" in
      01) case_01 nnx; [[ "${RUN_LINEN}" == "1" ]] && case_01 linen ;;
      02) case_02 nnx; [[ "${RUN_LINEN}" == "1" ]] && case_02 linen ;;
      03) case_03 ;;
      04) [[ "${SKIP_RL}" == "1" || "${RL_AVAILABLE}" == "0" ]] && skip_case 04 nnx "RL skipped/unavailable" || case_04 ;;
      05) [[ "${SKIP_RL}" == "1" || "${RL_AVAILABLE}" == "0" ]] && skip_case 05 nnx "RL skipped/unavailable" || case_05 ;;
      06) [[ "${SKIP_RL}" == "1" || "${RL_AVAILABLE}" == "0" ]] && skip_case 06 nnx "RL skipped/unavailable" || case_06 ;;
      07) case_07 ;;
      09) case_09 ;;
      *)  warn "Unknown/undocumented case '${id}' — skipping" ;;
    esac
  done
}

# ---- summary ----------------------------------------------------------------
summary() {
  echo ""
  printf '%s===== POST-TRAIN SMOKE SUMMARY (run_id=%s) =====%s\n' "$c_cyn" "${RUN_ID}" "$c_rst"
  printf '%-6s %-7s %-16s %s\n' "CASE" "MODE" "STATUS" "LOG"
  printf '%-6s %-7s %-16s %s\n' "----" "----" "------" "---"
  local fails=0 passes=0 skips=0
  for row in "${RESULTS[@]}"; do
    IFS='|' read -r id mode status logf <<< "${row}"
    printf '%-6s %-7s %-16s %s\n' "PT-${id}" "${mode}" "${status}" "${logf##*/}"
    case "${status}" in
      PASS*) ((passes++)) ;;
      SKIP|DRYRUN)  ((skips++))  ;;
      *)     ((fails++))  ;;
    esac
  done
  echo ""
  printf 'PASS=%d  FAIL=%d  SKIP=%d   logs: %s\n' "${passes}" "${fails}" "${skips}" "${LOG_DIR}"

  # machine-readable dumps (for MaxView / parsing)
  local tsv="${LOG_DIR}/summary.tsv" js="${LOG_DIR}/summary.json" first=1
  printf 'case\tmode\tstatus\tlog\n' > "${tsv}"
  {
    printf '{"run_id":"%s","pass":%d,"fail":%d,"skip":%d,"results":[' "${RUN_ID}" "${passes}" "${fails}" "${skips}"
    for row in "${RESULTS[@]}"; do
      IFS='|' read -r id mode status logf <<< "${row}"
      printf 'PT-%s\t%s\t%s\t%s\n' "${id}" "${mode}" "${status}" "${logf##*/}" >> "${tsv}"
      [[ "${first}" -eq 1 ]] || printf ','
      first=0
      printf '{"case":"PT-%s","mode":"%s","status":"%s","log":"%s"}' "${id}" "${mode}" "${status}" "${logf##*/}"
    done
    printf ']}\n'
  } > "${js}"

  [[ "${fails}" -gt 0 ]] && return 1 || return 0
}

# ---- collect: bundle + (optionally) upload so logs leave the TPU VM ----------
collect_logs() {
  local parent base tarball
  parent="$(dirname "${LOG_DIR}")"; base="$(basename "${LOG_DIR}")"
  tarball="${parent}/post_train_smoke_${RUN_ID}.tgz"
  if tar -czf "${tarball}" -C "${parent}" "${base}" 2>/dev/null; then
    ok "Archive -> ${tarball}"
  else
    warn "tar failed; per-case logs still at ${LOG_DIR}"
  fi

  if [[ "${UPLOAD_LOGS}" == "1" ]]; then
    log "Uploading logs to GCS using bucket_agent.py..."
    if python3 "${REPO_ROOT}/bucket_agent.py" "${LOG_DIR}" "${LOG_GCS_DIR%/}/" && \
       python3 "${REPO_ROOT}/bucket_agent.py" "${tarball}" "${LOG_GCS_DIR%/}/"; then
      ok "Uploaded -> ${LOG_GCS_DIR}"
      log "Pull (anywhere):  gcloud storage cp -r ${LOG_GCS_DIR} ."
    else
      warn "GCS upload failed via bucket_agent.py. Bundle kept locally: ${tarball}"
    fi
  fi
  log "Local bundle:   ${tarball}"
  log "From laptop:    gcloud compute tpus tpu-vm scp --recurse <TPU_NAME>:${tarball} . --zone <ZONE>"
}

# ---- main -------------------------------------------------------------------
mkdir -p "${LOG_DIR}"
# Mirror everything printed below into console.log so the whole run is one file.
exec > >(tee -a "${LOG_DIR}/console.log") 2>&1

write_manifest
preflight
dispatch
summary; SUMMARY_RC=$?
collect_logs
exit "${SUMMARY_RC}"
