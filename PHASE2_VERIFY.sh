#!/usr/bin/env bash
# =============================================================================
# L2 gradient accumulation verification harness — TPU host only
# =============================================================================
# Tests the `use_nnx_pipeline_l2_grad_accum` flag (G3 approach: outer repeat
# loop unrolled to Python for-loop, inner microbatch loop stays jax.lax.scan).
#
# Verification matrix:
#   (a) Baseline: L2 OFF, NNX pipeline     — expect ~32.7 GB (current regression)
#   (b) Linen ref: Linen pipeline          — expect ~23.3 GB (target)
#   (c) L2 ON: NNX pipeline + L2 flag      — expect ~25-28 GB (memory improvement)
#   (d) L2 ON + Linen dec: mixed path      — should not crash
#   (e) L2 ON + scan_layers=true           — scan_layers interaction check
#
# Usage (TPU host):
#   bash PHASE2_VERIFY.sh
#
# Env vars:
#   WORKTREE     absolute path (default: $(pwd))
#   STAMP        timestamp tag (default: date +%m%d_%H%M)
#   MODEL_NAME   default llama2-7b
#   SKIP_PYTEST  set to 1 to skip correctness gate
# =============================================================================

set -uo pipefail

WORKTREE="${WORKTREE:-$(pwd)}"
STAMP="${STAMP:-$(date +%m%d_%H%M)}"
MODEL_NAME="${MODEL_NAME:-llama2-7b}"
SKIP_PYTEST="${SKIP_PYTEST:-0}"

cd "$WORKTREE" || { echo "ERROR: $WORKTREE not found"; exit 2; }

OUT_DIR="${WORKTREE}/pipeline_l2_verify_${STAMP}"
HLO_DIR_BASE="${OUT_DIR}/hlo"
mkdir -p "${OUT_DIR}" "${HLO_DIR_BASE}"

MASTER_LOG="${OUT_DIR}/master_script_execution.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

FAILED_FILE="${OUT_DIR}/failed_runs.txt"
: > "${FAILED_FILE}"

DATE_STR=$(date +%Y%m%d)
GCS_BASE_OUTPUT="/dev/shm/l2_verify_${STAMP}"
mkdir -p "${GCS_BASE_OUTPUT}"

# -----------------------------------------------------------------------------
# Step 0: correctness gate — pytest
# -----------------------------------------------------------------------------
if [[ "${SKIP_PYTEST}" != "1" ]]; then
  PYTEST_LOG="${OUT_DIR}/pytest_l2_off.log"
  echo "=== pytest circular (L2 OFF, default) -> ${PYTEST_LOG}"
  pytest -xvs tests/unit/pipeline_parallelism_test.py \
    -k "circular_pipeline_ag_per_repeat and not l2" \
    > "${PYTEST_LOG}" 2>&1
  if [[ $? -ne 0 ]]; then
    echo "PYTEST_L2_OFF_FAILED" >> "${FAILED_FILE}"
  fi

  PYTEST_LOG="${OUT_DIR}/pytest_l2_on.log"
  echo "=== pytest L2 ON -> ${PYTEST_LOG}"
  pytest -xvs tests/unit/pipeline_parallelism_test.py \
    -k "l2_grad_accum" \
    > "${PYTEST_LOG}" 2>&1
  if [[ $? -ne 0 ]]; then
    echo "PYTEST_L2_ON_FAILED" >> "${FAILED_FILE}"
  fi
fi

# -----------------------------------------------------------------------------
# Helper: run one train.py invocation
# -----------------------------------------------------------------------------
run_one() {
  local TAG="$1"
  local ENABLE_NNX="$2"
  local USE_NNX_PIPE="$3"
  local L2_FLAG="$4"        # use_nnx_pipeline_l2_grad_accum
  local EXTRA_ARGS="${5:-}"  # optional extra CLI args

  local RUN_NAME="${MODEL_NAME}_${TAG}_${DATE_STR}"
  local LOG_FILE="${OUT_DIR}/${RUN_NAME}.log"
  local HLO_DIR="${HLO_DIR_BASE}/${RUN_NAME}"
  mkdir -p "${HLO_DIR}"

  export XLA_FLAGS="--xla_dump_to=${HLO_DIR} \
--xla_dump_hlo_module_re=jit_train_step.* \
--xla_dump_hlo_as_text \
--xla_dump_hlo_pass_re=.*"
  export JAX_LOG_COMPILES=1

  echo "------------------------------------------------------------"
  echo "▶ ${RUN_NAME}"
  echo "  enable_nnx=${ENABLE_NNX} use_nnx_pipeline=${USE_NNX_PIPE} l2=${L2_FLAG}"
  echo "  log: ${LOG_FILE}"

  local CMD=(
    python -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml
    run_name="${RUN_NAME}"
    base_output_directory="${GCS_BASE_OUTPUT}/${RUN_NAME}"
    model_name="${MODEL_NAME}"
    dataset_type=synthetic
    steps=15
    debug_sharding=false
    max_target_length=32
    async_checkpointing=true
    enable_checkpointing=false
    profiler=xplane
    upload_all_profiler_results=true
    dump_hlo_local_dir="${HLO_DIR}"
    enable_nnx="${ENABLE_NNX}"
    pure_nnx_decoder="${ENABLE_NNX}"
    use_nnx_pipeline="${USE_NNX_PIPE}"
    use_nnx_pipeline_l2_grad_accum="${L2_FLAG}"
    use_nnx_pipeline_custom_vjp_prefetch=false
    pipeline_fsdp_ag_per_repeat=true
    num_layers_per_pipeline_stage=1
    scan_layers_per_stage=false
    ici_pipeline_parallelism=2
    num_pipeline_microbatches=4
    per_device_batch_size=2
  )

  # Append extra args if provided
  if [[ -n "${EXTRA_ARGS}" ]]; then
    IFS=' ' read -ra EXTRA_ARRAY <<< "${EXTRA_ARGS}"
    CMD+=("${EXTRA_ARRAY[@]}")
  fi

  {
    echo "================================================================="
    echo "RUN NAME : ${RUN_NAME}"
    echo "START    : $(date '+%F %T')"
    echo "L2 FLAG  : use_nnx_pipeline_l2_grad_accum=${L2_FLAG}"
    echo "COMMAND  : ${CMD[*]}"
    echo "================================================================="
  } > "${LOG_FILE}"

  local START=$(date +%s)
  "${CMD[@]}" >> "${LOG_FILE}" 2>&1
  local EXIT_CODE=$?
  local DURATION=$(( $(date +%s) - START ))

  if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "${RUN_NAME} | exit=${EXIT_CODE} | dur=${DURATION}s" >> "${FAILED_FILE}"
    echo "FAILED ${RUN_NAME} (${DURATION}s)"
  else
    echo "OK ${RUN_NAME} (${DURATION}s)"
  fi
}

# -----------------------------------------------------------------------------
# Verification matrix
# -----------------------------------------------------------------------------

# (a) Baseline: L2 OFF, full NNX — expect ~32.7 GB
run_one "a_L2off_NNXDec_NNXPipe" true true false

# (b) Linen reference: L2 OFF, Linen pipeline — expect ~23.3 GB
run_one "b_L2off_LinenDec_LinenPipe" false false false

# (c) L2 ON: full NNX — THE KEY EXPERIMENT — expect ~25-28 GB
run_one "c_L2on_NNXDec_NNXPipe" true true true

# (d) L2 ON: Linen dec + NNX pipe — verify no crash with mixed path
run_one "d_L2on_LinenDec_NNXPipe" false true true

# (e) L2 ON + scan_layers=true — interaction check
run_one "e_L2on_NNXDec_NNXPipe_ScanLayers" true true true "scan_layers=true"

# -----------------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------------
SUMMARY="${OUT_DIR}/summary.tsv"
{
  printf "tag\ttotal_gb\ttemp_gb\tmedian_tflops\n"
  for log in "${OUT_DIR}"/*.log; do
    [[ "$log" == *master_script_execution.log ]] && continue
    [[ "$log" == *pytest_*.log ]] && continue
    tag=$(basename "$log" .log)
    total=$(grep -oE 'Total memory size: [0-9.]+ GB' "$log" | head -1 | grep -oE '[0-9.]+')
    temp=$(grep -oE 'Temp size: [0-9.]+ GB' "$log" | head -1 | grep -oE '[0-9.]+')
    median=$(grep -oE 'completed step: (8|9|1[0-4]),.*TFLOP/s/device: [0-9.]+' "$log" \
             | grep -oE 'TFLOP/s/device: [0-9.]+' | grep -oE '[0-9.]+' \
             | sort -n | awk '{a[NR]=$1} END {if(NR==0)print "NA"; else print a[int((NR+1)/2)]}')
    printf "%s\t%s\t%s\t%s\n" "$tag" "${total:-NA}" "${temp:-NA}" "${median:-NA}"
  done
} > "${SUMMARY}"

# -----------------------------------------------------------------------------
# Pass/fail evaluation
# -----------------------------------------------------------------------------
PASSFAIL="${OUT_DIR}/passfail.txt"
export SUMMARY_PATH="${SUMMARY}"
python3 - <<'PY' >"${PASSFAIL}" 2>&1
import csv, os, sys, pathlib
summary = pathlib.Path(os.environ["SUMMARY_PATH"])
rows = list(csv.DictReader(open(summary), delimiter="\t"))

def pick(tag_substr):
  for r in rows:
    if tag_substr in r["tag"]:
      return r
  return None

def gb(v):
  try: return float(v)
  except Exception: return None

# Known baselines
NNX_SCAN_TOTAL = 32.7   # Current NNX with scan outer loop
NNX_SCAN_TEMP  = 22.9
LINEN_TOTAL    = 23.3    # Linen baseline
LINEN_TEMP     = 13.6
L2_TARGET_TOTAL = 29.0   # Conservative target: at least -3.7 GB from baseline
L2_TARGET_TEMP  = 19.0   # Conservative target: at least -3.9 GB Temp

verdicts = []

# (a) NNX baseline parity
a = pick("a_L2off_NNXDec_NNXPipe")
if a:
  t = gb(a["total_gb"])
  ok = t is not None and abs(t - NNX_SCAN_TOTAL) <= NNX_SCAN_TOTAL * 0.05
  verdicts.append(("a_nnx_baseline",
                   "PASS" if ok else "FAIL",
                   f"total={a['total_gb']} (expected ~{NNX_SCAN_TOTAL} +/-5%)"))

# (b) Linen reference
b = pick("b_L2off_LinenDec_LinenPipe")
if b:
  t = gb(b["total_gb"])
  ok = t is not None and abs(t - LINEN_TOTAL) <= LINEN_TOTAL * 0.05
  verdicts.append(("b_linen_reference",
                   "PASS" if ok else "FAIL",
                   f"total={b['total_gb']} (expected ~{LINEN_TOTAL} +/-5%)"))

# (c) L2 ON memory target — must improve over baseline
c = pick("c_L2on_NNXDec_NNXPipe")
if c:
  total_c = gb(c["total_gb"])
  temp_c = gb(c["temp_gb"])
  # Must be below conservative target
  ok_total = total_c is not None and total_c <= L2_TARGET_TOTAL
  ok_temp = temp_c is not None and temp_c <= L2_TARGET_TEMP
  verdicts.append(("c_l2_total_target",
                   "PASS" if ok_total else "FAIL",
                   f"total={c['total_gb']} (target <= {L2_TARGET_TOTAL} GB)"))
  verdicts.append(("c_l2_temp_target",
                   "PASS" if ok_temp else "FAIL",
                   f"temp={c['temp_gb']} (target <= {L2_TARGET_TEMP} GB)"))
  # How close to Linen?
  if total_c is not None:
    gap = total_c - LINEN_TOTAL
    verdicts.append(("c_gap_to_linen",
                     "INFO",
                     f"gap={gap:.1f} GB (L2={total_c:.1f} vs Linen={LINEN_TOTAL})"))
  # Perf check: must not regress >15% vs baseline
  perf_a = gb(a["median_tflops"]) if a else None
  perf_c = gb(c["median_tflops"])
  if perf_a and perf_c:
    perf_ok = perf_c >= perf_a * 0.85
    verdicts.append(("c_perf_no_regress",
                     "PASS" if perf_ok else "FAIL",
                     f"tflops={c['median_tflops']} (>= 85% of baseline {a['median_tflops']})"))

# (d) Mixed path — must not crash (any total value = pass)
d = pick("d_L2on_LinenDec_NNXPipe")
if d:
  t = gb(d["total_gb"])
  ok = t is not None
  verdicts.append(("d_mixed_no_crash",
                   "PASS" if ok else "FAIL",
                   f"total={d['total_gb']} (must complete without crash)"))

# (e) scan_layers interaction — must not crash
e = pick("e_L2on_NNXDec_NNXPipe_ScanLayers")
if e:
  t = gb(e["total_gb"])
  ok = t is not None
  verdicts.append(("e_scan_layers_compat",
                   "PASS" if ok else "FAIL",
                   f"total={e['total_gb']} (must complete without crash)"))

print("L2 gradient accumulation verification:")
print()
all_pass = True
for name, status, info in verdicts:
  marker = f"[{status}]"
  print(f"  {marker:8s} {name:<28s} {info}")
  if status == "FAIL":
    all_pass = False
print()
print("OVERALL:", "PASS" if all_pass else "FAIL")
sys.exit(0 if all_pass else 1)
PY
PASSFAIL_RC=$?

# -----------------------------------------------------------------------------
# HLO diff: scan (a) vs for-loop (c)
# -----------------------------------------------------------------------------
HLO_DIFF_LOG="${OUT_DIR}/hlo_scan_vs_forloop.txt"
{
  echo "=== HLO diff: scan outer (a) vs for-loop outer (c) ==="
  for tag in a_L2off_NNXDec_NNXPipe c_L2on_NNXDec_NNXPipe; do
    HLO_FILE=$(find "${HLO_DIR_BASE}/${MODEL_NAME}_${tag}_${DATE_STR}" \
      -maxdepth 4 -type f -name 'module_*jit_train_step*.before_optimizations.txt' \
      2>/dev/null | head -1)
    if [[ -n "${HLO_FILE}" && -f "${HLO_FILE}" ]]; then
      echo
      echo "  tag=${tag}"
      echo "  file=${HLO_FILE}"
      echo "    while loops  : $(grep -c '^while(' "${HLO_FILE}" 2>/dev/null || echo 0)"
      echo "    fusions      : $(grep -c '^fused_' "${HLO_FILE}" 2>/dev/null || echo 0)"
      for op in all-gather all-reduce reduce-scatter; do
        printf "    %-20s : %s\n" "${op}" \
          "$(grep -c "${op}" "${HLO_FILE}" 2>/dev/null || echo 0)"
      done
    else
      echo "  ${tag}: HLO file not found"
    fi
  done
} > "${HLO_DIFF_LOG}"

echo
echo "============================================================"
echo "L2 verification artefacts:"
echo "  summary  : ${SUMMARY}"
echo "  pass/fail: ${PASSFAIL}"
echo "  hlo diff : ${HLO_DIFF_LOG}"
echo "  failed   : ${FAILED_FILE}"
echo "  master   : ${MASTER_LOG}"
echo "============================================================"

cat "${PASSFAIL}"
exit ${PASSFAIL_RC}
