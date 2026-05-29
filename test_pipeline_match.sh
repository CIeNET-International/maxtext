#!/bin/bash
# =============================================================================
# test_pipeline_match.sh
#   Goal: find the NNX circular pipeline whose XLA structure / xplane matches
#   main-branch Linen (~23.3 GB; the nn.remat SEPARATE-recompute-loop fingerprint
#   = 8 before_opt while-loops, vs 4 for pure-jax NNX).
#
#   KEY (CORRECTED by TPU data — the earlier review prediction was inverted):
#   the differentiating path is the PURE-NNX path (enable_nnx=true) — that is
#   where v46 MATCHED Linen at 23.5 GB / before_opt while=8 / 116 tok/s. The
#   ToLinen path (enable_nnx=false) is policy/structure-INSENSITIVE (~32.1 GB for
#   EVERY variant) → it is a CONTROL, not the match. We DUMP BOTH PATHS; the
#   PURE-NNX dump (steps [4]/[5]) is DECISIVE.
#
#   Per variant, DUMP ALL INFO:
#     correctness.log                       — pytest gradient parity
#     mem_nnx.log / mem_tolinen.log         — train both paths (Total memory + tok/s) + xplane
#     dump_hlo_nnx/ , dump_hlo_tolinen/     — after_opt, before_opt, buffer_assignment, memory_summary, cost_analysis
#     structural_nnx.txt / structural_tolinen.txt  — while(before/after), closed_call, all-gather, reduce-scatter, fusion
#     xplane (.xplane.pb)                   — under */tensorboard/.../plugins/profile/ (TensorBoard)
#     provenance.txt                        — variant file path, git blob, active imports, commit
#
#   The "pipeline" row is the BRANCH old-NNX baseline (NNXCircularPipeline, raw
#   jax.lax.scan, ~29.9 GB) — it is NOT Linen. The TRUE Linen baseline comes ONLY
#   from:  git checkout main && bash test_pipeline_match_linen_ref.sh
#
#   Usage:  bash test_pipeline_match.sh                 # all variants
#           bash test_pipeline_match.sh pipeline_v46    # one variant
# =============================================================================
set -uo pipefail   # NOT -e: one failing step must not abort the suite.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
RESULTS_DIR="${SCRIPT_DIR}/variant_test_results_match"
mkdir -p "$RESULTS_DIR"

if [ "$#" -ge 1 ]; then VARIANTS=("$@"); else
  VARIANTS=( "pipeline_v46" "pipeline" "pipeline_n3" "pipeline_v65i" )
fi

SWAP_FILES=(
    "src/maxtext/layers/decoders.py"
    "src/maxtext/layers/nnx_decoders.py"
    "tests/integration/pipeline_parallelism_test.py"
)

# ---- FIX 4: preflight — clear any stale backups from a previous kill -9 ------
# (so the swapped files are pristine and a later `git checkout main` for the
#  Linen ref will not conflict). If a .matchbak exists, restore it first.
preflight_clean() {
    local f
    for f in "${SWAP_FILES[@]}"; do
        [ -f "${f}.matchbak" ] && cp "${f}.matchbak" "$f" && rm -f "${f}.matchbak"
        rm -f "${f}.sedbak"
    done
    rm -f src/maxtext/layers/*.sedbak tests/integration/*.sedbak 2>/dev/null
}
preflight_clean

# ---- import-swap helpers -----------------------------------------------------
backup_once()   { local f; for f in "${SWAP_FILES[@]}"; do [ -f "${f}.matchbak" ] || cp "$f" "${f}.matchbak"; done; }
restore_imports(){ local f; for f in "${SWAP_FILES[@]}"; do [ -f "${f}.matchbak" ] && cp "${f}.matchbak" "$f"; done; }
cleanup_imports(){ local f; for f in "${SWAP_FILES[@]}"; do if [ -f "${f}.matchbak" ]; then cp "${f}.matchbak" "$f" && rm -f "${f}.matchbak"; fi; done; echo ">>> Restored original imports (working tree clean for git checkout)."; }
trap 'cleanup_imports' EXIT INT TERM

apply_variant_imports() {
    local V="$1"; restore_imports
    sed -i.sedbak "s|^from maxtext.layers import pipeline\$|from maxtext.layers import ${V} as pipeline|" \
        "src/maxtext/layers/decoders.py" "tests/integration/pipeline_parallelism_test.py"
    sed -i.sedbak "s|^from maxtext.layers.pipeline import create_nnx_pipeline\$|from maxtext.layers.${V} import create_nnx_pipeline|" \
        "src/maxtext/layers/nnx_decoders.py"
    rm -f src/maxtext/layers/*.sedbak tests/integration/*.sedbak
}

# ---- FIX 2: case-fold — ALWAYS refresh lowercase from capital if capital exists
# (git tracks variants at capital src/MaxText/layers/; Linux import wants lowercase
#  src/maxtext/layers/. Refresh-always avoids a stale lowercase copy masking edits.)
ensure_lowercase_variant() {
    local V="$1"
    if [ -f "src/MaxText/layers/${V}.py" ] && [ "src/MaxText/layers/${V}.py" != "src/maxtext/layers/${V}.py" ]; then
        cp "src/MaxText/layers/${V}.py" "src/maxtext/layers/${V}.py" && echo ">>> case-fold: refreshed src/maxtext/layers/${V}.py from capital-M source"
    fi
}

# ---- IDENTICAL config block shared with the Linen reference script -----------
# $1 run_name  $2 base_output_directory  $3 enable_nnx  $4 pure_nnx_decoder  $5 steps  $6 extra
pipeline_cfg() {
  echo "maxtext/configs/base.yml \
    run_name=$1 base_output_directory=$2 \
    model_name=llama2-7b dataset_type=synthetic \
    steps=$5 max_target_length=32 per_device_batch_size=2 \
    num_pipeline_microbatches=4 ici_pipeline_parallelism=2 num_layers_per_pipeline_stage=1 \
    pipeline_fsdp_ag_per_repeat=true scan_pipeline_iterations=true \
    enable_checkpointing=false async_checkpointing=false upload_all_profiler_results=false \
    managed_mldiagnostics=false enable_nnx=$3 pure_nnx_decoder=$4 $6"
}

count_structural() {  # $1 hlo_dir  $2 out_file  $3 label
  local H="$1" OUT="$2"; : > "$OUT"
  local AO="$H/train_step.after_opt.txt" BO="$H/train_step.before_opt.txt" BA="$H/buffer_assignment.txt"
  {
    echo "# $3 — fingerprint: Linen/V46-ToLinen before_opt while=8; pure-jax NNX=4"
    [ -f "$BO" ] && echo "before_opt while(:  $(grep -c 'while(' "$BO" 2>/dev/null)"
    [ -f "$BA" ] && grep -iE 'While loops in post-opt|peak' "$BA" 2>/dev/null
    if [ -f "$AO" ]; then
      echo "after_opt  while(:        $(grep -c 'while(' "$AO" 2>/dev/null)"
      echo "after_opt  closed_call:   $(grep -c 'closed_call\|@closed_call' "$AO" 2>/dev/null)"
      echo "after_opt  all-gather:    $(grep -c 'all-gather' "$AO" 2>/dev/null)"
      echo "after_opt  reduce-scatter:$(grep -c 'reduce-scatter' "$AO" 2>/dev/null)"
      echo "after_opt  fusion(:       $(grep -c 'fusion(' "$AO" 2>/dev/null)"
      echo "after_opt  bytes/lines:   $(wc -c <"$AO" 2>/dev/null) / $(wc -l <"$AO" 2>/dev/null)"
    fi
    [ -f "$H/memory_summary.txt" ] && grep -iE 'temp_size|peak' "$H/memory_summary.txt" 2>/dev/null
  } >> "$OUT" 2>&1
}

mem_line()  { grep -a "Total memory size:" "$1" 2>/dev/null | tail -1 | sed -n 's/.*Total memory size:[[:space:]]*\([0-9.]*[[:space:]]*GB\).*/\1/p'; }
# steady median of Tokens/s/device (excludes warmup spikes + perplexity=1.000)
thru_line() { grep -aE "Tokens/s/device:" "$1" 2>/dev/null | grep -aoE 'Tokens/s/device: [0-9.]+' | awk '{print $2}' | sort -n | awk '{a[NR]=$1} END{if(NR>2) print a[int(NR/2)]; else if(NR>0) print a[NR]; else print "N/A"}'; }
wbo()       { grep -a 'before_opt while(' "$1" 2>/dev/null | grep -aoE '[0-9]+' | head -1; }

# =============================================================================
test_variant() {
    local V="$1" VFILE="src/maxtext/layers/${1}.py" VDIR="${RESULTS_DIR}/${1}"
    mkdir -p "$VDIR"
    echo ""; echo "============================================================="; echo "VARIANT: ${V}"; echo "============================================================="

    if [ "$V" = "pipeline" ]; then
        restore_imports; echo ">>> baseline = BRANCH old-NNX pipeline.py (NNXCircularPipeline, ~29.9 GB). NOT Linen."
    else
        ensure_lowercase_variant "$V"
        if [ ! -f "$VFILE" ]; then echo "SKIP: ${V} (not at src/maxtext/layers/ nor src/MaxText/layers/)"; return; fi
        backup_once; apply_variant_imports "$V"; echo ">>> imports redirected -> ${V}"
    fi

    { echo "variant: ${V}"; echo "file: ${VFILE}"; echo "git_blob: $(git hash-object "$VFILE" 2>/dev/null)";
      echo "git_tracked: $(git ls-files | grep -i "layers/${V}.py" | head -1)";
      echo "decoders import: $(grep -nE 'import .*as pipeline|import pipeline$' src/maxtext/layers/decoders.py | head -1)";
      echo "nnx_decoders import: $(grep -nE 'import create_nnx_pipeline' src/maxtext/layers/nnx_decoders.py | head -1)";
      echo "commit: $(git rev-parse HEAD 2>/dev/null)"; } > "${VDIR}/provenance.txt" 2>&1

    echo "--- [1] correctness (gradient parity) ---"
    python -m pytest "tests/integration/pipeline_parallelism_test.py::PipelineParallelismTest::test_circular_pipeline_ag_per_repeat" -v \
        2>&1 | tee "${VDIR}/correctness.log"; echo "pytest rc=${PIPESTATUS[0]}"

    # ToLinen path (enable_nnx=false) — CONTROL (~32.1 GB for every variant)
    echo "--- [2] train ToLinen (enable_nnx=false) + xplane  [control ~32.1] ---"
    python -m maxtext.trainers.pre_train.train $(pipeline_cfg "match_${V}_tolinen" "${VDIR}/tolinen_out" false false 10 "profiler=xplane skip_first_n_steps_for_profiler=2 profiler_steps=4") \
        2>&1 | tee "${VDIR}/mem_tolinen.log"
    echo "--- [3] HLO dump ToLinen (compile-only) [control] ---"
    python dump_hlo_programmatic.py "${VDIR}/dump_hlo_tolinen" $(pipeline_cfg "m_${V}_tolinen_hlo" "${VDIR}/dump_hlo_tolinen" false false 1 "") \
        2>&1 | tee "${VDIR}/hlo_tolinen.log"
    count_structural "${VDIR}/dump_hlo_tolinen" "${VDIR}/structural_tolinen.txt" "ToLinen path (enable_nnx=false)"

    # pure-NNX path (enable_nnx=true) — THE DECISIVE arm (where v46 matched Linen 23.5)
    echo "--- [4] train pure-NNX (enable_nnx=true) + xplane  [DECISIVE PATH] ---"
    python -m maxtext.trainers.pre_train.train $(pipeline_cfg "match_${V}_nnx" "${VDIR}/nnx_out" true true 10 "profiler=xplane skip_first_n_steps_for_profiler=2 profiler_steps=4") \
        2>&1 | tee "${VDIR}/mem_nnx.log"
    echo "--- [5] HLO dump pure-NNX (compile-only) [DECISIVE] ---"
    python dump_hlo_programmatic.py "${VDIR}/dump_hlo_nnx" $(pipeline_cfg "m_${V}_nnx_hlo" "${VDIR}/dump_hlo_nnx" true true 1 "") \
        2>&1 | tee "${VDIR}/hlo_nnx.log"
    count_structural "${VDIR}/dump_hlo_nnx" "${VDIR}/structural_nnx.txt" "pure-NNX path (enable_nnx=true)"

    find "${VDIR}" -name "*.xplane.pb" 2>/dev/null > "${VDIR}/xplane_files.txt"
    echo ">>> done ${V}: ToLinen mem=$(mem_line "${VDIR}/mem_tolinen.log") while_bo=$(wbo "${VDIR}/structural_tolinen.txt") | nnx mem=$(mem_line "${VDIR}/mem_nnx.log") while_bo=$(wbo "${VDIR}/structural_nnx.txt")"
    [ "$V" != "pipeline" ] && restore_imports
}

echo "============================================================="
echo "PIPELINE XPLANE-MATCH SUITE — variants: ${VARIANTS[*]}"
echo "Target: PURE-NNX path (enable_nnx=true) <= 23.5 GB AND >= 116 tok/s == Linen. ToLinen arm = control (~32.1)."
echo "Run the Linen baseline separately:  git checkout main && bash test_pipeline_match_linen_ref.sh"
echo "============================================================="
for V in "${VARIANTS[@]}"; do test_variant "$V"; done

# ---- comparison table --------------------------------------------------------
echo ""; echo "============================================================="
echo "COMPARISON — decisive = PURE-NNX path. MEET iff NNX mem<=23.5 GB AND NNX tok/s>=116 (== Linen ref)."
echo "(ToLin mem = control ~32.1 for all; 'pipeline' row = branch old-NNX baseline, NOT Linen)"
echo "============================================================="
printf "%-15s | %-5s | %-9s | %-9s | %-7s | %-11s | %s\n" "Variant" "Test" "NNX mem" "NNX tok/s" "NNX wbo" "ToLin(ctrl)" "MEET"
for V in "${VARIANTS[@]}"; do
    D="${RESULTS_DIR}/${V}"
    T="N/A"; grep -aqE "[0-9]+ passed" "$D/correctness.log" 2>/dev/null && ! grep -aqE "[0-9]+ (failed|error)" "$D/correctness.log" 2>/dev/null && T="PASS"
    grep -aqE "[0-9]+ (failed|error)" "$D/correctness.log" 2>/dev/null && T="FAIL"
    MNX="$(mem_line "$D/mem_nnx.log")"; WNX="$(wbo "$D/structural_nnx.txt")"; TKN="$(thru_line "$D/mem_nnx.log")"
    MTL="$(mem_line "$D/mem_tolinen.log")"
    MEET="?"; mnum="$(echo "$MNX" | grep -oE '[0-9.]+' | head -1)"; tnum="$(echo "${TKN:-0}" | grep -oE '[0-9.]+' | head -1)"
    if [ -n "$mnum" ] && [ -n "$tnum" ]; then awk "BEGIN{exit !($mnum<=23.5 && $tnum>=116)}" && MEET="YES" || MEET="no"; fi
    printf "%-15s | %-5s | %-9s | %-9s | %-7s | %-11s | %s\n" "$V" "${T:-N/A}" "${MNX:-N/A}" "${TKN:-N/A}" "${WNX:-N/A}" "${MTL:-N/A}" "$MEET"
done
echo "============================================================="
echo "Per-variant: ${RESULTS_DIR}/<variant>/{correctness,mem_tolinen,mem_nnx,hlo_*}.log structural_{tolinen,nnx}.txt provenance.txt dump_hlo_{tolinen,nnx}/ + xplane"
echo "Linen baseline (run on main): variant_test_results_match/_linen_ref_main/"
