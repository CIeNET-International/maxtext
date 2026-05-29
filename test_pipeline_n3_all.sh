#!/bin/bash
# =============================================================================
# test_pipeline_n3_all.sh
# Unified test runner for ALL pipeline_n3 variants in ONE run (TPU required).
#
# For each variant it runs three checks and writes per-variant logs:
#   1. Correctness  — pytest test_circular_pipeline_ag_per_repeat (gradient parity)
#   2. Memory + throughput — production path (Linen decoder + NNX pipeline)
#   3. HLO dump     — compile-only, extracts memory_summary + while/closed_call counts
# Then prints ONE comparison table.
#
# Module-swap mechanism (IMPORT REDIRECT via sed):
#   The decoder and test import the pipeline at 3 sites:
#     decoders.py:36                      from maxtext.layers import pipeline
#     nnx_decoders.py:70                  from maxtext.layers.pipeline import create_nnx_pipeline
#     pipeline_parallelism_test.py:33     from maxtext.layers import pipeline
#   To activate a variant we sed-rewrite these import lines to point at the variant
#   module (e.g. `from maxtext.layers import pipeline_n3 as pipeline`). pipeline.py is
#   NEVER touched. All 3 files are backed up once as <file>.n3bak; a trap on
#   EXIT/INT/TERM ALWAYS restores them and removes the backups.
#
# Variant files define `NNXCircularPipeline`, `create_pipeline`, and
# `create_nnx_pipeline` identically, so the redirect is safe.
#
# Usage:  bash test_pipeline_n3_all.sh
# Syntax check (no TPU): bash -n test_pipeline_n3_all.sh
# =============================================================================

set -uo pipefail   # NOT -e: one variant failing must not abort the whole suite.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Import sites to redirect (backed up as <file>.n3bak).
SWAP_FILES=(
    "src/maxtext/layers/decoders.py"
    "src/maxtext/layers/nnx_decoders.py"
    "tests/integration/pipeline_parallelism_test.py"
)
RESULTS_DIR="${SCRIPT_DIR}/variant_test_results_n3_all"
CONFIG="maxtext/configs/base.yml"

# Variants to test, in table order. Each is a file in src/maxtext/layers/.
# "pipeline" is the baseline (already the real file — never overlaid).
VARIANTS=(
    "pipeline_n3"
    "pipeline_n3_no_lt"
    "pipeline_n3_no_carry"
    "pipeline_n3_sharding"
    "pipeline"
)

mkdir -p "$RESULTS_DIR"

# -----------------------------------------------------------------------------
# backup_once — back up each swap file ONCE as <file>.n3bak.
# restore_imports — copy backups back over the live files (keeps backups).
# cleanup_imports — restore then delete backups. Wired to trap EXIT/INT/TERM.
# All idempotent; safe to call multiple times.
# -----------------------------------------------------------------------------
backup_once() {
    local f
    for f in "${SWAP_FILES[@]}"; do
        [ -f "${f}.n3bak" ] || cp "$f" "${f}.n3bak"
    done
}

restore_imports() {
    local f
    for f in "${SWAP_FILES[@]}"; do
        [ -f "${f}.n3bak" ] && cp "${f}.n3bak" "$f"
    done
}

cleanup_imports() {
    local f
    for f in "${SWAP_FILES[@]}"; do
        if [ -f "${f}.n3bak" ]; then
            cp "${f}.n3bak" "$f" && rm -f "${f}.n3bak"
        fi
    done
    echo ""
    echo ">>> Restored original imports in all swap files."
}
trap 'cleanup_imports' EXIT INT TERM

# apply_variant_imports <variant> — redirect all 3 import sites to the variant.
# Starts from a clean state (restore_imports) so seds never stack.
apply_variant_imports() {
    local V="$1"
    restore_imports
    # decoders.py + test:  `from maxtext.layers import pipeline` -> `... import <V> as pipeline`
    sed -i.sedbak "s|^from maxtext.layers import pipeline\$|from maxtext.layers import ${V} as pipeline|" \
        "src/maxtext/layers/decoders.py" \
        "tests/integration/pipeline_parallelism_test.py"
    # nnx_decoders.py:  `from maxtext.layers.pipeline import create_nnx_pipeline`
    sed -i.sedbak "s|^from maxtext.layers.pipeline import create_nnx_pipeline\$|from maxtext.layers.${V} import create_nnx_pipeline|" \
        "src/maxtext/layers/nnx_decoders.py"
    rm -f src/maxtext/layers/*.sedbak tests/integration/*.sedbak
    echo ">>> Redirected imports -> ${V}"
    echo "    decoders.py:        $(grep -nE '^from maxtext.layers import .* as pipeline$|^from maxtext.layers import pipeline$' src/maxtext/layers/decoders.py | head -1)"
    echo "    nnx_decoders.py:    $(grep -nE '^from maxtext.layers(\.[a-z0-9_]+)? import create_nnx_pipeline$' src/maxtext/layers/nnx_decoders.py | head -1)"
}

# -----------------------------------------------------------------------------
# Helpers to pull values out of logs with grep. Print "N/A" when missing.
# -----------------------------------------------------------------------------

# extract_memory <logfile>  -> e.g. "23.1 GB" (last "Total memory size:" wins)
extract_memory() {
    local log="$1"
    [ -f "$log" ] || { echo "N/A"; return; }
    local line
    line="$(grep -a "Total memory size:" "$log" 2>/dev/null | tail -1)"
    if [ -z "$line" ]; then echo "N/A"; return; fi
    # Take the token after "size:" plus its unit (e.g. "23.12 GB").
    local val
    val="$(echo "$line" | sed -n 's/.*Total memory size:[[:space:]]*\([0-9.]*[[:space:]]*[A-Za-z]*\).*/\1/p' | head -1)"
    [ -n "$val" ] && echo "$val" || echo "N/A"
}

# extract_throughput <logfile> -> e.g. "117 tok/s" (last reported value)
extract_throughput() {
    local log="$1"
    [ -f "$log" ] || { echo "N/A"; return; }
    local line val
    # Try several keys used across MaxText versions.
    line="$(grep -aiE "per_device_tokens_per_sec|tokens_per_second|Tokens/s|tokens/sec" "$log" 2>/dev/null | tail -1)"
    if [ -n "$line" ]; then
        val="$(echo "$line" | grep -aoE '[0-9]+\.?[0-9]*' | tail -1)"
        [ -n "$val" ] && { echo "${val} tok/s"; return; }
    fi
    echo "N/A"
}

# extract_whileloops <logfile> -> count of "while" loops in the HLO dump output.
extract_whileloops() {
    local log="$1"
    [ -f "$log" ] || { echo "N/A"; return; }
    local val
    # Prefer an explicit summary line if dump_hlo_programmatic emits one.
    val="$(grep -aiE "while[_ ]?loops?[[:space:]:=]+[0-9]+" "$log" 2>/dev/null | grep -aoE '[0-9]+' | tail -1)"
    if [ -n "$val" ]; then echo "$val"; return; fi
    # Fallback: count "while {" occurrences in the dumped HLO if present in the log.
    val="$(grep -ac "while(" "$log" 2>/dev/null)"
    if [ "${val:-0}" -gt 0 ]; then echo "$val"; return; fi
    echo "N/A"
}

# -----------------------------------------------------------------------------
# test_variant <variant_name> — redirect imports (unless baseline), run 3 checks, log all.
# Each check is independent; a failure in one does not stop the others.
# -----------------------------------------------------------------------------
test_variant() {
    local VARIANT="$1"
    local VFILE="src/maxtext/layers/${VARIANT}.py"
    local VDIR="${RESULTS_DIR}/${VARIANT}"
    local OUTDIR="${VDIR}/out"

    echo ""
    echo "============================================================="
    echo "VARIANT: ${VARIANT}"
    echo "============================================================="

    if [ "$VARIANT" = "pipeline" ]; then
        # Baseline: restore original imports so all sites point at pipeline.py.
        restore_imports
        echo ">>> Baseline: original imports (pipeline.py)."
    else
        if [ ! -f "$VFILE" ]; then
            echo "SKIP: ${VARIANT} (file not found)"
            return
        fi
        backup_once
        apply_variant_imports "$VARIANT"
    fi

    mkdir -p "$VDIR" "$OUTDIR"

    # --- Check 1: Correctness (gradient parity) -------------------------------
    echo ""
    echo "--- [1/3] Correctness: test_circular_pipeline_ag_per_repeat ---"
    local PYTEST_LOG="${VDIR}/correctness.log"
    python -m pytest \
        "tests/integration/pipeline_parallelism_test.py::PipelineParallelismTest::test_circular_pipeline_ag_per_repeat" \
        -v 2>&1 | tee "$PYTEST_LOG"
    local TEST_RC=${PIPESTATUS[0]}
    if [ "$TEST_RC" -eq 0 ]; then
        echo "PASS: ${VARIANT} correctness"
    else
        echo "FAIL: ${VARIANT} correctness (exit ${TEST_RC})"
        # Surface the gradient mismatch value if present.
        grep -a "abs_diff_max" "$PYTEST_LOG" 2>/dev/null | tail -3 || true
    fi

    # --- Check 2: Memory + throughput (Linen decoder + NNX pipeline) ----------
    echo ""
    echo "--- [2/3] Memory + Throughput (production path) ---"
    local MEM_LOG="${VDIR}/memory.log"
    python -m maxtext.trainers.pre_train.train "$CONFIG" \
        run_name="n3all_${VARIANT}" \
        base_output_directory="$OUTDIR" \
        model_name=llama2-7b \
        dataset_type=synthetic \
        steps=15 \
        max_target_length=32 \
        per_device_batch_size=2 \
        num_pipeline_microbatches=4 \
        ici_pipeline_parallelism=2 \
        num_layers_per_pipeline_stage=1 \
        pipeline_fsdp_ag_per_repeat=true \
        scan_pipeline_iterations=true \
        enable_checkpointing=false \
        async_checkpointing=false \
        upload_all_profiler_results=false \
        enable_nnx=false \
        pure_nnx_decoder=false \
        2>&1 | tee "$MEM_LOG"
    local MEM_RC=${PIPESTATUS[0]}
    if [ "$MEM_RC" -ne 0 ]; then
        echo "WARN: ${VARIANT} memory run exited ${MEM_RC} (table may show N/A)"
    fi

    # --- Check 3: HLO dump (compile-only, fast) -------------------------------
    echo ""
    echo "--- [3/3] HLO dump (compile-only) ---"
    local HLO_LOG="${VDIR}/hlo.log"
    python dump_hlo_programmatic.py "${OUTDIR}/hlo" "$CONFIG" \
        model_name=llama2-7b \
        dataset_type=synthetic \
        steps=1 \
        max_target_length=32 \
        per_device_batch_size=2 \
        num_pipeline_microbatches=4 \
        ici_pipeline_parallelism=2 \
        num_layers_per_pipeline_stage=1 \
        pipeline_fsdp_ag_per_repeat=true \
        scan_pipeline_iterations=true \
        enable_nnx=false \
        pure_nnx_decoder=false \
        2>&1 | tee "$HLO_LOG"
    local HLO_RC=${PIPESTATUS[0]}
    if [ "$HLO_RC" -ne 0 ]; then
        echo "WARN: ${VARIANT} HLO dump exited ${HLO_RC} (WhileLoops may show N/A)"
    fi
    # If the dump produced an after_opt HLO file, append its while-loop count to the
    # log so extract_whileloops can find it.
    local HLO_FILE="${OUTDIR}/hlo/train_step.after_opt.txt"
    if [ -f "$HLO_FILE" ]; then
        local WL
        WL="$(grep -ac "while(" "$HLO_FILE" 2>/dev/null)"
        echo "while_loops = ${WL}" | tee -a "$HLO_LOG"
    fi

    echo ""
    echo ">>> Done: ${VARIANT}"

    # Restore original imports after each NON-baseline variant so the next
    # redirect starts clean. (Baseline already uses the originals.)
    if [ "$VARIANT" != "pipeline" ]; then
        restore_imports
    fi
}

# -----------------------------------------------------------------------------
# Run all variants.
# -----------------------------------------------------------------------------
echo "============================================================="
echo "PIPELINE N3 — UNIFIED VARIANT TEST SUITE"
echo "Results dir: ${RESULTS_DIR}"
echo "Variants: ${VARIANTS[*]}"
echo "============================================================="

for V in "${VARIANTS[@]}"; do
    test_variant "$V"
done

# -----------------------------------------------------------------------------
# Final comparison table — pull values from the per-variant logs with grep.
# -----------------------------------------------------------------------------
echo ""
echo "============================================================="
echo "VARIANT COMPARISON (target: <=23.3 GB, >=116 tok/s, test PASS)"
echo "============================================================="
printf "%-20s | %-6s | %-9s | %-11s | %s\n" "Variant" "Test" "Memory" "Throughput" "WhileLoops"
printf "%-20s-|-%-6s-|-%-9s-|-%-11s-|-%s\n" "--------------------" "------" "---------" "-----------" "----------"

for V in "${VARIANTS[@]}"; do
    VDIR="${RESULTS_DIR}/${V}"
    VFILE="src/maxtext/layers/${V}.py"

    # Skipped variant (non-baseline file missing): mark and continue.
    if [ "$V" != "pipeline" ] && [ ! -f "$VFILE" ]; then
        printf "%-20s | %-6s | %-9s | %-11s | %s\n" "$V" "SKIP" "N/A" "N/A" "N/A"
        continue
    fi

    PYTEST_LOG="${VDIR}/correctness.log"
    MEM_LOG="${VDIR}/memory.log"
    HLO_LOG="${VDIR}/hlo.log"

    # Test verdict from pytest summary line.
    TEST="N/A"
    if [ -f "$PYTEST_LOG" ]; then
        if grep -aqE "[0-9]+ passed" "$PYTEST_LOG" 2>/dev/null && ! grep -aqE "[0-9]+ failed" "$PYTEST_LOG" 2>/dev/null; then
            TEST="PASS"
        elif grep -aqE "[0-9]+ (failed|error)" "$PYTEST_LOG" 2>/dev/null; then
            TEST="FAIL"
        fi
    fi

    MEM="$(extract_memory "$MEM_LOG")"
    THRU="$(extract_throughput "$MEM_LOG")"
    WL="$(extract_whileloops "$HLO_LOG")"

    printf "%-20s | %-6s | %-9s | %-11s | %s\n" "$V" "$TEST" "$MEM" "$THRU" "$WL"
done

echo "============================================================="
echo "Per-variant logs: ${RESULTS_DIR}/<variant>/{correctness,memory,hlo}.log"
echo "============================================================="

# trap on EXIT runs cleanup_imports — restores all swap files and removes backups.
