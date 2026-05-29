#!/bin/bash
# Pipeline Variant Unit Test Runner
#
# Runs pipeline_parallelism_test.py::test_circular_pipeline_ag_per_repeat
# for each variant by swapping the pipeline import.
#
# Usage:
#   bash test_pipeline_unit_tests.sh [variant...] [--dump FILE]
#
# Examples:
#   bash test_pipeline_unit_tests.sh baseline v33          # test specific variants
#   bash test_pipeline_unit_tests.sh all                   # test all available variants
#   bash test_pipeline_unit_tests.sh all --dump results.txt # test all + dump report

set -uo pipefail
# NOTE: no -e flag — we want to continue after test failures

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="${SCRIPT_DIR}/src/maxtext/layers"
TEST_FILE="${SCRIPT_DIR}/tests/unit/pipeline_parallelism_test.py"
DECODERS_FILE="${PIPELINE_DIR}/decoders.py"
NNX_DECODERS_FILE="${PIPELINE_DIR}/nnx_decoders.py"
OUTPUT_DIR="${SCRIPT_DIR}/variant_test_results"
RESULTS_FILE="${OUTPUT_DIR}/unit_test_summary.txt"
BACKUP_DECODERS="${DECODERS_FILE}.bak_ut"
BACKUP_NNX_DECODERS="${NNX_DECODERS_FILE}.bak_ut"
BACKUP_TEST_FILE="${TEST_FILE}.bak_ut"
DUMP_FILE=""

mkdir -p "$OUTPUT_DIR"

# Check test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "ERROR: Test file not found: $TEST_FILE"
    exit 1
fi

swap_pipeline() {
    local variant=$1

    if [ ! -f "$BACKUP_DECODERS" ]; then
        cp "$DECODERS_FILE" "$BACKUP_DECODERS"
        cp "$NNX_DECODERS_FILE" "$BACKUP_NNX_DECODERS"
        cp "$TEST_FILE" "$BACKUP_TEST_FILE"
    fi

    if [ "$variant" = "baseline" ]; then
        sed -i.tmp 's/from maxtext.layers import pipeline_v[0-9a-z]* as pipeline/from maxtext.layers import pipeline/' "$DECODERS_FILE"
        sed -i.tmp 's/from maxtext.layers.pipeline_v[0-9a-z]* import/from maxtext.layers.pipeline import/' "$NNX_DECODERS_FILE"
        sed -i.tmp 's/from maxtext.layers import pipeline_v[0-9a-z]* as pipeline/from maxtext.layers import pipeline/' "$TEST_FILE"
    else
        local variant_file="pipeline_${variant}"
        if [ ! -f "${PIPELINE_DIR}/${variant_file}.py" ]; then
            echo "SKIP"
            return 1
        fi
        sed -i.tmp "s/from maxtext.layers import pipeline.*/from maxtext.layers import ${variant_file} as pipeline/" "$DECODERS_FILE"
        sed -i.tmp "s/from maxtext.layers.pipeline.* import create_nnx_pipeline/from maxtext.layers.${variant_file} import create_nnx_pipeline/" "$NNX_DECODERS_FILE"
        # Swap in test file too — test imports pipeline directly
        sed -i.tmp "s/from maxtext.layers import pipeline.*/from maxtext.layers import ${variant_file} as pipeline/" "$TEST_FILE"
    fi

    rm -f "${DECODERS_FILE}.tmp" "${NNX_DECODERS_FILE}.tmp" "${TEST_FILE}.tmp"
}

restore_pipeline() {
    if [ -f "$BACKUP_DECODERS" ]; then
        cp "$BACKUP_DECODERS" "$DECODERS_FILE"
        cp "$BACKUP_NNX_DECODERS" "$NNX_DECODERS_FILE"
        cp "$BACKUP_TEST_FILE" "$TEST_FILE"
        rm -f "$BACKUP_DECODERS" "$BACKUP_NNX_DECODERS" "$BACKUP_TEST_FILE"
    fi
}

# Cleanup on exit/interrupt
trap restore_pipeline EXIT

run_unit_test() {
    local variant=$1
    local log_file="${OUTPUT_DIR}/unittest_${variant}.log"

    echo -n "  $variant: "

    swap_pipeline "$variant" || { echo "SKIP (file not found)"; echo "${variant}|SKIP|file not found" >> "$RESULTS_FILE"; return 0; }

    # Run the specific test
    python -m pytest "$TEST_FILE" \
        -k "test_circular_pipeline_ag_per_repeat" \
        -x --tb=short \
        --override-ini="markers=tpu_only: mark test as tpu only" \
        2>&1 > "$log_file"
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "PASS"
        echo "${variant}|PASS|" >> "$RESULTS_FILE"
    elif grep -q "no tests ran\|deselected" "$log_file"; then
        echo "SKIP (test not collected)"
        echo "${variant}|SKIP|test not collected" >> "$RESULTS_FILE"
    else
        # Extract failure reason
        local reason=""
        if grep -q "grad mismatch" "$log_file"; then
            local max_diff=$(grep "abs_diff_max=" "$log_file" | grep -oP 'abs_diff_max=[\d.]+' | head -1)
            local f1_norm=$(grep "f1_grad_norm=" "$log_file" | grep -oP 'f1_grad_norm=[\d.]+' | head -1)
            local f2_norm=$(grep "f2_grad_norm=" "$log_file" | grep -oP 'f2_grad_norm=[\d.]+' | head -1)
            reason="grad mismatch: ${max_diff}, ${f1_norm}, ${f2_norm}"
        elif grep -q "value mismatch" "$log_file"; then
            reason="value mismatch"
        elif grep -q "Traceback" "$log_file"; then
            reason=$(grep -A1 "Error\|Exception" "$log_file" | tail -1 | head -c 80)
        else
            reason="exit code $exit_code"
        fi
        echo "FAIL ($reason)"
        echo "${variant}|FAIL|${reason}" >> "$RESULTS_FILE"
    fi
}

# Parse args
VARIANTS_TO_RUN=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dump)
            DUMP_FILE="$2"
            shift 2
            ;;
        *)
            VARIANTS_TO_RUN+=("$1")
            shift
            ;;
    esac
done

if [ ${#VARIANTS_TO_RUN[@]} -eq 0 ] || [ "${VARIANTS_TO_RUN[0]}" = "all" ]; then
    VARIANTS_TO_RUN=(baseline)
    for f in "${PIPELINE_DIR}"/pipeline_v*.py; do
        [ -f "$f" ] || continue
        v=$(basename "$f" .py | sed 's/pipeline_//')
        VARIANTS_TO_RUN+=("$v")
    done
fi

# Header
echo "================================================================"
echo "Pipeline Unit Test Runner: test_circular_pipeline_ag_per_repeat"
echo "Date: $(date)"
echo "Branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"
echo "Variants: ${#VARIANTS_TO_RUN[@]}"
echo "================================================================"

# Clear previous results
> "$RESULTS_FILE"
echo "variant|status|detail" >> "$RESULTS_FILE"

# Run tests
for variant in "${VARIANTS_TO_RUN[@]}"; do
    run_unit_test "$variant"
done

# Format summary
format_summary() {
    echo ""
    echo "================================================================"
    echo "UNIT TEST SUMMARY — $(date)"
    echo "Test: test_circular_pipeline_ag_per_repeat"
    echo "================================================================"

    local pass=0 fail=0 skip=0
    printf "%-10s %-8s %s\n" "Variant" "Status" "Detail"
    printf "%-10s %-8s %s\n" "-------" "------" "------"
    while IFS='|' read -r variant status detail; do
        [ "$variant" = "variant" ] && continue
        printf "%-10s %-8s %s\n" "$variant" "$status" "$detail"
        case "$status" in
            PASS) pass=$((pass + 1)) ;;
            FAIL) fail=$((fail + 1)) ;;
            SKIP) skip=$((skip + 1)) ;;
        esac
    done < "$RESULTS_FILE"

    echo ""
    echo "Total: $((pass + fail + skip)) | Pass: $pass | Fail: $fail | Skip: $skip"
    echo ""
    echo "Logs: $OUTPUT_DIR/unittest_*.log"
}

format_summary

if [ -n "$DUMP_FILE" ]; then
    format_summary > "$DUMP_FILE"
    echo "Summary dumped to: $DUMP_FILE"
fi
