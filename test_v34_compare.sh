#!/bin/bash
# V34 Comparison: Baseline vs V34 (custom_vjp to prevent replicated weight storage)
#
# Compares compile-time memory, runtime memory, throughput, and loss.
# Also dumps HLO buffer sizes to check if replicated weight shapes disappear.
#
# Usage: bash test_v34_compare.sh [--dump FILE]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="${SCRIPT_DIR}/src/maxtext/layers"
DECODERS_FILE="${PIPELINE_DIR}/decoders.py"
NNX_DECODERS_FILE="${PIPELINE_DIR}/nnx_decoders.py"
OUTPUT_DIR="${SCRIPT_DIR}/variant_test_results"
DUMP_FILE=""

mkdir -p "$OUTPUT_DIR"

# Parse --dump flag
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dump) DUMP_FILE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Backup originals
cp "$DECODERS_FILE" "${DECODERS_FILE}.bak_v34"
cp "$NNX_DECODERS_FILE" "${NNX_DECODERS_FILE}.bak_v34"

restore() {
    cp "${DECODERS_FILE}.bak_v34" "$DECODERS_FILE"
    cp "${NNX_DECODERS_FILE}.bak_v34" "$NNX_DECODERS_FILE"
    rm -f "${DECODERS_FILE}.bak_v34" "${NNX_DECODERS_FILE}.bak_v34"
}
trap restore EXIT

COMMON_ARGS=(
    src/maxtext/configs/base.yml
    model_name=llama2-7b
    dataset_type=synthetic
    steps=3
    enable_checkpointing=False
    enable_goodput_recording=False
    max_target_length=32
    per_device_batch_size=2
    ici_pipeline_parallelism=2
    num_layers_per_pipeline_stage=1
    num_pipeline_microbatches=4
    pipeline_fsdp_ag_per_repeat=True
    scan_layers_per_stage=False
    enable_nnx=False
    pure_nnx_decoder=False
)

# Declare defaults to avoid unbound variable errors
baseline_total="N/A"; baseline_temp="N/A"; baseline_args="N/A"; baseline_output="N/A"
baseline_runtime="N/A"; baseline_tps="N/A"; baseline_loss="N/A"
v34_total="N/A"; v34_temp="N/A"; v34_args="N/A"; v34_output="N/A"
v34_runtime="N/A"; v34_tps="N/A"; v34_loss="N/A"

run_variant() {
    local name=$1
    local pipeline_module=$2
    local log_file="${OUTPUT_DIR}/v34_compare_${name}.log"

    echo "  Running $name..."

    # Swap imports
    if [ "$pipeline_module" = "pipeline" ]; then
        sed -i.tmp 's/from maxtext.layers import pipeline_v[0-9a-z]* as pipeline/from maxtext.layers import pipeline/' "$DECODERS_FILE"
        sed -i.tmp 's/from maxtext.layers.pipeline_v[0-9a-z]* import/from maxtext.layers.pipeline import/' "$NNX_DECODERS_FILE"
    else
        sed -i.tmp "s/from maxtext.layers import pipeline.*/from maxtext.layers import ${pipeline_module} as pipeline/" "$DECODERS_FILE"
        sed -i.tmp "s/from maxtext.layers.pipeline.* import create_nnx_pipeline/from maxtext.layers.${pipeline_module} import create_nnx_pipeline/" "$NNX_DECODERS_FILE"
    fi
    rm -f "${DECODERS_FILE}.tmp" "${NNX_DECODERS_FILE}.tmp"

    python -m maxtext.trainers.pre_train.train \
        "${COMMON_ARGS[@]}" \
        run_name="v34_${name}" \
        base_output_directory=/dev/shm/v34_compare/${name} \
        2>&1 | tee "$log_file"

    # Extract metrics — all 4 memory fields are on one line, use awk field splitting
    local memline
    memline=$(grep "Total memory size:" "$log_file" | head -1)
    local total=$(echo "$memline" | awk -F'Total memory size: ' '{print $2}' | awk '{print $1}')
    local output_size=$(echo "$memline" | awk -F'Output size: ' '{print $2}' | awk '{print $1}')
    local temp=$(echo "$memline" | awk -F', Temp size: ' '{print $2}' | awk '{print $1}')
    local args=$(echo "$memline" | awk -F'Argument size: ' '{print $2}' | awk '{print $1}')

    local runtime=$(grep "peak=" "$log_file" | head -1 | sed 's/.*peak=\([0-9.]*\) GB.*/\1/')

    local step2line
    step2line=$(grep "completed step: 2" "$log_file" | head -1)
    local tps=$(echo "$step2line" | awk -F'Tokens/s/device: ' '{print $2}' | awk '{print $1}' | tr -d ',')
    local loss=$(echo "$step2line" | awk -F' loss: ' '{print $2}' | awk '{print $1}' | tr -d ',')

    # Default to N/A
    total=${total:-N/A}; output_size=${output_size:-N/A}; temp=${temp:-N/A}; args=${args:-N/A}
    runtime=${runtime:-N/A}; tps=${tps:-N/A}; loss=${loss:-N/A}

    echo ""
    echo "  $name: Total=${total} | Temp=${temp} | Args=${args} | Output=${output_size}"
    echo "  $name: Runtime peak=${runtime} GB | Tokens/s=${tps} | loss=${loss}"

    # Store for comparison
    eval "${name}_total='$total'"
    eval "${name}_temp='$temp'"
    eval "${name}_args='$args'"
    eval "${name}_output='$output_size'"
    eval "${name}_runtime='$runtime'"
    eval "${name}_tps='$tps'"
    eval "${name}_loss='$loss'"
}

echo "================================================================"
echo "V34 COMPARISON: Baseline vs Custom VJP (replicated weight fix)"
echo "Date: $(date)"
echo "Branch: $(git branch --show-current 2>/dev/null)"
echo "================================================================"

run_variant "baseline" "pipeline"
echo ""
run_variant "v34" "pipeline_v34"

# Summary
format_summary() {
    echo ""
    echo "================================================================"
    echo "V34 COMPARISON RESULTS"
    echo "================================================================"
    echo ""
    printf "%-12s %-12s %-12s %-12s %-12s %-12s %-12s %s\n" \
        "Variant" "Total" "Temp" "Args" "Output" "Runtime" "Tokens/s" "Loss"
    printf "%-12s %-12s %-12s %-12s %-12s %-12s %-12s %s\n" \
        "-------" "-----" "----" "----" "------" "-------" "--------" "----"
    printf "%-12s %-12s %-12s %-12s %-12s %-12s %-12s %s\n" \
        "baseline" "$baseline_total" "$baseline_temp" "$baseline_args" "$baseline_output" \
        "$baseline_runtime" "$baseline_tps" "$baseline_loss"
    printf "%-12s %-12s %-12s %-12s %-12s %-12s %-12s %s\n" \
        "v34" "$v34_total" "$v34_temp" "$v34_args" "$v34_output" \
        "$v34_runtime" "$v34_tps" "$v34_loss"
    echo ""
    echo "KEY QUESTION: Does V34's Temp drop significantly?"
    echo "  If yes → replicated weights were the cause"
    echo "  If no  → the gap is from something else"
    echo ""
    echo "AUTHOR'S QUESTION: Is extra memory from FSDP-sharded or replicated weights?"
    echo "  - FSDP-sharded (layers_params): closure const, stored once. Small per device."
    echo "  - Replicated (w_curr/w_next): post-all-gather, full global shape. In outer carry."
    echo "  - V34 adds custom_vjp to prevent replicated weights from being saved as carry residuals."
    echo "  - If V34 Temp drops: extra memory WAS from replicated weights."
    echo "  - If V34 Temp stays same: extra memory is from something else."
    echo ""
    echo "WHAT TO CHECK:"
    echo "  1. Temp size: baseline=${baseline_temp}G vs v34=${v34_temp}G"
    echo "  2. Loss should match (0.049 at step 2)"
    echo "  3. Runtime peak should be similar (~9.93 GB)"
    echo ""
    echo "Logs:"
    echo "  ${OUTPUT_DIR}/v34_compare_baseline.log"
    echo "  ${OUTPUT_DIR}/v34_compare_v34.log"
}

format_summary

if [ -n "$DUMP_FILE" ]; then
    format_summary > "$DUMP_FILE"
    echo "Summary dumped to: $DUMP_FILE"
fi
