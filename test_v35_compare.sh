#!/bin/bash
# V35 Comparison: Baseline vs V35 (remove replicated weights from outer scan carry)
#
# V35 approach: recompute both w_curr and w_next each repeat via all-gather,
# instead of carrying w_curr through the outer scan. Trades 1 extra all-gather
# per repeat for removing ~3.4 GB replicated weights from scan carry.
#
# Usage: bash test_v35_compare.sh [--dump FILE]

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
cp "$DECODERS_FILE" "${DECODERS_FILE}.bak_v35"
cp "$NNX_DECODERS_FILE" "${NNX_DECODERS_FILE}.bak_v35"

restore() {
    cp "${DECODERS_FILE}.bak_v35" "$DECODERS_FILE"
    cp "${NNX_DECODERS_FILE}.bak_v35" "$NNX_DECODERS_FILE"
    rm -f "${DECODERS_FILE}.bak_v35" "${NNX_DECODERS_FILE}.bak_v35"
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
v35_total="N/A"; v35_temp="N/A"; v35_args="N/A"; v35_output="N/A"
v35_runtime="N/A"; v35_tps="N/A"; v35_loss="N/A"

run_variant() {
    local name=$1
    local pipeline_module=$2
    local log_file="${OUTPUT_DIR}/v35_compare_${name}.log"

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
        run_name="v35_${name}" \
        base_output_directory=/dev/shm/v35_compare/${name} \
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
    echo ""

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
echo "V35 COMPARISON: Baseline vs No-w_curr-in-carry"
echo "Date: $(date)"
echo "Branch: $(git branch --show-current 2>/dev/null)"
echo "================================================================"
echo ""
echo "V35 changes:"
echo "  1. Remove w_curr from outer scan carry"
echo "  2. Recompute both w_curr and w_next each repeat (2 all-gathers)"
echo "  3. jax.checkpoint on outer_body (recompute in backward)"
echo "  4. make_inner_body(bsw) factory (BSW as closure, not mutable list)"
echo ""

run_variant "baseline" "pipeline"
echo ""
run_variant "v35" "pipeline_v35"

# Summary
format_summary() {
    echo ""
    echo "================================================================"
    echo "V35 COMPARISON RESULTS"
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
        "v35" "$v35_total" "$v35_temp" "$v35_args" "$v35_output" \
        "$v35_runtime" "$v35_tps" "$v35_loss"
    echo ""
    echo "KEY QUESTIONS:"
    echo "  1. Does V35's Total/Temp drop vs baseline?"
    echo "     If yes → replicated w_curr in carry was the cause"
    echo "     If no  → gap is from _partial_pack / XLA scheduling"
    echo ""
    echo "  2. Does V35 loss match baseline?"
    echo "     Must match for correctness"
    echo ""
    echo "  3. How much throughput regression from extra all-gather?"
    echo "     1 extra all-gather per repeat — acceptable if memory drops"
}

format_summary
if [ -n "$DUMP_FILE" ]; then
    format_summary > "$DUMP_FILE"
    echo "Summary saved to $DUMP_FILE"
fi
