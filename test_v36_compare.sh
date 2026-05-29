#!/bin/bash
# V36 Comparison: Baseline vs 3-level custom_vjp (matching Linen's pipeline_utils.py)
#
# V36 ports all 3 levels from pipeline_utils.py:
#   Level 1: Per-microbatch custom_vjp + jax.remat
#   Level 2: Scan over microbatches with d+g gradient accumulation
#   Level 3: Per-repeat weight handling with linear_transpose
# Goal: produce ~8 HLO while loops for better XLA buffer scheduling.
#
# Usage: bash test_v36_compare.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="${SCRIPT_DIR}/src/maxtext/layers"
DECODERS_FILE="${PIPELINE_DIR}/decoders.py"
NNX_DECODERS_FILE="${PIPELINE_DIR}/nnx_decoders.py"
OUTPUT_DIR="${SCRIPT_DIR}/variant_test_results"
DUMP_FILE=""

mkdir -p "$OUTPUT_DIR"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dump) DUMP_FILE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

cp "$DECODERS_FILE" "${DECODERS_FILE}.bak_v36"
cp "$NNX_DECODERS_FILE" "${NNX_DECODERS_FILE}.bak_v36"

restore() {
    cp "${DECODERS_FILE}.bak_v36" "$DECODERS_FILE"
    cp "${NNX_DECODERS_FILE}.bak_v36" "$NNX_DECODERS_FILE"
    rm -f "${DECODERS_FILE}.bak_v36" "${NNX_DECODERS_FILE}.bak_v36"
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

baseline_total="N/A"; baseline_temp="N/A"; baseline_args="N/A"; baseline_output="N/A"
baseline_runtime="N/A"; baseline_tps="N/A"; baseline_loss="N/A"
v36_total="N/A"; v36_temp="N/A"; v36_args="N/A"; v36_output="N/A"
v36_runtime="N/A"; v36_tps="N/A"; v36_loss="N/A"

run_variant() {
    local name=$1
    local pipeline_module=$2
    local log_file="${OUTPUT_DIR}/v36_compare_${name}.log"

    echo "  Running $name..."

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
        run_name="v36_${name}" \
        base_output_directory=/dev/shm/v36_compare/${name} \
        2>&1 | tee "$log_file"

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

    total=${total:-N/A}; output_size=${output_size:-N/A}; temp=${temp:-N/A}; args=${args:-N/A}
    runtime=${runtime:-N/A}; tps=${tps:-N/A}; loss=${loss:-N/A}

    echo ""
    echo "  $name: Total=${total} | Temp=${temp} | Args=${args} | Output=${output_size}"
    echo "  $name: Runtime peak=${runtime} GB | Tokens/s=${tps} | loss=${loss}"
    echo ""

    eval "${name}_total='$total'"
    eval "${name}_temp='$temp'"
    eval "${name}_args='$args'"
    eval "${name}_output='$output_size'"
    eval "${name}_runtime='$runtime'"
    eval "${name}_tps='$tps'"
    eval "${name}_loss='$loss'"
}

echo "================================================================"
echo "V36 COMPARISON: 3-level custom_vjp (matching pipeline_utils.py)"
echo "Date: $(date)"
echo "Branch: $(git branch --show-current 2>/dev/null)"
echo "================================================================"
echo ""
echo "V36 changes (from HLO analysis — Linen has 8 while loops, NNX has 4):"
echo "  Level 1: Per-microbatch custom_vjp + jax.remat → recomputation while loops"
echo "  Level 2: Scan d+g accumulation → separate scan backward loop"
echo "  Level 3: Per-repeat linear_transpose → weight handling loop"
echo "  Goal: ~8 HLO while loops → better XLA buffer scheduling → lower temp"
echo ""

run_variant "baseline" "pipeline"
echo ""
run_variant "v36" "pipeline_v36"

format_summary() {
    echo ""
    echo "================================================================"
    echo "V36 COMPARISON RESULTS"
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
        "v36" "$v36_total" "$v36_temp" "$v36_args" "$v36_output" \
        "$v36_runtime" "$v36_tps" "$v36_loss"
    echo ""
    echo "KEY: If V36 Temp drops toward 13.6 GB (Linen's value),"
    echo "     3-level custom_vjp produces better HLO scheduling."
}

format_summary
if [ -n "$DUMP_FILE" ]; then
    format_summary > "$DUMP_FILE"
    echo "Summary saved to $DUMP_FILE"
fi
