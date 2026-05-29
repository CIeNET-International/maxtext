#!/bin/bash
# V37/V38/V39 Comparison: 3 structural approaches to match Linen's HLO scheduling
#
# V37: Decoder-level scan (move scan outside __call__, checkpoint at outer level)
# V38: Zero-closure scan (all JAX arrays explicit in carry, num_consts≈0)
# V39: Multiple while_loops (replace scan with explicit while_loop calls)
#
# Usage: bash test_v37_v38_v39_compare.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="${SCRIPT_DIR}/src/maxtext/layers"
DECODERS_FILE="${PIPELINE_DIR}/decoders.py"
NNX_DECODERS_FILE="${PIPELINE_DIR}/nnx_decoders.py"
OUTPUT_DIR="${SCRIPT_DIR}/variant_test_results"

mkdir -p "$OUTPUT_DIR"

cp "$DECODERS_FILE" "${DECODERS_FILE}.bak_v37"
cp "$NNX_DECODERS_FILE" "${NNX_DECODERS_FILE}.bak_v37"

restore() {
    cp "${DECODERS_FILE}.bak_v37" "$DECODERS_FILE"
    cp "${NNX_DECODERS_FILE}.bak_v37" "$NNX_DECODERS_FILE"
    rm -f "${DECODERS_FILE}.bak_v37" "${NNX_DECODERS_FILE}.bak_v37"
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

# Declare all defaults
for v in baseline v37 v38 v39; do
    eval "${v}_total=N/A; ${v}_temp=N/A; ${v}_tps=N/A; ${v}_loss=N/A"
done

run_variant() {
    local name=$1
    local pipeline_module=$2
    local log_file="${OUTPUT_DIR}/${name}_compare.log"

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
        run_name="${name}" \
        base_output_directory=/dev/shm/v37_39/${name} \
        2>&1 | tee "$log_file"

    local memline=$(grep "Total memory size:" "$log_file" | head -1)
    local total=$(echo "$memline" | awk -F'Total memory size: ' '{print $2}' | awk '{print $1}')
    local temp=$(echo "$memline" | awk -F', Temp size: ' '{print $2}' | awk '{print $1}')
    local step2=$(grep "completed step: 2" "$log_file" | head -1)
    local tps=$(echo "$step2" | awk -F'Tokens/s/device: ' '{print $2}' | awk '{print $1}' | tr -d ',')
    local loss=$(echo "$step2" | awk -F' loss: ' '{print $2}' | awk '{print $1}' | tr -d ',')

    total=${total:-N/A}; temp=${temp:-N/A}; tps=${tps:-N/A}; loss=${loss:-N/A}
    echo "  $name: Total=${total} | Temp=${temp} | Tok/s=${tps} | Loss=${loss}"
    echo ""

    eval "${name}_total='$total'"
    eval "${name}_temp='$temp'"
    eval "${name}_tps='$tps'"
    eval "${name}_loss='$loss'"
}

echo "================================================================"
echo "V37/V38/V39: Structural approaches to match Linen HLO scheduling"
echo "Date: $(date)"
echo "Target: Linen = 23.3 GB Total, 13.6 GB Temp"
echo "================================================================"
echo ""

run_variant "baseline" "pipeline"
run_variant "v37" "pipeline_v37"
run_variant "v38" "pipeline_v38"
run_variant "v39" "pipeline_v39"

echo ""
echo "================================================================"
echo "RESULTS"
echo "================================================================"
echo ""
printf "%-12s %-12s %-12s %-12s %s\n" "Variant" "Total" "Temp" "Tokens/s" "Loss"
printf "%-12s %-12s %-12s %-12s %s\n" "-------" "-----" "----" "--------" "----"
printf "%-12s %-12s %-12s %-12s %s\n" "Linen" "23.3 GB" "13.6 GB" "~180" "(ref)"
printf "%-12s %-12s %-12s %-12s %s\n" "baseline" "$baseline_total" "$baseline_temp" "$baseline_tps" "$baseline_loss"
printf "%-12s %-12s %-12s %-12s %s\n" "v37" "$v37_total" "$v37_temp" "$v37_tps" "$v37_loss"
printf "%-12s %-12s %-12s %-12s %s\n" "v38" "$v38_total" "$v38_temp" "$v38_tps" "$v38_loss"
printf "%-12s %-12s %-12s %-12s %s\n" "v39" "$v39_total" "$v39_temp" "$v39_tps" "$v39_loss"
echo ""
echo "V37: scan outside __call__ + outer jax.checkpoint"
echo "V38: zero-closure scan (all arrays explicit in carry)"
echo "V39: while_loop instead of scan (more HLO boundaries)"
