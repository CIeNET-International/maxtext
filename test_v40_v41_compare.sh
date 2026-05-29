#!/bin/bash
# V40/V41 Comparison: Using nn.scan/nn.remat/_partial_pack in NNX pipeline
#
# V40: pipeline_utils functions (adapter bridges NNX→Linen signature)
# V41: nn.scan(nn.remat(RepeatStage)) with Linen Module wrapper
#
# V41 also needs decoders_v41.py (import change only)
#
# Usage: bash test_v40_v41_compare.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="${SCRIPT_DIR}/src/maxtext/layers"
DECODERS_FILE="${PIPELINE_DIR}/decoders.py"
NNX_DECODERS_FILE="${PIPELINE_DIR}/nnx_decoders.py"
OUTPUT_DIR="${SCRIPT_DIR}/variant_test_results"

mkdir -p "$OUTPUT_DIR"

cp "$DECODERS_FILE" "${DECODERS_FILE}.bak_v40"
cp "$NNX_DECODERS_FILE" "${NNX_DECODERS_FILE}.bak_v40"

restore() {
    cp "${DECODERS_FILE}.bak_v40" "$DECODERS_FILE"
    cp "${NNX_DECODERS_FILE}.bak_v40" "$NNX_DECODERS_FILE"
    rm -f "${DECODERS_FILE}.bak_v40" "${NNX_DECODERS_FILE}.bak_v40"
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

for v in baseline v40 v41; do
    eval "${v}_total=N/A; ${v}_temp=N/A; ${v}_tps=N/A; ${v}_loss=N/A"
done

run_variant() {
    local name=$1
    local pipeline_module=$2
    local decoder_module=${3:-""}  # optional: versioned decoder file
    local log_file="${OUTPUT_DIR}/${name}_compare.log"

    echo "  Running $name..."

    # Swap pipeline imports in both files
    if [ "$pipeline_module" = "pipeline" ]; then
        sed -i.tmp 's/from maxtext.layers import pipeline_v[0-9a-z]* as pipeline/from maxtext.layers import pipeline/' "$DECODERS_FILE"
        sed -i.tmp 's/from maxtext.layers.pipeline_v[0-9a-z]* import/from maxtext.layers.pipeline import/' "$NNX_DECODERS_FILE"
    else
        sed -i.tmp "s/from maxtext.layers import pipeline.*/from maxtext.layers import ${pipeline_module} as pipeline/" "$DECODERS_FILE"
        sed -i.tmp "s/from maxtext.layers.pipeline.* import create_nnx_pipeline/from maxtext.layers.${pipeline_module} import create_nnx_pipeline/" "$NNX_DECODERS_FILE"
    fi
    rm -f "${DECODERS_FILE}.tmp" "${NNX_DECODERS_FILE}.tmp"

    # If versioned decoder, swap imports in models.py for both decoders and nnx_decoders
    if [ -n "$decoder_module" ]; then
        local MODELS_FILE="${SCRIPT_DIR}/src/maxtext/models/models.py"
        local nnx_decoder_module="${decoder_module/decoders/nnx_decoders}"
        if [ -f "$MODELS_FILE" ]; then
            cp "$MODELS_FILE" "${MODELS_FILE}.bak_v40"
            # Swap Linen decoder import
            sed -i.tmp "s/from maxtext.layers import decoders$/from maxtext.layers import ${decoder_module} as decoders/" "$MODELS_FILE"
            # Swap NNX decoder import
            sed -i.tmp "s/from maxtext.layers import nnx_decoders$/from maxtext.layers import ${nnx_decoder_module} as nnx_decoders/" "$MODELS_FILE"
            rm -f "${MODELS_FILE}.tmp"
        fi
    fi

    python -m maxtext.trainers.pre_train.train \
        "${COMMON_ARGS[@]}" \
        run_name="${name}" \
        base_output_directory=/dev/shm/v40_41/${name} \
        2>&1 | tee "$log_file"

    # Restore models.py if we changed it
    if [ -n "$decoder_module" ] && [ -f "${SCRIPT_DIR}/src/maxtext/models/models.py.bak_v40" ]; then
        cp "${SCRIPT_DIR}/src/maxtext/models/models.py.bak_v40" "${SCRIPT_DIR}/src/maxtext/models/models.py"
        rm -f "${SCRIPT_DIR}/src/maxtext/models/models.py.bak_v40"
    fi

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
echo "V40/V41: nn.scan + nn.remat + _partial_pack approaches"
echo "Date: $(date)"
echo "Target: Linen = 23.3 GB Total, 13.6 GB Temp"
echo "================================================================"
echo ""
echo "V40: pipeline_utils functions with NNX adapter (no nn.scan — adapter not nn.Module)"
echo "V41: nn.scan(nn.remat(RepeatStage)) with Linen Module wrapper inside __call__"
echo ""

run_variant "baseline" "pipeline"
run_variant "v40" "pipeline_v40"
run_variant "v41" "pipeline_v41" "decoders_v41"

echo ""
echo "================================================================"
echo "RESULTS"
echo "================================================================"
echo ""
printf "%-12s %-12s %-12s %-12s %s\n" "Variant" "Total" "Temp" "Tokens/s" "Loss"
printf "%-12s %-12s %-12s %-12s %s\n" "-------" "-----" "----" "--------" "----"
printf "%-12s %-12s %-12s %-12s %s\n" "Linen" "23.3 GB" "13.6 GB" "~180" "(ref)"
printf "%-12s %-12s %-12s %-12s %s\n" "baseline" "$baseline_total" "$baseline_temp" "$baseline_tps" "$baseline_loss"
printf "%-12s %-12s %-12s %-12s %s\n" "v40" "$v40_total" "$v40_temp" "$v40_tps" "$v40_loss"
printf "%-12s %-12s %-12s %-12s %s\n" "v41" "$v41_total" "$v41_temp" "$v41_tps" "$v41_loss"
echo ""
echo "V40: pipeline_utils + adapter (jax.lax.scan fallback — _partial_pack NOT triggered)"
echo "V41: nn.scan(nn.remat(RepeatStage)) — _partial_pack triggered at scan boundary"
echo "     RepeatStage has no Linen vars (data via closure/carry) — effect TBD"
