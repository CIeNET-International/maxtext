#!/bin/bash
# V49 (Solution 6): HLO computation boundary via jax.lax.optimization_barrier.
#
# inner_body / outer_body apply hlo_boundary() (optimization_barrier) to the
# scan carry. optimization_barrier lowers to stablehlo.optimization_barrier,
# which XLA cannot fuse/CSE/inline across -- forcing pre- and post-boundary
# buffers into separate scheduling regions, mimicking Linen _partial_pack
# scope isolation. The barrier is a value- and gradient-identity.
#
# If V49 -> ~23.3 GB: boundary closed the gap. If -> 29.9 GB: it did not.
# Loss MUST match baseline (correctness gate).
#
# Usage: bash test_v49_compare.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="${SCRIPT_DIR}/src/maxtext/layers"
DECODERS_FILE="${PIPELINE_DIR}/decoders.py"
NNX_DECODERS_FILE="${PIPELINE_DIR}/nnx_decoders.py"
OUTPUT_DIR="${SCRIPT_DIR}/variant_test_results"

mkdir -p "$OUTPUT_DIR"

cp "$DECODERS_FILE" "${DECODERS_FILE}.bak_v49"
cp "$NNX_DECODERS_FILE" "${NNX_DECODERS_FILE}.bak_v49"

restore() {
    cp "${DECODERS_FILE}.bak_v49" "$DECODERS_FILE"
    cp "${NNX_DECODERS_FILE}.bak_v49" "$NNX_DECODERS_FILE"
    rm -f "${DECODERS_FILE}.bak_v49" "${NNX_DECODERS_FILE}.bak_v49"
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

baseline_total="N/A"; baseline_temp="N/A"; baseline_tps="N/A"; baseline_loss="N/A"
v49_total="N/A"; v49_temp="N/A"; v49_tps="N/A"; v49_loss="N/A"

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

    local HLO_DIR="${OUTPUT_DIR}/hlo_v49_${name}"
    rm -rf "${HLO_DIR}"
    mkdir -p "${HLO_DIR}"
    unset XLA_FLAGS

    python -m maxtext.trainers.pre_train.train \
        "${COMMON_ARGS[@]}" \
        run_name="${name}" \
        base_output_directory=/dev/shm/v49/${name} \
        dump_hlo=true \
        dump_hlo_local_dir="${HLO_DIR}" \
        dump_hlo_gcs_dir="gs://mesa-maxtext/pipeline_loop_0521/hlo_v49_${name}" \
        dump_hlo_delete_local_after=false \
        dump_jaxpr=true \
        dump_jaxpr_local_dir="${OUTPUT_DIR}/jaxpr_v49_${name}" \
        dump_jaxpr_delete_local_after=false \
        dump_jaxpr_gcs_dir="gs://mesa-maxtext/pipeline_loop_0521/jaxpr_v49_${name}" \
        2>&1 | tee "$log_file"

    local memline=$(grep "Total memory size:" "$log_file" | head -1)
    local total=$(echo "$memline" | awk -F'Total memory size: ' '{print $2}' | awk '{print $1}')
    local temp=$(echo "$memline" | awk -F', Temp size: ' '{print $2}' | awk '{print $1}')
    local step2=$(grep "completed step: 2" "$log_file" | head -1)
    local tps=$(echo "$step2" | awk -F'Tokens/s/device: ' '{print $2}' | awk '{print $1}' | tr -d ',')
    local loss=$(echo "$step2" | awk -F' loss: ' '{print $2}' | awk '{print $1}' | tr -d ',')

    total=${total:-N/A}; temp=${temp:-N/A}; tps=${tps:-N/A}; loss=${loss:-N/A}
    echo ""
    echo "  $name: Total=${total} | Temp=${temp} | Tok/s=${tps} | Loss=${loss}"
    echo ""

    eval "${name}_total='$total'"
    eval "${name}_temp='$temp'"
    eval "${name}_tps='$tps'"
    eval "${name}_loss='$loss'"
}

echo "================================================================"
echo "V49: DEFINITIVE TEST — nn.scan + nn.remat + REAL Linen params"
echo "Date: $(date)"
echo "================================================================"
echo ""
echo "Claim: 6.6 GB gap is from _partial_pack scope isolation."
echo "V49: RepeatStage(nn.Module) with self.variable('params', ...)"
echo "     nn.scan(nn.remat(RepeatStage), variable_broadcast=['params'])"
echo ""
echo "If V49 → ~23.3 GB: claim PROVEN"
echo "If V49 → ~29.9 GB: something else matters beyond _partial_pack"
echo ""

run_variant "baseline" "pipeline"
run_variant "v49" "pipeline_v49"

echo ""
echo "================================================================"
echo "V49 RESULTS"
echo "================================================================"
echo ""
printf "%-12s %-12s %-12s %-12s %s\n" "Variant" "Total" "Temp" "Tokens/s" "Loss"
printf "%-12s %-12s %-12s %-12s %s\n" "-------" "-----" "----" "--------" "----"
printf "%-12s %-12s %-12s %-12s %s\n" "Linen" "23.3 GB" "13.6 GB" "~180" "(ref)"
printf "%-12s %-12s %-12s %-12s %s\n" "baseline" "$baseline_total" "$baseline_temp" "$baseline_tps" "$baseline_loss"
printf "%-12s %-12s %-12s %-12s %s\n" "v49" "$v49_total" "$v49_temp" "$v49_tps" "$v49_loss"
echo ""

# Post-run: check HLO while loop count
echo "=== HLO While Loop Count ==="
for name in baseline v49; do
    hlo=$(find "${OUTPUT_DIR}/hlo_v49_${name}" -name "*jit_train_step*before_optimizations*" 2>/dev/null | head -1)
    if [ -n "$hlo" ]; then
        wc=$(grep -c "while(" "$hlo" 2>/dev/null)
        echo "  $name: $wc while loops (Linen=8, NNX baseline=4)"
    else
        echo "  $name: HLO not found"
    fi
done
echo "================================================================"
