#!/bin/bash
# V50 — Solution 7: jax.named_call -> jax.jit(inline=False) function boundary
#
# inner_body (per-microbatch pipeline iteration) is wrapped in
# jax.jit(inline=False) so it is staged out as a distinct named jit/pjit
# sub-computation -- a real jaxpr/HLO function boundary -- instead of being
# inlined into the surrounding jax.lax.scan body.
#
# jax.named_call exists in JAX 0.9.2 but is now only a name-stack annotation
# (source_info_util.extend_name_stack) and gets fully inlined in the jaxpr,
# identical to jax.jit(inline=True) -- zero structural effect (the V45
# mistake). jax.jit(inline=False) instead emits a real, non-inlined jit/pjit
# call equation, so V50 uses that (the task's documented fallback).
#
# Hypothesis: a real jit/pjit function boundary creates a separate XLA
# sub-computation that closes the 6.6 GB gap vs Linen's _partial_pack.
#
# If V50 -> ~23.3 GB: hypothesis proven. If -> 29.9 GB: boundary not enough.
# Loss at step 2 MUST match baseline (jit inline=False is a pure transform).
#
# Usage: bash test_v50_compare.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="${SCRIPT_DIR}/src/maxtext/layers"
DECODERS_FILE="${PIPELINE_DIR}/decoders.py"
NNX_DECODERS_FILE="${PIPELINE_DIR}/nnx_decoders.py"
OUTPUT_DIR="${SCRIPT_DIR}/variant_test_results"

mkdir -p "$OUTPUT_DIR"

cp "$DECODERS_FILE" "${DECODERS_FILE}.bak_v50"
cp "$NNX_DECODERS_FILE" "${NNX_DECODERS_FILE}.bak_v50"

restore() {
    cp "${DECODERS_FILE}.bak_v50" "$DECODERS_FILE"
    cp "${NNX_DECODERS_FILE}.bak_v50" "$NNX_DECODERS_FILE"
    rm -f "${DECODERS_FILE}.bak_v50" "${NNX_DECODERS_FILE}.bak_v50"
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
v50_total="N/A"; v50_temp="N/A"; v50_tps="N/A"; v50_loss="N/A"

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

    local HLO_DIR="${OUTPUT_DIR}/hlo_v50_${name}"
    rm -rf "${HLO_DIR}"
    mkdir -p "${HLO_DIR}"
    unset XLA_FLAGS

    python -m maxtext.trainers.pre_train.train \
        "${COMMON_ARGS[@]}" \
        run_name="${name}" \
        base_output_directory=/dev/shm/v50/${name} \
        dump_hlo=true \
        dump_hlo_local_dir="${HLO_DIR}" \
        dump_hlo_gcs_dir="gs://mesa-maxtext/pipeline_loop_0521/hlo_v50_${name}" \
        dump_hlo_delete_local_after=false \
        dump_jaxpr=true \
        dump_jaxpr_local_dir="${OUTPUT_DIR}/jaxpr_v50_${name}" \
        dump_jaxpr_delete_local_after=false \
        dump_jaxpr_gcs_dir="gs://mesa-maxtext/pipeline_loop_0521/jaxpr_v50_${name}" \
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
echo "V50: jax.named_call -> jax.jit(inline=False) function boundary"
echo "Date: $(date)"
echo "================================================================"
echo ""
echo "jax.named_call in JAX 0.9.2 is only a name-stack annotation and gets"
echo "inlined (V45 mistake). V50 uses jax.jit(inline=False) on inner_body,"
echo "which emits a real non-inlined jit/pjit call equation -- a genuine"
echo "jaxpr/HLO function boundary, mirroring Linen's _partial_pack."
echo ""
echo "If V50 -> ~23.3 GB: hypothesis PROVEN"
echo "If V50 -> ~29.9 GB: function boundary alone is not enough"
echo ""

run_variant "baseline" "pipeline"
run_variant "v50" "pipeline_v50"

echo ""
echo "================================================================"
echo "V50 RESULTS"
echo "================================================================"
echo ""
printf "%-12s %-12s %-12s %-12s %s\n" "Variant" "Total" "Temp" "Tokens/s" "Loss"
printf "%-12s %-12s %-12s %-12s %s\n" "-------" "-----" "----" "--------" "----"
printf "%-12s %-12s %-12s %-12s %s\n" "Linen" "23.3 GB" "13.6 GB" "~180" "(ref)"
printf "%-12s %-12s %-12s %-12s %s\n" "baseline" "$baseline_total" "$baseline_temp" "$baseline_tps" "$baseline_loss"
printf "%-12s %-12s %-12s %-12s %s\n" "v50" "$v50_total" "$v50_temp" "$v50_tps" "$v50_loss"
echo ""

# Post-run: check HLO while loop count
echo "=== HLO While Loop Count ==="
for name in baseline v50; do
    hlo=$(find "${OUTPUT_DIR}/hlo_v50_${name}" -name "*jit_train_step*before_optimizations*" 2>/dev/null | head -1)
    if [ -n "$hlo" ]; then
        wc=$(grep -c "while(" "$hlo" 2>/dev/null)
        echo "  $name: $wc while loops (Linen=8, NNX baseline=4)"
    else
        echo "  $name: HLO not found"
    fi
done

# Post-run: check jaxpr for function boundaries
echo ""
echo "=== Jaxpr Function Boundary Check ==="
for name in baseline v50; do
    jaxpr_dir="${OUTPUT_DIR}/jaxpr_v50_${name}"
    jaxpr_file=$(find "$jaxpr_dir" -name "*.jaxpr" 2>/dev/null | head -1)
    if [ -n "$jaxpr_file" ]; then
        # Count occurrences of 'closed_call' / 'pjit[' / 'jit[' primitives --
        # these indicate separate jaxpr function boundaries (what
        # jax.jit(inline=False) emits). In JAX 0.9.2 the jit/pjit call
        # primitive renders as 'jit[' in the jaxpr.
        closed_calls=$(grep -c "closed_call\|pjit\[\|jit\[" "$jaxpr_file" 2>/dev/null || echo 0)
        scan_count=$(grep -c "scan\[" "$jaxpr_file" 2>/dev/null || echo 0)
        echo "  $name: $closed_calls closed_call/pjit/jit, $scan_count scan primitives"
        echo "         (V50 should show MORE closed_call/pjit/jit than baseline)"
    else
        echo "  $name: jaxpr not found"
    fi
done
echo "================================================================"
