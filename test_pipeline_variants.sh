#!/bin/bash
# Pipeline Variant Memory Test Runner
#
# Tests pipeline variants for compile-time memory reduction.
# Each variant is a modified pipeline file. The script swaps the import
# in decoders.py, runs train.py, and captures memory_analysis() output.
#
# Usage:
#   bash test_pipeline_variants.sh [variant...] [--dump FILE]
#
# Examples:
#   bash test_pipeline_variants.sh all                    # run default set
#   bash test_pipeline_variants.sh baseline v33           # run specific variants
#   bash test_pipeline_variants.sh v33 --dump report.txt  # dump summary to file
#   bash test_pipeline_variants.sh all --dump results.md  # run all + dump
#
# Results saved to variant_test_results/summary.txt and per-variant .log files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="${SCRIPT_DIR}/src/maxtext/layers"
DECODERS_FILE="${PIPELINE_DIR}/decoders.py"
NNX_DECODERS_FILE="${PIPELINE_DIR}/nnx_decoders.py"
OUTPUT_DIR="${SCRIPT_DIR}/variant_test_results"
RESULTS_FILE="${OUTPUT_DIR}/summary.txt"
BACKUP_DECODERS="${DECODERS_FILE}.bak"
BACKUP_NNX_DECODERS="${NNX_DECODERS_FILE}.bak"
DUMP_FILE=""

mkdir -p "$OUTPUT_DIR"

# Variant descriptions
declare -A DESCRIPTIONS
DESCRIPTIONS[baseline]="Baseline pipeline.py (29.9 GB reference)"
DESCRIPTIONS[v1]="V1: jax.vmap + FrozenDict + stop_gradient (Triple Freeze)"
DESCRIPTIONS[v2]="V2: Single custom_vjp + jax.vmap + axes_scan broadcast"
DESCRIPTIONS[v3]="V3: Remove _stamp_at_current_trace + jax.vmap + explicit ckpt args"
DESCRIPTIONS[v4]="V4: Inner axes_scan broadcast + Outer jax.lax.scan"
DESCRIPTIONS[v5]="V5: FrozenDict + carry promotion + stop_gradient"
DESCRIPTIONS[v6]="V6: Fix vmap return (no param stacking)"
DESCRIPTIONS[v7]="V7: Fix vmap return + stop_gradient mutables"
DESCRIPTIONS[v8]="V8: Fix vmap return + eliminate metric scatter"
DESCRIPTIONS[v9]="V9: Fix vmap + stop_grad + no scatter (3 fixes)"
DESCRIPTIONS[v10]="V10: All five root cause fixes combined"
DESCRIPTIONS[v11]="V11: Params in scan carry + stop_gradient fence"
DESCRIPTIONS[v11b]="V11b: Params in carry, NO stop_gradient"
DESCRIPTIONS[v11c]="V11c: Params in carry, stop_gradient only on carry output"
DESCRIPTIONS[v12]="V12: Explicit-arg checkpoint wrapper (BSW explicit)"
DESCRIPTIONS[v13]="V13: Full scope_fn/repack_fn reproduction (all explicit)"
DESCRIPTIONS[v14]="V14: 3-level custom_vjp (pipeline_utils pattern)"
DESCRIPTIONS[v15]="V15: Targeted custom_vjp w_curr compensation + stop_gradient"
DESCRIPTIONS[v16]="V16: Minimal remat - save only iteration_input"
DESCRIPTIONS[v17]="V17: Inner scan unroll=2"
DESCRIPTIONS[v18]="V18: prevent_cse=True in checkpoint"
DESCRIPTIONS[v19]="V19: Save dots_with_no_batch_dims (save MORE)"
DESCRIPTIONS[v20]="V20: Fully unroll outer repeats"
DESCRIPTIONS[v21]="V21: Remove _stamp_at_current_trace (identity passthrough)"
DESCRIPTIONS[v22]="V22: Scan ys=None (no stacked metrics, match Linen)"
DESCRIPTIONS[v23]="V23: Clean 3-level custom_vjp (Linen pattern, no bsw_ref, no stop_grad)"
DESCRIPTIONS[v24]="V24: jax.named_call around run_one_iteration"
DESCRIPTIONS[v25]="V25: Extra jax.checkpoint inside run_one_iteration"
DESCRIPTIONS[v26]="V26: Split weight fetch from compute (shard_map outside ckpt)"
DESCRIPTIONS[v27]="V27: jax.remat + outer body checkpoint (dual-level remat)"
DESCRIPTIONS[v28]="V28: Pure functional inner body (no self closures)"
DESCRIPTIONS[v29]="V29: NNX partial_pack scan (params explicit to remat)"
DESCRIPTIONS[v30]="V30: Flax axes_scan.scan (broadcast constancy verification)"
DESCRIPTIONS[v31]="V31: Params as explicit arg[0] to jax.checkpoint"
DESCRIPTIONS[v32]="V32: nnx.clone on non-diff state (Issue #5116 pattern)"
DESCRIPTIONS[v33]="V33: Params in INNER scan carry (Flax author suggestion)"

swap_pipeline() {
    local variant=$1

    # Backup originals (first time only)
    if [ ! -f "$BACKUP_DECODERS" ]; then
        cp "$DECODERS_FILE" "$BACKUP_DECODERS"
        cp "$NNX_DECODERS_FILE" "$BACKUP_NNX_DECODERS"
    fi

    if [ "$variant" = "baseline" ]; then
        # Use pipeline.py directly — handle both possible current import patterns
        sed -i.tmp 's/from maxtext.layers import pipeline_v[0-9a-z]* as pipeline/from maxtext.layers import pipeline/' "$DECODERS_FILE"
        sed -i.tmp 's/from maxtext.layers.pipeline_v[0-9a-z]* import/from maxtext.layers.pipeline import/' "$NNX_DECODERS_FILE"
    else
        local variant_file="pipeline_${variant}"
        # Check variant file exists
        if [ ! -f "${PIPELINE_DIR}/${variant_file}.py" ]; then
            echo "ERROR: ${PIPELINE_DIR}/${variant_file}.py not found"
            return 1
        fi
        # Swap imports
        sed -i.tmp "s/from maxtext.layers import pipeline.*/from maxtext.layers import ${variant_file} as pipeline/" "$DECODERS_FILE"
        sed -i.tmp "s/from maxtext.layers.pipeline.* import create_nnx_pipeline/from maxtext.layers.${variant_file} import create_nnx_pipeline/" "$NNX_DECODERS_FILE"
    fi

    # Clean up sed temp files
    rm -f "${DECODERS_FILE}.tmp" "${NNX_DECODERS_FILE}.tmp"

    echo "  Switched to: $variant"
}

restore_pipeline() {
    if [ -f "$BACKUP_DECODERS" ]; then
        cp "$BACKUP_DECODERS" "$DECODERS_FILE"
        cp "$BACKUP_NNX_DECODERS" "$NNX_DECODERS_FILE"
        echo "  Restored original imports"
    fi
}

run_variant() {
    local variant=$1
    local desc="${DESCRIPTIONS[$variant]:-Unknown variant}"

    echo ""
    echo "================================================================"
    echo "  Testing: $variant — $desc"
    echo "================================================================"

    swap_pipeline "$variant"

    local log_file="${OUTPUT_DIR}/variant_${variant}.log"

    echo "  Running train.py..."
    python -m maxtext.trainers.pre_train.train \
        src/maxtext/configs/base.yml \
        run_name="variant_${variant}" \
        model_name=llama2-7b \
        dataset_type=synthetic \
        steps=3 \
        enable_checkpointing=False \
        enable_goodput_recording=False \
        max_target_length=32 \
        per_device_batch_size=2 \
        ici_pipeline_parallelism=2 \
        num_layers_per_pipeline_stage=1 \
        num_pipeline_microbatches=4 \
        pipeline_fsdp_ag_per_repeat=True \
        scan_layers_per_stage=False \
        base_output_directory=/dev/shm/variants/${variant} \
        enable_nnx=False \
        pure_nnx_decoder=False \
        2>&1 | tee "$log_file"

    # Extract memory from log
    local mem=$(grep "Total memory size:" "$log_file" | head -1 | grep -oP '[\d.]+(?= GB)' | head -1)
    local temp=$(grep "Temp size:" "$log_file" | head -1 | grep -oP '[\d.]+(?= GB)' | head -1)
    local loss=$(grep "completed step: 2" "$log_file" | head -1 | grep -oP 'loss: [\d.]+' | head -1)
    local tps=$(grep "completed step: 2" "$log_file" | head -1 | grep -oP 'Tokens/s/device: [\d.]+' | head -1)
    local status="SUCCESS"

    if grep -q "Traceback\|RESOURCE_EXHAUSTED" "$log_file"; then
        status="FAILED"
        mem="ERROR"
        temp="ERROR"
    fi

    echo "  Result: Total=${mem} GB, Temp=${temp} GB, ${loss:-N/A}, Status=${status}"
    echo "${variant}|${mem}|${temp}|${status}|${loss:-N/A}|${tps:-N/A}|${desc}" >> "$RESULTS_FILE"
}

# Parse args: extract --dump flag, rest are variants
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
    # Auto-detect all available variant files
    VARIANTS_TO_RUN=(baseline)
    for f in "${PIPELINE_DIR}"/pipeline_v*.py; do
        [ -f "$f" ] || continue
        v=$(basename "$f" .py | sed 's/pipeline_//')
        VARIANTS_TO_RUN+=("$v")
    done
fi

# Main
echo "Pipeline Variant Memory Test"
echo "Date: $(date)"
echo "Branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"
echo "Target: reduce from 29.9 GB toward 23.3 GB"
echo "Variants: ${VARIANTS_TO_RUN[*]}"
echo ""

# Clear previous results
> "$RESULTS_FILE"
echo "variant|total_gb|temp_gb|status|loss|tps|description" >> "$RESULTS_FILE"

# Run each variant
for variant in "${VARIANTS_TO_RUN[@]}"; do
    run_variant "$variant" || echo "  WARNING: $variant failed"
done

# Restore originals
restore_pipeline

# Format summary
format_summary() {
    echo ""
    echo "================================================================"
    echo "SUMMARY — $(date)"
    echo "Branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"
    echo "================================================================"
    printf "%-10s %-10s %-10s %-10s %-18s %-20s %s\n" "Variant" "Total GB" "Temp GB" "Status" "Loss" "Tok/s" "Description"
    printf "%-10s %-10s %-10s %-10s %-18s %-20s %s\n" "-------" "--------" "-------" "------" "----" "-----" "-----------"
    while IFS='|' read -r variant total temp status loss tps desc; do
        [ "$variant" = "variant" ] && continue  # skip header
        printf "%-10s %-10s %-10s %-10s %-18s %-20s %s\n" "$variant" "$total" "$temp" "$status" "$loss" "$tps" "$desc"
    done < "$RESULTS_FILE"
    echo ""
    echo "Linen baseline (main branch): 23.3 GB"
    echo ""
    echo "Results: $OUTPUT_DIR/"
    echo "  summary.txt     — pipe-separated results"
    echo "  variant_*.log   — full train.py output per variant"
}

# Print summary to stdout
format_summary

# Dump to file if requested
if [ -n "$DUMP_FILE" ]; then
    format_summary > "$DUMP_FILE"
    echo ""
    echo "Summary dumped to: $DUMP_FILE"
fi
