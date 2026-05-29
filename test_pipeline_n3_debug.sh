#!/bin/bash
# Test pipeline_n3_debug.py — heavily-instrumented variant (verbose trace logs).
# Redirects pipeline imports to `pipeline_n3_debug` via sed, runs, restores via trap.
set -uo pipefail   # NOT -e: a failing pytest must not skip the memory runs.

export PIPELINE_N3_DEBUG=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
OUTPUT_DIR="${SCRIPT_DIR}/variant_test_results_n3_debug"
mkdir -p "${OUTPUT_DIR}/logs"

VARIANT="${1:-pipeline_n3_debug}"

# --- Import redirect: point all pipeline import sites at $VARIANT --------------
SWAP_FILES=(
    "src/maxtext/layers/decoders.py"
    "src/maxtext/layers/nnx_decoders.py"
    "tests/integration/pipeline_parallelism_test.py"
)
cleanup_imports() {
    local f
    for f in "${SWAP_FILES[@]}"; do
        [ -f "${f}.n3bak" ] && cp "${f}.n3bak" "$f" && rm -f "${f}.n3bak"
    done
    echo ">>> Restored original pipeline imports."
}
trap 'cleanup_imports' EXIT INT TERM

if [ ! -f "src/maxtext/layers/${VARIANT}.py" ]; then
    echo "ERROR: variant file src/maxtext/layers/${VARIANT}.py not found"; exit 1
fi
for f in "${SWAP_FILES[@]}"; do [ -f "${f}.n3bak" ] || cp "$f" "${f}.n3bak"; done
sed -i.sedbak "s|^from maxtext.layers import pipeline\$|from maxtext.layers import ${VARIANT} as pipeline|" \
    "src/maxtext/layers/decoders.py" "tests/integration/pipeline_parallelism_test.py"
sed -i.sedbak "s|^from maxtext.layers.pipeline import create_nnx_pipeline\$|from maxtext.layers.${VARIANT} import create_nnx_pipeline|" \
    "src/maxtext/layers/nnx_decoders.py"
rm -f src/maxtext/layers/*.sedbak tests/integration/*.sedbak
echo ">>> Redirected pipeline imports -> ${VARIANT}"

echo "=========================================="
echo "PIPELINE N3 TEST SUITE (DEBUG MODE) — variant: ${VARIANT}"
echo "=========================================="
echo "PIPELINE_N3_DEBUG=$PIPELINE_N3_DEBUG"

# Test 1: Unit test (correctness + gradient parity)
echo ""
echo "--- Test 1: Gradient Parity ---"
python -m pytest tests/integration/pipeline_parallelism_test.py::PipelineParallelismTest::test_circular_pipeline_ag_per_repeat -v \
    2>&1 | tee "${OUTPUT_DIR}/logs/pytest_ag_per_repeat.log"
RESULT1=${PIPESTATUS[0]}
if [ $RESULT1 -eq 0 ]; then
    echo "PASS: test_circular_pipeline_ag_per_repeat"
else
    echo "FAIL: test_circular_pipeline_ag_per_repeat (exit code $RESULT1)"
fi

# Test 2: Memory measurement (Linen decoder + NNX pipeline) — debug variant
echo ""
echo "--- Test 2: Memory + Throughput (Linen decoder, debug) ---"
python -m maxtext.trainers.pre_train.train maxtext/configs/base.yml \
    run_name="n3_linen_decoder_debug" \
    base_output_directory="${OUTPUT_DIR}" \
    model_name=llama2-7b \
    dataset_type=synthetic \
    steps=15 \
    debug_sharding=false \
    max_target_length=32 \
    async_checkpointing=false \
    enable_checkpointing=false \
    per_device_batch_size=2 \
    num_pipeline_microbatches=4 \
    ici_pipeline_parallelism=2 \
    num_layers_per_pipeline_stage=1 \
    pipeline_fsdp_ag_per_repeat=true \
    scan_pipeline_iterations=true \
    upload_all_profiler_results=false \
    enable_nnx=false \
    pure_nnx_decoder=false \
    2>&1 | tee "${OUTPUT_DIR}/logs/linen_decoder.log"

echo ""
echo "--- Results ---"
echo "Memory:"
grep "Total memory size:" "${OUTPUT_DIR}/logs/linen_decoder.log" | tail -1
echo "Throughput:"
grep "Tokens/s/device\|tokens_per_second" "${OUTPUT_DIR}/logs/linen_decoder.log" | tail -3

# Test 3: Memory measurement (NNX decoder + NNX pipeline) — debug variant
echo ""
echo "--- Test 3: Memory + Throughput (NNX decoder, debug) ---"
python -m maxtext.trainers.pre_train.train maxtext/configs/base.yml \
    run_name="n3_nnx_decoder_debug" \
    base_output_directory="${OUTPUT_DIR}" \
    model_name=llama2-7b \
    dataset_type=synthetic \
    steps=15 \
    debug_sharding=false \
    max_target_length=32 \
    async_checkpointing=false \
    enable_checkpointing=false \
    per_device_batch_size=2 \
    num_pipeline_microbatches=4 \
    ici_pipeline_parallelism=2 \
    num_layers_per_pipeline_stage=1 \
    pipeline_fsdp_ag_per_repeat=true \
    scan_pipeline_iterations=true \
    upload_all_profiler_results=false \
    enable_nnx=true \
    pure_nnx_decoder=true \
    2>&1 | tee "${OUTPUT_DIR}/logs/nnx_decoder.log"

echo ""
echo "--- Results ---"
echo "Memory:"
grep "Total memory size:" "${OUTPUT_DIR}/logs/nnx_decoder.log" | tail -1
echo "Throughput:"
grep "Tokens/s/device\|tokens_per_second" "${OUTPUT_DIR}/logs/nnx_decoder.log" | tail -3

# Test 4: HLO analysis (compile-only) — debug variant
echo ""
echo "--- Test 4: HLO Dump (compile-only, debug) ---"
python dump_hlo_programmatic.py "${OUTPUT_DIR}/hlo" maxtext/configs/base.yml \
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
    enable_nnx=true \
    pure_nnx_decoder=true \
    2>&1 | tee "${OUTPUT_DIR}/logs/hlo_dump.log"

echo ""
echo "=========================================="
echo "SUMMARY (DEBUG MODE)"
echo "=========================================="
echo "Target: Memory <= 23.3 GB, Throughput >= 116 tok/s"
echo ""
echo "Linen decoder results:"
grep "Total memory size:" "${OUTPUT_DIR}/logs/linen_decoder.log" 2>/dev/null | tail -1 || echo "  NOT RUN"
echo "NNX decoder results:"
grep "Total memory size:" "${OUTPUT_DIR}/logs/nnx_decoder.log" 2>/dev/null | tail -1 || echo "  NOT RUN"
echo ""
echo "Test exit code: $RESULT1 (0=pass)"
