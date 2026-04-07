#!/usr/bin/env bash
# ============================================================
# check_diag_logs.sh
# Parse NNX pipeline diagnostic log lines from train_compile output.
#
# Usage:
#   bash check_diag_logs.sh <log_file>
#   bash check_diag_logs.sh my_maxtext_logs_nnx/train_compile_tpu7x-2048_nnx.log
# ============================================================

LOG_FILE="${1:-my_maxtext_logs_nnx/train_compile_tpu7x-2048_nnx.log}"

if [[ ! -f "$LOG_FILE" ]]; then
  echo "Log file not found: $LOG_FILE"
  exit 1
fi

echo "=== Checking: $LOG_FILE ==="
echo ""

# ----------------------------------------------------------------
# Q1: kernel is None guards — should NEVER fire if BSW fix is correct.
# ----------------------------------------------------------------
echo "--- Q1: kernel=None diagnostic hits (expect 0) ---"
KERNEL_NONE=$(grep "\[DIAG kernel_none\]" "$LOG_FILE")
if [[ -z "$KERNEL_NONE" ]]; then
  echo "  OK: No kernel=None hits. BSW fix is working correctly."
else
  echo "  WARNING: kernel=None was triggered! BSW spec fix may be incomplete."
  echo "$KERNEL_NONE" | head -20
fi
echo ""

# ----------------------------------------------------------------
# Q2: _slice_leaf type — verifies the jax.core.Tracer fix.
#
#   Expected output during jit/scan tracing:
#     type(w) = DynamicJaxprTracer or BatchTracer
#     is_abstract = False  (our fix: not ShapeDtypeStruct)
#     was_tracer  = True   (it IS a Tracer, but we no longer treat it as abstract)
#
#   If you see is_abstract=True with was_tracer=True, the Tracer check is back.
#   If you see type=ShapeDtypeStruct, it's a real eval_shape context (rare).
# ----------------------------------------------------------------
echo "--- Q2: _slice_leaf type diagnostic ---"
SLICE_DIAG=$(grep "\[DIAG slice_leaf\]" "$LOG_FILE" | head -20)
if [[ -z "$SLICE_DIAG" ]]; then
  echo "  No slice_leaf log found. gather_weights_across_stages_vmap was not called."
else
  echo "$SLICE_DIAG"
  echo ""

  # Check for bad cases: is_abstract=True with was_tracer=True (means Tracer bug is back)
  BAD=$(echo "$SLICE_DIAG" | grep "is_abstract=True" | grep "was_tracer=True")
  if [[ -n "$BAD" ]]; then
    echo "  ERROR: is_abstract=True for a Tracer. The jax.core.Tracer bug is present!"
    echo "$BAD"
  else
    echo "  OK: No abstract=True on Tracers detected."
  fi

  # Summarize types seen
  echo ""
  echo "  Leaf types seen:"
  echo "$SLICE_DIAG" | grep -oP "type\(w\)=\K[^,]+" | sort | uniq -c | sort -rn
fi
echo ""

# ----------------------------------------------------------------
# Overall summary
# ----------------------------------------------------------------
echo "--- Summary ---"
ERRORS=$(grep "\[DIAG kernel_none\]" "$LOG_FILE" | wc -l)
echo "  kernel=None hits : $ERRORS  (want 0)"

SLICE_COUNT=$(grep "\[DIAG slice_leaf\]" "$LOG_FILE" | wc -l)
echo "  slice_leaf logs  : $SLICE_COUNT"

BAD_ABSTRACT=$(grep "\[DIAG slice_leaf\]" "$LOG_FILE" | grep "is_abstract=True" | grep "was_tracer=True" | wc -l)
echo "  bad abstract hits: $BAD_ABSTRACT  (want 0)"

if [[ $ERRORS -eq 0 && $BAD_ABSTRACT -eq 0 ]]; then
  echo ""
  echo "  PASS: All diagnostics clean."
else
  echo ""
  echo "  FAIL: See warnings above."
fi
