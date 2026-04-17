"""Numerical test for BSW staleness hypothesis in NNXCircularPipeline.

HYPOTHESIS: In the nested scan design, BSW is created once per outer iteration
(per-repeat), but delayed stages (forwarding_delay=1) cross repeat boundaries
WITHIN the inner scan. This means delayed stages use stale (wrong repeat's)
weights for some inner iterations.

PROOF BY EXAMPLE:
  Config: num_stages=2, forwarding_delay=1, num_microbatches=4, num_repeats=2

  Outer k=1, BSW created at loop_iteration=4:
    Stage 0: repeat_id = 4//4 = 1  → BSW[stage0] = repeat 1 weights
    Stage 1: repeat_id = max(4-1,0)//4 = 0 → BSW[stage1] = repeat 0 weights

  Inner scan iterations:
    j=0 (iter=4): Stage 1 repeat_id = max(4-1,0)//4 = 0 → BSW correct ✓
    j=1 (iter=5): Stage 1 repeat_id = max(5-1,0)//4 = 1 → BSW STALE ✗
    j=2 (iter=6): Stage 1 repeat_id = max(6-1,0)//4 = 1 → BSW STALE ✗
    j=3 (iter=7): Stage 1 repeat_id = max(7-1,0)//4 = 1 → BSW STALE ✗

This test validates the staleness by computing per-stage repeat_ids across
all iterations and comparing what BSW provides vs what's actually needed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import jax.numpy as jnp
import numpy as np


def get_microbatch_and_repeat_ids(loop_iteration, num_stages, forwarding_delay, num_microbatches):
    """Reproduces PipelineSharedMixin.get_microbatch_and_repeat_ids."""
    stage_ids = np.arange(num_stages)
    microbatches_processed = np.maximum(loop_iteration - forwarding_delay * stage_ids, 0)
    microbatch_ids = microbatches_processed % num_microbatches
    repeat_ids = microbatches_processed // num_microbatches
    return microbatch_ids, repeat_ids


def test_bsw_staleness_detection():
    """Test that BSW created once per repeat produces stale weights for delayed stages."""

    configs = [
        # (num_stages, forwarding_delay, num_microbatches, num_repeats)
        (2, 1, 4, 2),  # Standard: 2 stages, delay 1, 4 microbatches, 2 repeats
        (2, 1, 2, 2),  # Minimal: 2 stages, delay 1, 2 microbatches, 2 repeats
        (4, 1, 4, 2),  # 4 stages, delay 1
        (2, 1, 4, 3),  # 3 repeats
    ]

    for num_stages, forwarding_delay, num_microbatches, num_repeats in configs:
        total_iterations = num_microbatches * num_repeats + (num_stages - 1) * forwarding_delay
        bubble_iterations = (num_stages - 1) * forwarding_delay

        print(f"\n{'='*70}")
        print(f"Config: stages={num_stages}, delay={forwarding_delay}, "
              f"MB={num_microbatches}, repeats={num_repeats}")
        print(f"Total iterations: {total_iterations}, Bubble: {bubble_iterations}")
        print(f"{'='*70}")

        stale_count = 0
        total_checks = 0

        # Simulate outer scan (repeats) + inner scan (microbatches)
        for repeat_k in range(num_repeats):
            # BSW creation: at the START of outer iteration k
            bsw_creation_iter = repeat_k * num_microbatches
            _, bsw_repeat_ids = get_microbatch_and_repeat_ids(
                bsw_creation_iter, num_stages, forwarding_delay, num_microbatches
            )

            print(f"\n  Outer k={repeat_k}: BSW created at iter={bsw_creation_iter}")
            print(f"    BSW repeat_ids per stage: {bsw_repeat_ids}")

            # Inner scan: microbatches within this repeat
            for j in range(num_microbatches):
                actual_iter = repeat_k * num_microbatches + j
                _, actual_repeat_ids = get_microbatch_and_repeat_ids(
                    actual_iter, num_stages, forwarding_delay, num_microbatches
                )

                for stage in range(num_stages):
                    total_checks += 1
                    bsw_has = bsw_repeat_ids[stage]
                    needs = actual_repeat_ids[stage]
                    is_stale = bsw_has != needs

                    if is_stale:
                        stale_count += 1
                        print(f"    j={j} (iter={actual_iter}): Stage {stage} "
                              f"BSW has repeat {bsw_has}, NEEDS repeat {needs} ✗ STALE")

        # Bubble iterations (use last BSW)
        if bubble_iterations > 0:
            bubble_start = num_repeats * num_microbatches
            _, bubble_bsw_repeat_ids = get_microbatch_and_repeat_ids(
                bubble_start, num_stages, forwarding_delay, num_microbatches
            )
            print(f"\n  Bubble: BSW created at iter={bubble_start}")
            print(f"    Bubble BSW repeat_ids: {bubble_bsw_repeat_ids}")

            for b in range(bubble_iterations):
                actual_iter = bubble_start + b
                _, actual_repeat_ids = get_microbatch_and_repeat_ids(
                    actual_iter, num_stages, forwarding_delay, num_microbatches
                )
                for stage in range(num_stages):
                    total_checks += 1
                    bsw_has = bubble_bsw_repeat_ids[stage]
                    needs = actual_repeat_ids[stage]
                    is_stale = bsw_has != needs
                    if is_stale:
                        stale_count += 1
                        print(f"    bubble b={b} (iter={actual_iter}): Stage {stage} "
                              f"BSW has repeat {bsw_has}, NEEDS repeat {needs} ✗ STALE")

        if stale_count > 0:
            print(f"\n  *** STALENESS CONFIRMED: {stale_count}/{total_checks} checks stale "
                  f"({100*stale_count/total_checks:.1f}%)")
        else:
            print(f"\n  All {total_checks} checks clean — no staleness detected")

    return stale_count


def test_compare_flat_vs_nested_bsw():
    """Compare flat scan (BSW every iteration) vs nested scan (BSW per repeat).

    In the flat scan (NNXPipeline), BSW is created EVERY iteration — always correct.
    In the nested scan (NNXCircularPipeline), BSW is created ONCE per repeat — potentially stale.

    This test shows exactly which iterations diverge.
    """
    num_stages = 2
    forwarding_delay = 1
    num_microbatches = 4
    num_repeats = 2
    total_iterations = num_microbatches * num_repeats + (num_stages - 1) * forwarding_delay

    print(f"\n{'='*70}")
    print("COMPARISON: Flat scan (per-iteration BSW) vs Nested scan (per-repeat BSW)")
    print(f"Config: stages={num_stages}, MB={num_microbatches}, repeats={num_repeats}")
    print(f"{'='*70}")

    mismatches = 0

    for iteration in range(total_iterations):
        # FLAT: BSW created at THIS iteration (always correct)
        _, flat_repeat_ids = get_microbatch_and_repeat_ids(
            iteration, num_stages, forwarding_delay, num_microbatches
        )

        # NESTED: BSW created at START of repeat
        repeat_k = min(iteration // num_microbatches, num_repeats - 1)
        bsw_creation_iter = repeat_k * num_microbatches
        # For bubble iterations: BSW recreated at bubble start
        if iteration >= num_repeats * num_microbatches:
            bsw_creation_iter = num_repeats * num_microbatches

        _, nested_repeat_ids = get_microbatch_and_repeat_ids(
            bsw_creation_iter, num_stages, forwarding_delay, num_microbatches
        )

        match = np.array_equal(flat_repeat_ids, nested_repeat_ids)
        if not match:
            mismatches += 1
            print(f"  iter={iteration}: MISMATCH")
            for s in range(num_stages):
                if flat_repeat_ids[s] != nested_repeat_ids[s]:
                    print(f"    Stage {s}: flat=repeat {flat_repeat_ids[s]}, "
                          f"nested=repeat {nested_repeat_ids[s]} (STALE)")
        else:
            print(f"  iter={iteration}: OK (both use repeat_ids={flat_repeat_ids})")

    if mismatches > 0:
        print(f"\n  *** {mismatches}/{total_iterations} iterations have BSW mismatches")
        print(f"  *** The nested scan uses WRONG repeat weights for these iterations")
        print(f"  *** This affects model correctness when num_stages > 1 AND num_repeats > 1")
    else:
        print(f"\n  All {total_iterations} iterations match — no staleness")

    return mismatches


def test_segmented_bsw_eliminates_staleness():
    """Test that the segmented inner scan approach eliminates BSW staleness.

    The fix splits the inner scan at stage transition boundaries and recreates
    BSW at each boundary. This ensures all stages always use correct weights.
    """
    configs = [
        (2, 1, 4, 2),
        (2, 1, 2, 2),
        (4, 1, 4, 2),
        (2, 1, 4, 3),
    ]

    total_stale = 0

    for num_stages, forwarding_delay, num_microbatches, num_repeats in configs:
        bubble_iterations = (num_stages - 1) * forwarding_delay

        # Compute transition boundaries (same as pipeline.py fix)
        transition_points = sorted(set(
            forwarding_delay * s for s in range(1, num_stages)
        ))

        def compute_segments(num_iters):
            boundaries = [0] + [t for t in transition_points if t < num_iters] + [num_iters]
            return [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)
                    if boundaries[i+1] > boundaries[i]]

        print(f"\n{'='*70}")
        print(f"SEGMENTED: stages={num_stages}, delay={forwarding_delay}, "
              f"MB={num_microbatches}, repeats={num_repeats}")
        segments = compute_segments(num_microbatches)
        print(f"Segments per repeat: {segments} ({len(segments)} BSW creations/repeat)")

        stale_count = 0
        total_checks = 0

        for repeat_k in range(num_repeats):
            # Simulate segmented inner scan
            inner_iter = repeat_k * num_microbatches
            for seg_len in segments:
                # BSW created at start of each segment
                _, bsw_repeat_ids = get_microbatch_and_repeat_ids(
                    inner_iter, num_stages, forwarding_delay, num_microbatches
                )
                for j in range(seg_len):
                    actual_iter = inner_iter + j
                    _, actual_repeat_ids = get_microbatch_and_repeat_ids(
                        actual_iter, num_stages, forwarding_delay, num_microbatches
                    )
                    for stage in range(num_stages):
                        total_checks += 1
                        if bsw_repeat_ids[stage] != actual_repeat_ids[stage]:
                            stale_count += 1
                            print(f"  STALE: repeat={repeat_k}, iter={actual_iter}, "
                                  f"stage={stage}: BSW={bsw_repeat_ids[stage]}, "
                                  f"needs={actual_repeat_ids[stage]}")
                inner_iter += seg_len

        # Check bubble too
        if bubble_iterations > 0:
            bubble_start = num_repeats * num_microbatches
            bubble_segments = compute_segments(bubble_iterations)
            bubble_iter = bubble_start
            for seg_len in bubble_segments:
                _, bsw_rids = get_microbatch_and_repeat_ids(
                    bubble_iter, num_stages, forwarding_delay, num_microbatches
                )
                for j in range(seg_len):
                    actual_iter = bubble_iter + j
                    _, actual_rids = get_microbatch_and_repeat_ids(
                        actual_iter, num_stages, forwarding_delay, num_microbatches
                    )
                    for stage in range(num_stages):
                        total_checks += 1
                        if bsw_rids[stage] != actual_rids[stage]:
                            stale_count += 1
                            print(f"  STALE (bubble): iter={actual_iter}, stage={stage}")
                bubble_iter += seg_len

        total_stale += stale_count
        if stale_count == 0:
            print(f"  ALL {total_checks} checks PASS — zero staleness")
        else:
            print(f"  *** {stale_count}/{total_checks} STALE — fix incomplete!")

    return total_stale


if __name__ == "__main__":
    print("=" * 70)
    print("BSW Staleness Tests")
    print("=" * 70)

    print("\n--- Test 1: Detect staleness in ORIGINAL design ---")
    stale_count = test_bsw_staleness_detection()

    print("\n--- Test 2: Flat vs Nested BSW comparison ---")
    mismatches = test_compare_flat_vs_nested_bsw()

    print("\n--- Test 3: Verify SEGMENTED fix eliminates staleness ---")
    segmented_stale = test_segmented_bsw_eliminates_staleness()

    print("\n" + "=" * 70)
    if stale_count > 0:
        print(f"ORIGINAL design: {stale_count} stale checks (confirms the bug)")
    if mismatches > 0:
        print(f"FLAT vs NESTED: {mismatches} iteration mismatches")
    if segmented_stale == 0:
        print("SEGMENTED FIX: ALL checks pass — staleness eliminated!")
    else:
        print(f"SEGMENTED FIX: {segmented_stale} stale checks remain — FIX INCOMPLETE")
    print("=" * 70)
