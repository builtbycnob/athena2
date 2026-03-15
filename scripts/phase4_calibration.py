#!/usr/bin/env python3
"""ATHENA2 Phase 3: SOTA Calibration Stack.

Implements the full calibration pipeline:
1. BSCE-GRA verification (training-time calibration)
2. Post-hoc calibration comparison (temperature, isotonic, Venn-ABERS)
3. Class-conditional conformal prediction via TorchCP
4. SWAG uncertainty integration
5. Evaluation infrastructure (reliability diagrams, Brier decomposition, ACE)

Usage:
    uv run python scripts/phase4_calibration.py                        # Full pipeline
    uv run python scripts/phase4_calibration.py --step posthoc         # Post-hoc only
    uv run python scripts/phase4_calibration.py --step conformal       # Conformal only
    uv run python scripts/phase4_calibration.py --step eval            # Evaluation only

Requires: pip install athena[worldmodel]
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("athena2.phase4")


def step_bsce_verification(model_dir: Path, output_dir: Path) -> dict:
    """Phase 3.1: Verify BSCE-GRA training-time calibration."""
    from athena2.evaluation.metrics import (
        adaptive_calibration_error, expected_calibration_error,
        brier_score,
    )

    logger.info("=" * 60)
    logger.info("PHASE 3.1: BSCE-GRA Training-Time Calibration Verification")
    logger.info("=" * 60)

    # Load validation predictions (from training)
    val_probs = np.load(model_dir / "val_probs.npy")
    val_labels = np.load(model_dir / "val_labels.npy")

    ace = adaptive_calibration_error(val_labels, val_probs)
    ece = expected_calibration_error(val_labels, val_probs)
    brier = brier_score(val_labels, val_probs)

    logger.info(f"  ACE (no post-hoc): {ace:.4f}")
    logger.info(f"  ECE (no post-hoc): {ece:.4f}")
    logger.info(f"  Brier (no post-hoc): {brier:.4f}")
    logger.info(f"  BSCE-GRA target: ACE < 0.05 → {'PASS' if ace < 0.05 else 'FAIL'}")

    return {"ace_raw": ace, "ece_raw": ece, "brier_raw": brier}


def step_posthoc_calibration(model_dir: Path, output_dir: Path) -> dict:
    """Phase 3.2: Post-hoc calibration comparison."""
    from athena2.calibration.temperature import compare_calibration_methods

    logger.info("=" * 60)
    logger.info("PHASE 3.2: Post-Hoc Calibration Comparison")
    logger.info("=" * 60)

    # Load test predictions
    test_logits = np.load(model_dir / "test_logits.npy")
    test_probs = np.load(model_dir / "test_probs.npy")
    test_labels = np.load(model_dir / "test_labels.npy")

    # Split: first half for calibration, second half for evaluation
    n = len(test_labels)
    cal_idx = np.arange(0, n // 2)
    eval_idx = np.arange(n // 2, n)

    results = compare_calibration_methods(
        logits=test_logits[eval_idx],
        y_prob=test_probs[eval_idx],
        y_true=test_labels[eval_idx],
        cal_logits=test_logits[cal_idx],
        cal_probs=test_probs[cal_idx],
        cal_labels=test_labels[cal_idx],
    )

    logger.info(f"\nCalibration Comparison:")
    for r in results:
        logger.info(f"  {r.summary()}")

    winner = results[0] if results else None
    if winner:
        logger.info(f"\nWINNER: {winner.method} (ACE: {winner.ace_after:.4f})")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = [{"method": r.method, "ace": r.ace_after, "ece": r.ece_after} for r in results]
    (output_dir / "calibration_comparison.json").write_text(json.dumps(summary, indent=2))

    return {"winner": winner.method if winner else None, "results": summary}


def step_conformal_prediction(model_dir: Path, output_dir: Path) -> dict:
    """Phase 3.3: Class-Conditional Conformal Prediction."""
    from athena2.calibration.conformal import (
        class_conditional_calibrate,
        class_conditional_predict,
        evaluate_class_conditional,
        calibrate_with_torchcp,
        predict_with_torchcp,
    )

    logger.info("=" * 60)
    logger.info("PHASE 3.3: Class-Conditional Conformal Prediction")
    logger.info("=" * 60)

    test_probs = np.load(model_dir / "test_probs.npy")
    test_labels = np.load(model_dir / "test_labels.npy")

    # Convert to 2-class format
    probs_2d = np.column_stack([1 - test_probs, test_probs])

    # Split: calibration / evaluation
    n = len(test_labels)
    cal_idx = np.arange(0, n // 2)
    eval_idx = np.arange(n // 2, n)

    results = {}

    for target in [0.90, 0.95]:
        logger.info(f"\n  Target coverage: {target:.0%}")

        # Try TorchCP first, fall back to manual
        cal_result = calibrate_with_torchcp(
            test_labels[cal_idx], probs_2d[cal_idx], target,
        )
        pred_sets = predict_with_torchcp(cal_result, probs_2d[eval_idx], target)

        # Also run manual for comparison
        thresholds = class_conditional_calibrate(
            test_labels[cal_idx], probs_2d[cal_idx], target,
        )
        manual_sets = class_conditional_predict(probs_2d[eval_idx], thresholds)
        manual_result = evaluate_class_conditional(
            test_labels[eval_idx], probs_2d[eval_idx],
            manual_sets, thresholds, target,
        )

        logger.info(f"\n  Manual CC-CP Results:")
        logger.info(manual_result.summary())

        results[f"coverage_{int(target*100)}"] = {
            "method": cal_result["method"],
            "overall_coverage": manual_result.overall_coverage,
            "per_class_coverage": {str(k): v for k, v in manual_result.per_class_coverage.items()},
            "singleton_fraction": manual_result.singleton_fraction,
            "meets_guarantees": manual_result.meets_guarantees(),
        }

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "conformal_results.json").write_text(json.dumps(results, indent=2))

    return results


def step_evaluation(model_dir: Path, output_dir: Path) -> dict:
    """Phase 3.5: Full evaluation infrastructure."""
    from athena2.evaluation.metrics import (
        evaluate, compute_calibration_curve, brier_decomposition,
        adaptive_calibration_error,
    )

    logger.info("=" * 60)
    logger.info("PHASE 3.5: Full Evaluation Infrastructure")
    logger.info("=" * 60)

    test_probs = np.load(model_dir / "test_probs.npy")
    test_labels = np.load(model_dir / "test_labels.npy")

    # Full evaluation
    report = evaluate(test_labels, test_probs)

    # Reliability diagram data
    curve = compute_calibration_curve(test_labels, test_probs, n_bins=15)

    # Brier decomposition
    brier = brier_decomposition(test_labels, test_probs, n_bins=15)

    logger.info(f"\nFull Evaluation:")
    logger.info(report.to_markdown())

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "full_evaluation.md").write_text(report.to_markdown())

    # Save reliability diagram data for plotting
    diagram_data = {
        "bin_centers": curve.bin_centers,
        "bin_accuracies": curve.bin_accuracies,
        "bin_counts": curve.bin_counts,
        "bin_confidences": curve.bin_confidences,
        "ece": curve.ece,
        "brier": {
            "brier": brier.brier,
            "reliability": brier.reliability,
            "resolution": brier.resolution,
            "uncertainty": brier.uncertainty,
        },
    }
    (output_dir / "reliability_diagram.json").write_text(json.dumps(diagram_data, indent=2))

    return {
        "ace": report.ace,
        "ece": report.ece,
        "f1": report.macro_f1,
        "brier": report.brier.brier,
    }


def main():
    parser = argparse.ArgumentParser(description="ATHENA2 Phase 3: Calibration")
    parser.add_argument("--step", choices=["bsce", "posthoc", "conformal", "eval", "all"],
                        default="all")
    parser.add_argument("--model-dir", type=Path, default=Path("data/models/phase2/best_model"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/models/phase3"))
    args = parser.parse_args()

    t0 = time.time()
    results = {}

    if args.step in ("bsce", "all"):
        results["bsce"] = step_bsce_verification(args.model_dir, args.output_dir)

    if args.step in ("posthoc", "all"):
        results["posthoc"] = step_posthoc_calibration(args.model_dir, args.output_dir)

    if args.step in ("conformal", "all"):
        results["conformal"] = step_conformal_prediction(args.model_dir, args.output_dir)

    if args.step in ("eval", "all"):
        results["eval"] = step_evaluation(args.model_dir, args.output_dir)

    elapsed = time.time() - t0
    logger.info(f"\nPhase 3 complete in {elapsed/60:.1f} min")
    (args.output_dir / "phase3_results.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
