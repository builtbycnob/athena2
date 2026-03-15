"""Class-Conditional Conformal Prediction via TorchCP.

Stratified conformal prediction with per-class coverage guarantees.
Critical for the 70/30 dismissal/approval imbalance — naive CP gives
false sense of security for the minority class.

Reference: NeurIPS 2025 class-conditional conformal prediction.
Library: TorchCP (production-grade PyTorch conformal prediction).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClassConditionalResult:
    """Result of class-conditional conformal prediction."""
    overall_coverage: float
    per_class_coverage: dict[int, float]
    per_class_target: dict[int, float]
    overall_avg_set_size: float
    per_class_avg_set_size: dict[int, float]
    n_singleton: int
    n_pair: int
    n_empty: int
    n_total: int
    per_class_thresholds: dict[int, float]
    singleton_fraction: float

    def meets_guarantees(self) -> bool:
        """Check if all per-class coverage targets are met."""
        for cls, target in self.per_class_target.items():
            if self.per_class_coverage.get(cls, 0) < target - 0.02:  # 2% tolerance
                return False
        return True

    def summary(self) -> str:
        lines = [
            f"Class-Conditional Conformal Prediction ({self.n_total:,} samples)",
            f"  Overall coverage: {self.overall_coverage:.1%}",
            f"  Average set size: {self.overall_avg_set_size:.2f}",
            f"  Singletons: {self.n_singleton:,} ({self.singleton_fraction:.1%})",
            f"  Pairs: {self.n_pair:,} ({self.n_pair/max(self.n_total,1):.1%})",
        ]
        for cls in sorted(self.per_class_coverage.keys()):
            target = self.per_class_target.get(cls, 0.9)
            actual = self.per_class_coverage[cls]
            met = "OK" if actual >= target - 0.02 else "FAIL"
            lines.append(
                f"  Class {cls}: coverage={actual:.1%} "
                f"(target={target:.0%}) [{met}], "
                f"avg_set={self.per_class_avg_set_size.get(cls, 0):.2f}"
            )
        return "\n".join(lines)


def class_conditional_calibrate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_coverage: float = 0.90,
    n_classes: int = 2,
) -> dict[int, float]:
    """Calibrate conformal thresholds per class.

    For each class, computes the threshold on the nonconformity scores
    of that class's calibration samples to achieve the target coverage.

    Args:
        y_true: True labels on calibration set, shape (N,).
        y_prob: Predicted probabilities, shape (N, C).
        target_coverage: Desired coverage per class (e.g., 0.90).
        n_classes: Number of classes.

    Returns:
        Dict mapping class index → threshold.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_prob.ndim == 1:
        # Binary: convert P(class=1) to [P(class=0), P(class=1)]
        y_prob = np.column_stack([1 - y_prob, y_prob])

    thresholds = {}
    for cls in range(n_classes):
        cls_mask = y_true == cls
        n_cls = int(cls_mask.sum())

        if n_cls == 0:
            thresholds[cls] = 1.0
            continue

        # Nonconformity score for this class: 1 - P(true class)
        scores = 1.0 - y_prob[cls_mask, cls]

        # Quantile with finite-sample correction
        adjusted_quantile = min(1.0, math.ceil((n_cls + 1) * target_coverage) / n_cls)
        thresholds[cls] = float(np.quantile(scores, adjusted_quantile))

    return thresholds


def class_conditional_predict(
    y_prob: np.ndarray,
    thresholds: dict[int, float],
    n_classes: int = 2,
) -> list[set[int]]:
    """Generate prediction sets using per-class thresholds.

    A class c is included in the prediction set if 1 - P(c) <= threshold_c.
    This gives class-conditional coverage guarantees.

    Args:
        y_prob: Predicted probabilities, shape (N, C).
        thresholds: Per-class thresholds from calibration.
        n_classes: Number of classes.

    Returns:
        List of prediction sets.
    """
    y_prob = np.asarray(y_prob, dtype=float)

    if y_prob.ndim == 1:
        y_prob = np.column_stack([1 - y_prob, y_prob])

    prediction_sets = []
    for i in range(len(y_prob)):
        pset = set()
        for cls in range(n_classes):
            nonconformity = 1.0 - y_prob[i, cls]
            if nonconformity <= thresholds.get(cls, 1.0):
                pset.add(cls)
        if not pset:
            # Include most likely class as fallback
            pset.add(int(np.argmax(y_prob[i])))
        prediction_sets.append(pset)

    return prediction_sets


def evaluate_class_conditional(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    prediction_sets: list[set[int]],
    thresholds: dict[int, float],
    target_coverage: float = 0.90,
    n_classes: int = 2,
) -> ClassConditionalResult:
    """Evaluate class-conditional conformal prediction.

    Args:
        y_true: True labels, shape (N,).
        y_prob: Predicted probabilities, shape (N, C).
        prediction_sets: Generated prediction sets.
        thresholds: Per-class thresholds.
        target_coverage: Target coverage level.
        n_classes: Number of classes.

    Returns:
        ClassConditionalResult with per-class metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    n_total = len(y_true)

    # Overall coverage
    covered = sum(1 for yt, ps in zip(y_true, prediction_sets) if yt in ps)
    overall_coverage = covered / max(n_total, 1)

    # Set sizes
    set_sizes = [len(ps) for ps in prediction_sets]
    n_singleton = sum(1 for s in set_sizes if s == 1)
    n_pair = sum(1 for s in set_sizes if s == 2)
    n_empty = sum(1 for s in set_sizes if s == 0)

    # Per-class metrics
    per_class_coverage = {}
    per_class_avg_set_size = {}
    per_class_target = {}

    for cls in range(n_classes):
        cls_mask = y_true == cls
        n_cls = int(cls_mask.sum())
        per_class_target[cls] = target_coverage

        if n_cls == 0:
            per_class_coverage[cls] = 1.0
            per_class_avg_set_size[cls] = 0.0
            continue

        cls_covered = sum(
            1 for i, (yt, ps) in enumerate(zip(y_true, prediction_sets))
            if cls_mask[i] and yt in ps
        )
        per_class_coverage[cls] = cls_covered / n_cls

        cls_set_sizes = [len(prediction_sets[i]) for i in range(n_total) if cls_mask[i]]
        per_class_avg_set_size[cls] = float(np.mean(cls_set_sizes)) if cls_set_sizes else 0.0

    return ClassConditionalResult(
        overall_coverage=overall_coverage,
        per_class_coverage=per_class_coverage,
        per_class_target=per_class_target,
        overall_avg_set_size=float(np.mean(set_sizes)) if set_sizes else 0.0,
        per_class_avg_set_size=per_class_avg_set_size,
        n_singleton=n_singleton,
        n_pair=n_pair,
        n_empty=n_empty,
        n_total=n_total,
        per_class_thresholds=thresholds,
        singleton_fraction=n_singleton / max(n_total, 1),
    )


# ── TorchCP Integration ──────────────────────────────────────────

def calibrate_with_torchcp(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_coverage: float = 0.90,
) -> dict[str, Any]:
    """Calibrate using TorchCP library if available.

    Falls back to manual implementation if TorchCP is not installed.

    Args:
        y_true: True labels on calibration set.
        y_prob: Predicted probabilities (N, C).
        target_coverage: Desired coverage.

    Returns:
        Dict with calibration results and method used.
    """
    try:
        import torch
        from torchcp.classification.predictors import ClassWisePredictor
        from torchcp.classification.scores import APS

        logger.info("Using TorchCP for class-conditional conformal prediction")

        logits = torch.tensor(np.log(y_prob + 1e-10), dtype=torch.float32)
        labels = torch.tensor(y_true, dtype=torch.long)

        score_fn = APS()
        predictor = ClassWisePredictor(score_fn)
        predictor.calibrate(logits, labels, alpha=1 - target_coverage)

        return {
            "method": "torchcp_classwise",
            "predictor": predictor,
            "score_fn": score_fn,
        }
    except ImportError:
        logger.info("TorchCP not installed, using manual class-conditional CP")
        thresholds = class_conditional_calibrate(y_true, y_prob, target_coverage)
        return {
            "method": "manual_classwise",
            "thresholds": thresholds,
        }


def predict_with_torchcp(
    calibration_result: dict[str, Any],
    y_prob: np.ndarray,
    target_coverage: float = 0.90,
) -> list[set[int]]:
    """Generate prediction sets using calibrated predictor.

    Args:
        calibration_result: Output from calibrate_with_torchcp().
        y_prob: Test probabilities.
        target_coverage: Target coverage.

    Returns:
        List of prediction sets.
    """
    if calibration_result["method"] == "torchcp_classwise":
        import torch

        logits = torch.tensor(np.log(y_prob + 1e-10), dtype=torch.float32)
        predictor = calibration_result["predictor"]
        sets_tensor = predictor.predict(logits)

        prediction_sets = []
        for row in sets_tensor:
            pset = set(int(c) for c in torch.where(row)[0].tolist())
            if not pset:
                pset = {int(np.argmax(y_prob[len(prediction_sets)]))}
            prediction_sets.append(pset)
        return prediction_sets
    else:
        thresholds = calibration_result["thresholds"]
        return class_conditional_predict(y_prob, thresholds)
