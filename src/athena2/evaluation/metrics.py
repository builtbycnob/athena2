"""Evaluation metrics for ATHENA2.

Implements: accuracy, macro F1, ACE (primary), ECE (secondary),
Brier score decomposition, calibration curves, reliability diagrams,
conformal prediction evaluation.

ACE (Adaptive Calibration Error) is the primary calibration metric because
ECE is binning-dependent (ICLR 2025). ACE uses adaptive binning.

All metrics follow the same interface: arrays of predictions + labels in, scalar out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── Basic Classification Metrics ───────────────────────────────────

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Simple accuracy: fraction of correct predictions."""
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None) -> float:
    """Macro-averaged F1 score across all classes."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    f1s = []
    for label in labels:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    return float(np.mean(f1s))


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    Robust for small samples (n < 30) and edge cases (0 or n successes).
    """
    if n == 0:
        return (0.0, 1.0)

    p_hat = successes / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom

    return (max(0.0, center - margin), min(1.0, center + margin))


# ── Calibration Metrics ────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Expected Calibration Error (ECE).

    Args:
        y_true: Binary labels (0 or 1).
        y_prob: Predicted probability of class 1.
        n_bins: Number of calibration bins.
        strategy: "uniform" (equal-width) or "adaptive" (equal-count).

    Returns:
        ECE value (lower is better, 0 = perfectly calibrated).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "adaptive":
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(y_prob, quantiles)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        n_in_bin = int(np.sum(mask))
        if n_in_bin == 0:
            continue

        avg_confidence = float(np.mean(y_prob[mask]))
        avg_accuracy = float(np.mean(y_true[mask]))
        ece += (n_in_bin / len(y_true)) * abs(avg_accuracy - avg_confidence)

    return ece


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score: mean squared error of probability estimates."""
    return float(np.mean((y_prob - y_true) ** 2))


@dataclass
class BrierDecomposition:
    """Brier score decomposition: Reliability - Resolution + Uncertainty.

    - Reliability (lower = better calibrated)
    - Resolution (higher = more discriminating)
    - Uncertainty (fixed, depends on base rate)
    """
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    n_bins: int


def brier_decomposition(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> BrierDecomposition:
    """Decompose Brier score into reliability + resolution + uncertainty.

    Uses the Murphy (1973) decomposition:
        Brier = Reliability - Resolution + Uncertainty
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    # Base rate
    bar_o = float(np.mean(y_true))
    uncertainty = bar_o * (1 - bar_o)

    # Bin predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)

    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        else:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        n_k = int(np.sum(mask))
        if n_k == 0:
            continue

        bar_o_k = float(np.mean(y_true[mask]))  # Observed frequency in bin
        bar_f_k = float(np.mean(y_prob[mask]))   # Mean forecast in bin

        reliability += (n_k / n) * (bar_f_k - bar_o_k) ** 2
        resolution += (n_k / n) * (bar_o_k - bar_o) ** 2

    bs = brier_score(y_true, y_prob)

    return BrierDecomposition(
        brier=bs,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        n_bins=n_bins,
    )


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """Binary cross-entropy loss."""
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -float(np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def adaptive_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Adaptive Calibration Error (ACE) — PRIMARY calibration metric.

    Uses adaptive (equal-count) binning instead of uniform binning,
    making it robust to prediction distribution shape. Recommended
    over ECE (ICLR 2025: ECE is binning-dependent and unreliable).

    Args:
        y_true: Binary labels (0 or 1).
        y_prob: Predicted probability of class 1.
        n_bins: Number of adaptive bins.

    Returns:
        ACE value (lower is better, 0 = perfectly calibrated).
    """
    return expected_calibration_error(y_true, y_prob, n_bins=n_bins, strategy="adaptive")


def per_class_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[int, float]:
    """Per-class calibration error.

    Critical for imbalanced datasets (70/30 dismissal/approval).

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities for positive class.
        n_bins: Number of bins.

    Returns:
        Dict mapping class label → ECE for that class.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    result = {}

    for cls in np.unique(y_true):
        mask = y_true == cls
        if mask.sum() < n_bins:
            continue
        cls_prob = y_prob[mask] if cls == 1 else 1.0 - y_prob[mask]
        cls_true = np.ones(mask.sum()) if cls == 1 else np.ones(mask.sum())
        # For class 0: calibration of 1-p against actual class 0 rate
        result[int(cls)] = expected_calibration_error(
            (y_true[mask] == cls).astype(float),
            cls_prob if cls == 1 else 1.0 - y_prob[mask],
            n_bins=n_bins,
            strategy="adaptive",
        )

    return result


# ── Calibration Curves ─────────────────────────────────────────────

@dataclass
class CalibrationCurve:
    """Data for plotting a reliability diagram."""
    bin_centers: list[float]
    bin_accuracies: list[float]
    bin_counts: list[int]
    bin_confidences: list[float]
    ece: float


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> CalibrationCurve:
    """Compute calibration curve for reliability diagram."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    bin_confidences = []

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        else:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        n_in_bin = int(np.sum(mask))
        if n_in_bin == 0:
            continue

        bin_centers.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))
        bin_accuracies.append(float(np.mean(y_true[mask])))
        bin_counts.append(n_in_bin)
        bin_confidences.append(float(np.mean(y_prob[mask])))

    ece = expected_calibration_error(y_true, y_prob, n_bins)

    return CalibrationCurve(
        bin_centers=bin_centers,
        bin_accuracies=bin_accuracies,
        bin_counts=bin_counts,
        bin_confidences=bin_confidences,
        ece=ece,
    )


# ── Temperature Scaling ────────────────────────────────────────────

def find_optimal_temperature(
    y_true: np.ndarray,
    logits: np.ndarray,
    lr: float = 0.01,
    max_iter: int = 1000,
) -> float:
    """Find optimal temperature T that minimizes NLL on calibration set.

    Uses simple grid search + refinement (no PyTorch dependency needed).

    Args:
        y_true: Binary labels.
        logits: Raw model logits (before sigmoid).
        lr: Not used (grid search).
        max_iter: Not used.

    Returns:
        Optimal temperature T.
    """
    y_true = np.asarray(y_true, dtype=float)
    logits = np.asarray(logits, dtype=float)

    def nll_at_temp(T: float) -> float:
        scaled = logits / T
        probs = 1.0 / (1.0 + np.exp(-scaled))
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        return -float(np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs)))

    # Coarse grid search
    temps = np.linspace(0.1, 5.0, 100)
    nlls = [nll_at_temp(t) for t in temps]
    best_idx = int(np.argmin(nlls))
    best_T = float(temps[best_idx])

    # Fine grid refinement
    fine_temps = np.linspace(max(0.05, best_T - 0.5), best_T + 0.5, 100)
    fine_nlls = [nll_at_temp(t) for t in fine_temps]
    best_idx = int(np.argmin(fine_nlls))

    return float(fine_temps[best_idx])


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits, return calibrated probabilities."""
    scaled = logits / temperature
    return 1.0 / (1.0 + np.exp(-scaled))


# ── Conformal Prediction ──────────────────────────────────────────

@dataclass
class ConformalResult:
    """Result of conformal prediction evaluation."""
    coverage: float         # Fraction of true labels in prediction sets
    avg_set_size: float     # Average prediction set size (1 = crisp, 2 = uncertain)
    target_coverage: float  # Requested coverage level
    threshold: float        # Calibrated threshold
    n_singleton: int        # Predictions with set size 1 (confident)
    n_pair: int             # Predictions with set size 2 (uncertain)


def conformal_calibrate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_coverage: float = 0.90,
) -> float:
    """Compute conformal prediction threshold on calibration set.

    Args:
        y_true: Binary labels on calibration set.
        y_prob: Predicted P(class=1) on calibration set.
        target_coverage: Desired coverage (e.g., 0.90 for 90%).

    Returns:
        Threshold q such that prediction sets achieve target coverage.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # Nonconformity score: 1 - P(true label)
    scores = np.where(y_true == 1, 1 - y_prob, y_prob)

    # Quantile for desired coverage (with finite-sample correction)
    n = len(scores)
    adjusted_quantile = min(1.0, (math.ceil((n + 1) * target_coverage)) / n)

    return float(np.quantile(scores, adjusted_quantile))


def conformal_predict(
    y_prob: np.ndarray,
    threshold: float,
) -> list[set[int]]:
    """Generate prediction sets using conformal threshold.

    Args:
        y_prob: Predicted P(class=1) for test cases.
        threshold: Calibrated threshold from conformal_calibrate().

    Returns:
        List of prediction sets (each is a set of class labels).
    """
    y_prob = np.asarray(y_prob, dtype=float)
    prediction_sets = []

    for p in y_prob:
        pset = set()
        if (1 - p) <= threshold:  # Include class 1
            pset.add(1)
        if p <= threshold:         # Include class 0
            pset.add(0)
        if not pset:               # Edge case: include both
            pset = {0, 1}
        prediction_sets.append(pset)

    return prediction_sets


def evaluate_conformal(
    y_true: np.ndarray,
    prediction_sets: list[set[int]],
    target_coverage: float,
    threshold: float,
) -> ConformalResult:
    """Evaluate conformal prediction quality."""
    y_true = np.asarray(y_true, dtype=int)

    covered = sum(1 for yt, ps in zip(y_true, prediction_sets) if yt in ps)
    coverage = covered / len(y_true)

    set_sizes = [len(ps) for ps in prediction_sets]
    avg_size = float(np.mean(set_sizes))
    n_singleton = sum(1 for s in set_sizes if s == 1)
    n_pair = sum(1 for s in set_sizes if s == 2)

    return ConformalResult(
        coverage=coverage,
        avg_set_size=avg_size,
        target_coverage=target_coverage,
        threshold=threshold,
        n_singleton=n_singleton,
        n_pair=n_pair,
    )


# ── Comprehensive Report ──────────────────────────────────────────

@dataclass
class EvaluationReport:
    """Complete evaluation report for a model."""
    accuracy: float
    accuracy_ci: tuple[float, float]
    macro_f1: float
    brier: BrierDecomposition
    ace: float  # Primary calibration metric (adaptive)
    ece: float  # Secondary (uniform binning, for reference)
    ece_adaptive: float  # Same as ACE (backward compat alias)
    log_loss_val: float
    calibration_curve: CalibrationCurve
    conformal: ConformalResult | None
    n_samples: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Generate Markdown evaluation report."""
        lines = [
            "# Evaluation Report\n",
            f"**Samples**: {self.n_samples:,}",
            "",
            "## Classification Metrics",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Accuracy | {self.accuracy:.4f} [{self.accuracy_ci[0]:.3f}, {self.accuracy_ci[1]:.3f}] |",
            f"| Macro F1 | {self.macro_f1:.4f} |",
            "",
            "## Calibration Metrics",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Brier Score | {self.brier.brier:.4f} |",
            f"| → Reliability | {self.brier.reliability:.4f} |",
            f"| → Resolution | {self.brier.resolution:.4f} |",
            f"| → Uncertainty | {self.brier.uncertainty:.4f} |",
            f"| **ACE** (primary) | **{self.ace:.4f}** |",
            f"| ECE (uniform) | {self.ece:.4f} |",
            f"| Log Loss | {self.log_loss_val:.4f} |",
        ]

        if self.conformal:
            lines.extend([
                "",
                "## Conformal Prediction",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Target Coverage | {self.conformal.target_coverage:.0%} |",
                f"| Actual Coverage | {self.conformal.coverage:.1%} |",
                f"| Avg Set Size | {self.conformal.avg_set_size:.2f} |",
                f"| Singleton (confident) | {self.conformal.n_singleton} ({self.conformal.n_singleton/self.n_samples:.0%}) |",
                f"| Pair (uncertain) | {self.conformal.n_pair} ({self.conformal.n_pair/self.n_samples:.0%}) |",
            ])

        return "\n".join(lines)


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray | None = None,
    conformal_threshold: float | None = None,
    target_coverage: float = 0.90,
    metadata: dict[str, Any] | None = None,
) -> EvaluationReport:
    """Run full evaluation suite.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probability of class 1.
        y_pred: Predicted labels (optional, derived from y_prob > 0.5).
        conformal_threshold: Pre-calibrated conformal threshold (optional).
        target_coverage: For conformal prediction.
        metadata: Extra info to include in report.

    Returns:
        Complete EvaluationReport.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_pred is None:
        y_pred = (y_prob > 0.5).astype(int)

    n = len(y_true)
    acc = accuracy(y_true, y_pred)
    acc_ci = wilson_ci(int(np.sum(y_true == y_pred)), n)
    f1 = macro_f1(y_true, y_pred, labels=[0, 1])
    brier = brier_decomposition(y_true, y_prob)
    ece_uniform = expected_calibration_error(y_true, y_prob, strategy="uniform")
    ace = adaptive_calibration_error(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    cal_curve = compute_calibration_curve(y_true, y_prob)

    conformal_result = None
    if conformal_threshold is not None:
        pred_sets = conformal_predict(y_prob, conformal_threshold)
        conformal_result = evaluate_conformal(y_true, pred_sets, target_coverage, conformal_threshold)

    return EvaluationReport(
        accuracy=acc,
        accuracy_ci=acc_ci,
        macro_f1=f1,
        brier=brier,
        ace=ace,
        ece=ece_uniform,
        ece_adaptive=ace,  # backward compat alias
        log_loss_val=ll,
        calibration_curve=cal_curve,
        conformal=conformal_result,
        n_samples=n,
        metadata=metadata or {},
    )
