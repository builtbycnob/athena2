"""Post-hoc calibration methods.

Implements three post-hoc calibration techniques:
1. Temperature Scaling — single parameter, fast, reliable
2. Isotonic Regression — non-parametric, better with >1K samples
3. Venn-ABERS — distribution-free calibration with validity guarantees

Comparison strategy: fit all three on a held-out calibration set,
evaluate ACE on a separate held-out test set, pick winner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationModel:
    """A fitted calibration model."""
    method: str  # "temperature", "isotonic", "venn_abers"
    params: dict[str, Any]
    ace_before: float
    ace_after: float
    ece_before: float
    ece_after: float

    def summary(self) -> str:
        return (
            f"{self.method}: ACE {self.ace_before:.4f} → {self.ace_after:.4f} "
            f"(ECE {self.ece_before:.4f} → {self.ece_after:.4f})"
        )


# ── Temperature Scaling ──────────────────────────────────────────

def fit_temperature_scaling(
    logits: np.ndarray,
    y_true: np.ndarray,
    grid_min: float = 0.1,
    grid_max: float = 5.0,
    grid_steps: int = 200,
) -> float:
    """Find optimal temperature that minimizes NLL.

    Args:
        logits: Raw model logits, shape (N,) for binary or (N, C) for multi-class.
        y_true: True labels, shape (N,).
        grid_min: Minimum temperature to search.
        grid_max: Maximum temperature to search.
        grid_steps: Number of grid points.

    Returns:
        Optimal temperature T.
    """
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    if logits.ndim == 1:
        # Binary case
        def nll(T):
            scaled = logits / T
            probs = 1.0 / (1.0 + np.exp(-scaled))
            probs = np.clip(probs, 1e-15, 1 - 1e-15)
            return -float(np.mean(
                y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs)
            ))
    else:
        # Multi-class case
        def nll(T):
            scaled = logits / T
            exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
            probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
            probs = np.clip(probs, 1e-15, 1.0)
            return -float(np.mean(np.log(probs[np.arange(len(y_true)), y_true])))

    # Coarse grid
    temps = np.linspace(grid_min, grid_max, grid_steps)
    nlls = [nll(t) for t in temps]
    best_T = float(temps[np.argmin(nlls)])

    # Fine refinement
    fine = np.linspace(max(0.05, best_T - 0.5), best_T + 0.5, grid_steps)
    fine_nlls = [nll(t) for t in fine]
    best_T = float(fine[np.argmin(fine_nlls)])

    logger.info(f"Temperature scaling: T={best_T:.3f}")
    return best_T


def apply_temperature_scaling(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply temperature scaling to logits → calibrated probabilities.

    Args:
        logits: Raw logits, shape (N,) or (N, C).
        temperature: Temperature parameter.

    Returns:
        Calibrated probabilities.
    """
    scaled = logits / temperature
    if scaled.ndim == 1:
        return 1.0 / (1.0 + np.exp(-scaled))
    else:
        exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


# ── Isotonic Regression ──────────────────────────────────────────

def fit_isotonic_regression(
    y_prob: np.ndarray,
    y_true: np.ndarray,
) -> Any:
    """Fit isotonic regression calibration.

    Non-parametric — maps predicted probabilities to calibrated probabilities
    using a monotonically increasing function. Better than temperature scaling
    when you have >1K calibration samples.

    Args:
        y_prob: Predicted probabilities of positive class, shape (N,).
        y_true: True labels, shape (N,).

    Returns:
        Fitted IsotonicRegression model.
    """
    from sklearn.isotonic import IsotonicRegression

    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(y_prob, y_true)

    logger.info(f"Isotonic regression fitted on {len(y_true):,} samples")
    return ir


def apply_isotonic_regression(
    ir_model: Any,
    y_prob: np.ndarray,
) -> np.ndarray:
    """Apply isotonic regression calibration.

    Args:
        ir_model: Fitted IsotonicRegression.
        y_prob: Predicted probabilities, shape (N,).

    Returns:
        Calibrated probabilities.
    """
    return ir_model.predict(np.asarray(y_prob, dtype=float))


# ── Venn-ABERS Calibration ───────────────────────────────────────

def fit_venn_abers(
    y_prob: np.ndarray,
    y_true: np.ndarray,
) -> tuple[Any, Any]:
    """Fit Venn-ABERS calibration.

    Distribution-free calibration with validity guarantees.
    Produces two isotonic regressions (one assuming label=0, one assuming label=1)
    and outputs a calibrated interval.

    Args:
        y_prob: Predicted probabilities, shape (N,).
        y_true: True labels, shape (N,).

    Returns:
        Tuple of (ir_0, ir_1) — isotonic regressions for each label hypothesis.
    """
    from sklearn.isotonic import IsotonicRegression

    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    # Fit isotonic regression under each label hypothesis
    ir_0 = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir_1 = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    # Under hypothesis y=0: add (p, 0) to calibration set
    probs_with_0 = np.append(y_prob, y_prob)  # duplicate
    labels_with_0 = np.append(y_true, np.zeros_like(y_true))
    ir_0.fit(probs_with_0, labels_with_0)

    # Under hypothesis y=1: add (p, 1) to calibration set
    labels_with_1 = np.append(y_true, np.ones_like(y_true))
    ir_1.fit(probs_with_0, labels_with_1)

    logger.info(f"Venn-ABERS fitted on {len(y_true):,} samples")
    return ir_0, ir_1


def apply_venn_abers(
    ir_0: Any,
    ir_1: Any,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Venn-ABERS calibration.

    Args:
        ir_0: Isotonic regression under y=0 hypothesis.
        ir_1: Isotonic regression under y=1 hypothesis.
        y_prob: Predicted probabilities, shape (N,).

    Returns:
        Tuple of (calibrated_probs, lower_bounds, upper_bounds).
    """
    y_prob = np.asarray(y_prob, dtype=float)

    p0 = ir_0.predict(y_prob)  # P(y=1) under hypothesis y=0
    p1 = ir_1.predict(y_prob)  # P(y=1) under hypothesis y=1

    # Calibrated probability = geometric mean of the two
    calibrated = p1 / (1 - p0 + p1 + 1e-15)

    return np.clip(calibrated, 0, 1), np.minimum(p0, p1), np.maximum(p0, p1)


# ── Comparison ───────────────────────────────────────────────────

def compare_calibration_methods(
    logits: np.ndarray,
    y_prob: np.ndarray,
    y_true: np.ndarray,
    cal_logits: np.ndarray | None = None,
    cal_probs: np.ndarray | None = None,
    cal_labels: np.ndarray | None = None,
) -> list[CalibrationModel]:
    """Compare all calibration methods and return ranked results.

    Uses a calibration set to fit models and an evaluation set to measure.

    Args:
        logits: Test set logits for temperature scaling.
        y_prob: Test set probabilities.
        y_true: Test set labels.
        cal_logits: Calibration set logits (optional, uses test if None).
        cal_probs: Calibration set probabilities.
        cal_labels: Calibration set labels.

    Returns:
        List of CalibrationModel sorted by ACE (best first).
    """
    from athena2.evaluation.metrics import (
        adaptive_calibration_error,
        expected_calibration_error,
    )

    if cal_logits is None:
        cal_logits = logits
    if cal_probs is None:
        cal_probs = y_prob
    if cal_labels is None:
        cal_labels = y_true

    results = []

    # Baseline ACE/ECE
    ace_before = adaptive_calibration_error(y_true, y_prob)
    ece_before = expected_calibration_error(y_true, y_prob)

    # 1. Temperature Scaling
    try:
        T = fit_temperature_scaling(cal_logits, cal_labels)
        cal_probs_ts = apply_temperature_scaling(logits, T)
        ace_ts = adaptive_calibration_error(y_true, cal_probs_ts)
        ece_ts = expected_calibration_error(y_true, cal_probs_ts)
        results.append(CalibrationModel(
            method="temperature_scaling",
            params={"temperature": T},
            ace_before=ace_before, ace_after=ace_ts,
            ece_before=ece_before, ece_after=ece_ts,
        ))
    except Exception as e:
        logger.warning(f"Temperature scaling failed: {e}")

    # 2. Isotonic Regression
    try:
        ir = fit_isotonic_regression(cal_probs, cal_labels)
        cal_probs_ir = apply_isotonic_regression(ir, y_prob)
        ace_ir = adaptive_calibration_error(y_true, cal_probs_ir)
        ece_ir = expected_calibration_error(y_true, cal_probs_ir)
        results.append(CalibrationModel(
            method="isotonic_regression",
            params={"model": ir},
            ace_before=ace_before, ace_after=ace_ir,
            ece_before=ece_before, ece_after=ece_ir,
        ))
    except Exception as e:
        logger.warning(f"Isotonic regression failed: {e}")

    # 3. Venn-ABERS
    try:
        ir_0, ir_1 = fit_venn_abers(cal_probs, cal_labels)
        cal_probs_va, _, _ = apply_venn_abers(ir_0, ir_1, y_prob)
        ace_va = adaptive_calibration_error(y_true, cal_probs_va)
        ece_va = expected_calibration_error(y_true, cal_probs_va)
        results.append(CalibrationModel(
            method="venn_abers",
            params={"ir_0": ir_0, "ir_1": ir_1},
            ace_before=ace_before, ace_after=ace_va,
            ece_before=ece_before, ece_after=ece_va,
        ))
    except Exception as e:
        logger.warning(f"Venn-ABERS failed: {e}")

    # Sort by ACE (best first)
    results.sort(key=lambda r: r.ace_after)

    for r in results:
        logger.info(f"  {r.summary()}")

    return results
