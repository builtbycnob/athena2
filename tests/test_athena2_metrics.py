"""Tests for ATHENA2 evaluation metrics."""

import math

import numpy as np
import pytest

from athena2.evaluation.metrics import (
    BrierDecomposition,
    CalibrationCurve,
    ConformalResult,
    EvaluationReport,
    accuracy,
    apply_temperature,
    brier_decomposition,
    brier_score,
    compute_calibration_curve,
    conformal_calibrate,
    conformal_predict,
    evaluate,
    evaluate_conformal,
    expected_calibration_error,
    find_optimal_temperature,
    log_loss,
    macro_f1,
    wilson_ci,
)


# ── Basic Metrics ──────────────────────────────────────────────────

class TestAccuracy:
    def test_perfect(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert accuracy(y_true, y_pred) == 1.0

    def test_half(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert accuracy(y_true, y_pred) == 0.5

    def test_zero(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        assert accuracy(y_true, y_pred) == 0.0


class TestMacroF1:
    def test_perfect(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert macro_f1(y_true, y_pred) == 1.0

    def test_all_same_prediction(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0])
        f1 = macro_f1(y_true, y_pred)
        assert 0 < f1 < 1  # One class has F1=0, other has some

    def test_imbalanced(self):
        y_true = np.array([0, 0, 0, 0, 1])
        y_pred = np.array([0, 0, 0, 0, 0])
        f1 = macro_f1(y_true, y_pred)
        # Class 0: P=4/5, R=4/4, F1=8/9. Class 1: P=0, R=0, F1=0. Macro: ~0.44
        assert 0.4 < f1 < 0.5


class TestWilsonCI:
    def test_all_correct(self):
        low, high = wilson_ci(10, 10)
        assert low > 0.7
        assert high == 1.0

    def test_half(self):
        low, high = wilson_ci(5, 10)
        assert low < 0.5
        assert high > 0.5

    def test_zero_n(self):
        assert wilson_ci(0, 0) == (0.0, 1.0)

    def test_small_sample(self):
        low, high = wilson_ci(1, 2)
        # Should be wide interval
        assert high - low > 0.3


# ── Calibration ────────────────────────────────────────────────────

class TestECE:
    def test_perfectly_calibrated(self):
        # If predictions match reality perfectly
        np.random.seed(42)
        n = 1000
        y_prob = np.random.uniform(0, 1, n)
        y_true = (np.random.uniform(0, 1, n) < y_prob).astype(int)
        ece = expected_calibration_error(y_true, y_prob)
        assert ece < 0.05  # Should be close to 0

    def test_overconfident(self):
        # Always predicts 0.9 but accuracy is 0.5
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.full(10, 0.9)
        ece = expected_calibration_error(y_true, y_prob)
        assert ece > 0.3  # Should be high

    def test_adaptive_strategy(self):
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)
        y_prob = np.random.uniform(0, 1, n)
        ece_uniform = expected_calibration_error(y_true, y_prob, strategy="uniform")
        ece_adaptive = expected_calibration_error(y_true, y_prob, strategy="adaptive")
        assert isinstance(ece_uniform, float)
        assert isinstance(ece_adaptive, float)


class TestBrierScore:
    def test_perfect(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0, 1, 0, 1])
        assert brier_score(y_true, y_prob) == 0.0

    def test_worst(self):
        y_true = np.array([0, 1])
        y_prob = np.array([1, 0])
        assert brier_score(y_true, y_prob) == 1.0


class TestBrierDecomposition:
    def test_components_sum(self):
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 2, n)
        y_prob = np.random.uniform(0, 1, n)
        decomp = brier_decomposition(y_true, y_prob)

        # Brier ≈ Reliability - Resolution + Uncertainty
        reconstructed = decomp.reliability - decomp.resolution + decomp.uncertainty
        assert abs(decomp.brier - reconstructed) < 0.02  # Allow small numerical error

    def test_uncertainty_is_base_rate(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        decomp = brier_decomposition(y_true, y_prob)
        base_rate = 2 / 5
        expected_uncertainty = base_rate * (1 - base_rate)
        assert abs(decomp.uncertainty - expected_uncertainty) < 0.01


class TestLogLoss:
    def test_perfect(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.001, 0.999])
        ll = log_loss(y_true, y_prob)
        assert ll < 0.01

    def test_worst(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.999, 0.001])
        ll = log_loss(y_true, y_prob)
        assert ll > 5


# ── Temperature Scaling ────────────────────────────────────────────

class TestTemperatureScaling:
    def test_optimal_temperature_returns_positive(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        logits = np.where(y_true == 1, 3.0, -3.0) + np.random.normal(0, 1, 200)

        T = find_optimal_temperature(y_true, logits)
        assert T > 0.05  # Temperature must be positive
        assert T < 5.0   # Within search range

    def test_apply_temperature(self):
        logits = np.array([0.0, 2.0, -2.0])
        probs_t1 = apply_temperature(logits, 1.0)
        probs_t2 = apply_temperature(logits, 2.0)

        # Higher temperature → more uniform probabilities
        assert probs_t2[1] < probs_t1[1]  # Less confident
        assert probs_t2[2] > probs_t1[2]  # Less confident


# ── Conformal Prediction ──────────────────────────────────────────

class TestConformalPrediction:
    def test_calibrate_threshold(self):
        np.random.seed(42)
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.4, 0.6, 0.15, 0.85])

        threshold = conformal_calibrate(y_true, y_prob, target_coverage=0.90)
        assert 0 < threshold < 1

    def test_prediction_sets(self):
        y_prob = np.array([0.1, 0.5, 0.9])
        threshold = 0.3

        sets = conformal_predict(y_prob, threshold)
        assert len(sets) == 3
        assert all(isinstance(s, set) for s in sets)

        # High confidence → singleton
        assert len(sets[0]) == 1  # P=0.1, confident class 0
        assert len(sets[2]) == 1  # P=0.9, confident class 1

        # Low confidence → pair
        # P=0.5: both scores are 0.5 > 0.3, so both excluded → empty → both included
        assert len(sets[1]) == 2

    def test_coverage_guarantee(self):
        np.random.seed(42)
        n_cal = 500
        n_test = 200

        # Generate calibrated probabilities
        y_true_cal = np.random.randint(0, 2, n_cal)
        y_prob_cal = np.clip(
            np.where(y_true_cal == 1, 0.7, 0.3) + np.random.normal(0, 0.2, n_cal),
            0.01, 0.99,
        )

        threshold = conformal_calibrate(y_true_cal, y_prob_cal, target_coverage=0.90)

        # Test on new data
        y_true_test = np.random.randint(0, 2, n_test)
        y_prob_test = np.clip(
            np.where(y_true_test == 1, 0.7, 0.3) + np.random.normal(0, 0.2, n_test),
            0.01, 0.99,
        )

        pred_sets = conformal_predict(y_prob_test, threshold)
        result = evaluate_conformal(y_true_test, pred_sets, 0.90, threshold)

        # Coverage should be ≥ 90% (with some slack for finite samples)
        assert result.coverage >= 0.85


# ── Calibration Curves ─────────────────────────────────────────────

class TestCalibrationCurve:
    def test_returns_correct_structure(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)

        curve = compute_calibration_curve(y_true, y_prob)
        assert isinstance(curve, CalibrationCurve)
        assert len(curve.bin_centers) == len(curve.bin_accuracies)
        assert len(curve.bin_centers) == len(curve.bin_counts)
        assert isinstance(curve.ece, float)


# ── Full Evaluation ────────────────────────────────────────────────

class TestEvaluate:
    def test_full_pipeline(self):
        np.random.seed(42)
        n = 200

        y_true = np.random.randint(0, 2, n)
        y_prob = np.clip(
            np.where(y_true == 1, 0.7, 0.3) + np.random.normal(0, 0.15, n),
            0.01, 0.99,
        )

        report = evaluate(y_true, y_prob)

        assert isinstance(report, EvaluationReport)
        assert 0 < report.accuracy < 1
        assert 0 < report.macro_f1 < 1
        assert report.brier.brier >= 0
        assert report.ece >= 0
        assert report.n_samples == n

    def test_with_conformal(self):
        np.random.seed(42)
        n = 200

        y_true = np.random.randint(0, 2, n)
        y_prob = np.clip(
            np.where(y_true == 1, 0.7, 0.3) + np.random.normal(0, 0.15, n),
            0.01, 0.99,
        )

        # Calibrate on first half, evaluate on second
        threshold = conformal_calibrate(y_true[:100], y_prob[:100])
        report = evaluate(
            y_true[100:],
            y_prob[100:],
            conformal_threshold=threshold,
        )

        assert report.conformal is not None
        assert report.conformal.coverage > 0.8

    def test_markdown_report(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 50)
        y_prob = np.random.uniform(0, 1, 50)

        report = evaluate(y_true, y_prob)
        md = report.to_markdown()

        assert "Evaluation Report" in md
        assert "Accuracy" in md
        assert "Brier Score" in md
