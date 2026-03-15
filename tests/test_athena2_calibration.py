"""Tests for ATHENA2 calibration modules.

Tests: class-conditional conformal prediction, temperature scaling,
isotonic regression, Venn-ABERS, noise detection.
"""

import numpy as np
import pytest


# ── Class-Conditional Conformal Prediction ───────────────────────

class TestClassConditionalCP:
    def test_calibrate_thresholds(self):
        from athena2.calibration.conformal import class_conditional_calibrate

        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 2, n)
        y_prob = np.column_stack([
            np.clip(np.where(y_true == 0, 0.7, 0.3) + np.random.normal(0, 0.1, n), 0.01, 0.99),
            np.clip(np.where(y_true == 1, 0.7, 0.3) + np.random.normal(0, 0.1, n), 0.01, 0.99),
        ])
        # Normalize
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

        thresholds = class_conditional_calibrate(y_true, y_prob, target_coverage=0.90)
        assert 0 in thresholds
        assert 1 in thresholds
        assert all(0 < t <= 1 for t in thresholds.values())

    def test_prediction_sets(self):
        from athena2.calibration.conformal import class_conditional_predict

        y_prob = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
        thresholds = {0: 0.3, 1: 0.3}
        sets = class_conditional_predict(y_prob, thresholds)
        assert len(sets) == 3
        assert all(isinstance(s, set) for s in sets)
        # High confidence → singleton
        assert len(sets[0]) == 1  # confident class 0
        assert len(sets[2]) == 1  # confident class 1

    def test_coverage_guarantee(self):
        from athena2.calibration.conformal import (
            class_conditional_calibrate,
            class_conditional_predict,
            evaluate_class_conditional,
        )

        np.random.seed(42)
        n_cal, n_test = 500, 200
        target = 0.90

        # Calibration set
        y_cal = np.random.randint(0, 2, n_cal)
        probs_cal = np.column_stack([
            np.clip(np.where(y_cal == 0, 0.7, 0.3) + np.random.normal(0, 0.15, n_cal), 0.01, 0.99),
            np.clip(np.where(y_cal == 1, 0.7, 0.3) + np.random.normal(0, 0.15, n_cal), 0.01, 0.99),
        ])
        probs_cal = probs_cal / probs_cal.sum(axis=1, keepdims=True)

        thresholds = class_conditional_calibrate(y_cal, probs_cal, target)

        # Test set
        y_test = np.random.randint(0, 2, n_test)
        probs_test = np.column_stack([
            np.clip(np.where(y_test == 0, 0.7, 0.3) + np.random.normal(0, 0.15, n_test), 0.01, 0.99),
            np.clip(np.where(y_test == 1, 0.7, 0.3) + np.random.normal(0, 0.15, n_test), 0.01, 0.99),
        ])
        probs_test = probs_test / probs_test.sum(axis=1, keepdims=True)

        pred_sets = class_conditional_predict(probs_test, thresholds)
        result = evaluate_class_conditional(y_test, probs_test, pred_sets, thresholds, target)

        assert result.overall_coverage >= 0.85  # Allow slack for finite samples
        assert result.n_total == n_test
        assert result.singleton_fraction > 0  # Should have some confident predictions

    def test_result_summary(self):
        from athena2.calibration.conformal import ClassConditionalResult

        result = ClassConditionalResult(
            overall_coverage=0.92, per_class_coverage={0: 0.91, 1: 0.93},
            per_class_target={0: 0.90, 1: 0.90},
            overall_avg_set_size=1.15, per_class_avg_set_size={0: 1.1, 1: 1.2},
            n_singleton=170, n_pair=30, n_empty=0, n_total=200,
            per_class_thresholds={0: 0.3, 1: 0.35},
            singleton_fraction=0.85,
        )
        assert result.meets_guarantees()
        s = result.summary()
        assert "Class-Conditional" in s

    def test_binary_input(self):
        """Test with 1D probability input (auto-converts to 2D)."""
        from athena2.calibration.conformal import class_conditional_calibrate

        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.clip(
            np.where(y_true == 1, 0.7, 0.3) + np.random.normal(0, 0.1, 100),
            0.01, 0.99,
        )
        # 1D input
        thresholds = class_conditional_calibrate(y_true, y_prob)
        assert 0 in thresholds
        assert 1 in thresholds


# ── Temperature Scaling ──────────────────────────────────────────

class TestTemperatureScaling:
    def test_fit_binary(self):
        from athena2.calibration.temperature import fit_temperature_scaling

        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        logits = np.where(y_true == 1, 2.0, -2.0) + np.random.normal(0, 1, 200)
        T = fit_temperature_scaling(logits, y_true)
        assert 0.05 < T < 5.0

    def test_apply_binary(self):
        from athena2.calibration.temperature import apply_temperature_scaling

        logits = np.array([0.0, 2.0, -2.0])
        probs = apply_temperature_scaling(logits, 1.0)
        assert len(probs) == 3
        assert 0.49 < probs[0] < 0.51  # logit 0 → ~0.5

    def test_higher_temp_softens(self):
        from athena2.calibration.temperature import apply_temperature_scaling

        logits = np.array([3.0, -3.0])
        p1 = apply_temperature_scaling(logits, 1.0)
        p2 = apply_temperature_scaling(logits, 3.0)
        # Higher T → more uniform
        assert p2[0] < p1[0]
        assert p2[1] > p1[1]


# ── Isotonic Regression ──────────────────────────────────────────

class TestIsotonicRegression:
    def test_fit_and_apply(self):
        from athena2.calibration.temperature import (
            fit_isotonic_regression,
            apply_isotonic_regression,
        )

        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_prob = np.clip(
            np.where(y_true == 1, 0.7, 0.3) + np.random.normal(0, 0.15, 200),
            0.01, 0.99,
        )

        ir = fit_isotonic_regression(y_prob, y_true)
        calibrated = apply_isotonic_regression(ir, y_prob)
        assert len(calibrated) == 200
        assert all(0 <= p <= 1 for p in calibrated)


# ── Venn-ABERS ───────────────────────────────────────────────────

class TestVennABERS:
    def test_fit_and_apply(self):
        from athena2.calibration.temperature import (
            fit_venn_abers,
            apply_venn_abers,
        )

        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_prob = np.clip(
            np.where(y_true == 1, 0.7, 0.3) + np.random.normal(0, 0.15, 200),
            0.01, 0.99,
        )

        ir_0, ir_1 = fit_venn_abers(y_prob, y_true)
        calibrated, lower, upper = apply_venn_abers(ir_0, ir_1, y_prob)

        assert len(calibrated) == 200
        assert all(0 <= p <= 1 for p in calibrated)


# ── Noise Detection ──────────────────────────────────────────────

class TestNoiseDetection:
    def test_fallback_detection(self):
        from athena2.data.noise_detection import detect_label_noise

        np.random.seed(42)
        n = 100
        labels = np.random.randint(0, 2, n)
        # Intentionally noisy: some predictions disagree with labels
        pred_probs = np.zeros((n, 2))
        for i in range(n):
            if np.random.random() < 0.1:  # 10% noise
                pred_probs[i, 1 - labels[i]] = 0.9
                pred_probs[i, labels[i]] = 0.1
            else:
                pred_probs[i, labels[i]] = 0.9
                pred_probs[i, 1 - labels[i]] = 0.1

        report = detect_label_noise(labels, pred_probs, noise_weight=0.5)
        assert report.total_samples == n
        assert report.noise_rate >= 0
        assert len(report.sample_weights) == n
        assert all(0 < w <= 1 for w in report.sample_weights)

    def test_with_languages(self):
        from athena2.data.noise_detection import detect_label_noise

        np.random.seed(42)
        n = 100
        labels = np.random.randint(0, 2, n)
        pred_probs = np.column_stack([
            np.where(labels == 0, 0.8, 0.2),
            np.where(labels == 1, 0.8, 0.2),
        ])
        languages = np.array(["de"] * 60 + ["fr"] * 30 + ["it"] * 10)

        report = detect_label_noise(labels, pred_probs, languages=languages)
        assert "de" in report.noise_by_language
        assert "fr" in report.noise_by_language
        assert "it" in report.noise_by_language

    def test_noise_report_summary(self):
        from athena2.data.noise_detection import NoiseReport

        report = NoiseReport(
            total_samples=1000, n_noisy=88, noise_rate=0.088,
            noise_by_language={"de": 0.05, "fr": 0.08, "it": 0.15},
            noise_by_label={"0": 0.06, "1": 0.12},
            noisy_indices=np.arange(88),
            sample_weights=np.ones(1000),
        )
        s = report.summary()
        assert "88" in s
        assert "8.8%" in s


# ── Updated Metrics ──────────────────────────────────────────────

class TestACE:
    def test_ace_is_adaptive_ece(self):
        from athena2.evaluation.metrics import adaptive_calibration_error, expected_calibration_error

        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_prob = np.random.uniform(0, 1, 200)

        ace = adaptive_calibration_error(y_true, y_prob, n_bins=10)
        ece_adaptive = expected_calibration_error(y_true, y_prob, n_bins=10, strategy="adaptive")
        assert ace == ece_adaptive

    def test_report_has_ace(self):
        from athena2.evaluation.metrics import evaluate

        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)
        report = evaluate(y_true, y_prob)
        assert hasattr(report, "ace")
        assert report.ace == report.ece_adaptive

    def test_markdown_has_ace(self):
        from athena2.evaluation.metrics import evaluate

        np.random.seed(42)
        y_true = np.random.randint(0, 2, 50)
        y_prob = np.random.uniform(0, 1, 50)
        report = evaluate(y_true, y_prob)
        md = report.to_markdown()
        assert "ACE" in md
