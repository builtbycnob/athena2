"""Tests for ATHENA2 baseline models."""
import numpy as np
import pytest


class TestTFIDFBaseline:
    def test_train_and_predict(self):
        from athena2.models.baselines import TFIDFBaseline

        texts = ["This case is dismissed"] * 50 + ["Appeal is granted"] * 50
        labels = np.array([0] * 50 + [1] * 50)
        baseline = TFIDFBaseline(max_features=100, ngram_range=(1, 2))
        metrics = baseline.train(texts, labels)
        assert "train_accuracy" in metrics
        preds = baseline.predict(texts[:5])
        assert len(preds) == 5

    def test_predict_proba(self):
        from athena2.models.baselines import TFIDFBaseline

        texts = ["dismiss"] * 30 + ["approve"] * 30
        labels = np.array([0] * 30 + [1] * 30)
        baseline = TFIDFBaseline(max_features=50)
        baseline.train(texts, labels)
        probs = baseline.predict_proba(texts[:5])
        assert all(0 <= p <= 1 for p in probs)

    def test_save_load(self, tmp_path):
        from athena2.models.baselines import TFIDFBaseline

        texts = ["dismiss"] * 30 + ["approve"] * 30
        labels = np.array([0] * 30 + [1] * 30)
        baseline = TFIDFBaseline(max_features=50)
        baseline.train(texts, labels)
        baseline.save(tmp_path)

        loaded = TFIDFBaseline()
        loaded.load(tmp_path)
        preds_orig = baseline.predict(texts[:5])
        preds_loaded = loaded.predict(texts[:5])
        np.testing.assert_array_equal(preds_orig, preds_loaded)


class TestTransformerBaseline:
    def test_init(self):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from athena2.models.baselines import TransformerBaseline

        tb = TransformerBaseline(model_name="xlm-roberta-base", max_length=64)
        assert tb.model_name == "xlm-roberta-base"
        assert tb.max_length == 64
