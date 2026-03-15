"""Label noise detection using cleanlab confident learning.

Swiss judgment prediction datasets have ~8.8% label noise, especially
in the Italian subset. This module identifies and down-weights noisy
labels instead of discarding them.

Reference: Northcutt et al., "Confident Learning" (JAIR 2021)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NoiseReport:
    """Label noise detection report."""
    total_samples: int
    n_noisy: int
    noise_rate: float
    noise_by_language: dict[str, float]
    noise_by_label: dict[str, float]
    noisy_indices: np.ndarray
    sample_weights: np.ndarray  # 1.0 for clean, <1.0 for noisy

    def summary(self) -> str:
        lines = [
            f"Label Noise Report ({self.total_samples:,} samples)",
            f"  Detected noisy: {self.n_noisy:,} ({self.noise_rate:.1%})",
        ]
        if self.noise_by_language:
            lines.append("  By language:")
            for lang, rate in sorted(self.noise_by_language.items(), key=lambda x: -x[1]):
                lines.append(f"    {lang}: {rate:.1%}")
        if self.noise_by_label:
            lines.append("  By label:")
            for label, rate in sorted(self.noise_by_label.items()):
                lines.append(f"    {label}: {rate:.1%}")
        return "\n".join(lines)


def detect_label_noise(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    languages: np.ndarray | None = None,
    noise_weight: float = 0.5,
    frac_noise: float = None,
) -> NoiseReport:
    """Detect label noise using cleanlab confident learning.

    Args:
        labels: True labels, shape (N,).
        pred_probs: Cross-validated predicted probabilities, shape (N, C).
            Must be out-of-sample (e.g., from K-fold cross-validation).
        languages: Optional language labels for per-language analysis.
        noise_weight: Weight to assign to noisy samples (default 0.5).
            Clean samples get weight 1.0.
        frac_noise: Expected fraction of noise (None = auto-detect).

    Returns:
        NoiseReport with noise indices and sample weights.
    """
    try:
        from cleanlab.filter import find_label_issues
        from cleanlab.rank import get_label_quality_scores
    except ImportError:
        logger.warning(
            "cleanlab not installed. Install with: pip install cleanlab>=2.6"
        )
        return _fallback_noise_detection(labels, pred_probs, languages, noise_weight)

    labels = np.asarray(labels, dtype=int)
    pred_probs = np.asarray(pred_probs, dtype=float)

    # Find label issues using confident learning
    noisy_mask = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    # Get per-sample quality scores
    quality_scores = get_label_quality_scores(labels, pred_probs)

    # Build sample weights: clean=1.0, noisy=noise_weight
    sample_weights = np.ones(len(labels), dtype=np.float32)
    if isinstance(noisy_mask, np.ndarray) and noisy_mask.dtype == bool:
        noisy_indices = np.where(noisy_mask)[0]
    else:
        noisy_indices = np.asarray(noisy_mask, dtype=int)

    sample_weights[noisy_indices] = noise_weight

    # Per-language breakdown
    noise_by_language = {}
    if languages is not None:
        languages = np.asarray(languages)
        for lang in np.unique(languages):
            lang_mask = languages == lang
            n_lang = int(lang_mask.sum())
            n_noisy_lang = int(np.isin(np.where(lang_mask)[0], noisy_indices).sum())
            noise_by_language[str(lang)] = n_noisy_lang / max(n_lang, 1)

    # Per-label breakdown
    noise_by_label = {}
    for label_val in np.unique(labels):
        label_mask = labels == label_val
        n_label = int(label_mask.sum())
        n_noisy_label = int(np.isin(np.where(label_mask)[0], noisy_indices).sum())
        noise_by_label[str(label_val)] = n_noisy_label / max(n_label, 1)

    return NoiseReport(
        total_samples=len(labels),
        n_noisy=len(noisy_indices),
        noise_rate=len(noisy_indices) / max(len(labels), 1),
        noise_by_language=noise_by_language,
        noise_by_label=noise_by_label,
        noisy_indices=noisy_indices,
        sample_weights=sample_weights,
    )


def _fallback_noise_detection(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    languages: np.ndarray | None,
    noise_weight: float,
) -> NoiseReport:
    """Simple fallback when cleanlab is not available.

    Uses prediction confidence disagreement to flag potentially noisy labels.
    """
    labels = np.asarray(labels, dtype=int)
    pred_probs = np.asarray(pred_probs, dtype=float)

    # Flag samples where predicted class disagrees with label AND model is confident
    pred_class = pred_probs.argmax(axis=1)
    disagreement = pred_class != labels
    max_prob = pred_probs.max(axis=1)
    confident_disagreement = disagreement & (max_prob > 0.8)

    noisy_indices = np.where(confident_disagreement)[0]
    sample_weights = np.ones(len(labels), dtype=np.float32)
    sample_weights[noisy_indices] = noise_weight

    noise_by_language = {}
    if languages is not None:
        languages = np.asarray(languages)
        for lang in np.unique(languages):
            lang_mask = languages == lang
            n_lang = int(lang_mask.sum())
            n_noisy = int(confident_disagreement[lang_mask].sum())
            noise_by_language[str(lang)] = n_noisy / max(n_lang, 1)

    noise_by_label = {}
    for label_val in np.unique(labels):
        label_mask = labels == label_val
        n_label = int(label_mask.sum())
        n_noisy = int(confident_disagreement[label_mask].sum())
        noise_by_label[str(label_val)] = n_noisy / max(n_label, 1)

    return NoiseReport(
        total_samples=len(labels),
        n_noisy=len(noisy_indices),
        noise_rate=len(noisy_indices) / max(len(labels), 1),
        noise_by_language=noise_by_language,
        noise_by_label=noise_by_label,
        noisy_indices=noisy_indices,
        sample_weights=sample_weights,
    )


def generate_cross_val_probs(
    texts: list[str],
    labels: np.ndarray,
    n_folds: int = 5,
) -> np.ndarray:
    """Generate out-of-sample predicted probabilities via K-fold cross-validation.

    Uses TF-IDF + Logistic Regression for speed (cleanlab needs out-of-sample preds).

    Args:
        texts: Input texts.
        labels: True labels.
        n_folds: Number of cross-validation folds.

    Returns:
        Out-of-sample predicted probabilities, shape (N, n_classes).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    labels = np.asarray(labels, dtype=int)
    n_classes = len(np.unique(labels))
    pred_probs = np.zeros((len(labels), n_classes), dtype=np.float64)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        logger.info(f"  Noise detection fold {fold+1}/{n_folds}...")

        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]

        vectorizer = TfidfVectorizer(
            max_features=30000, ngram_range=(1, 2), sublinear_tf=True,
        )
        X_train = vectorizer.fit_transform(train_texts)
        X_val = vectorizer.transform(val_texts)

        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", n_jobs=-1)
        clf.fit(X_train, labels[train_idx])

        pred_probs[val_idx] = clf.predict_proba(X_val)

    return pred_probs
