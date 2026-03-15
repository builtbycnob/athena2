#!/usr/bin/env python3
"""ATHENA2 Phase 1.2-1.4: Baselines + Data Quality Report.

Runs TF-IDF baseline and benchmarks BOTH transformer encoders on official
SJP-XL splits. Picks winner for all subsequent phases.

Usage:
    uv run python scripts/phase2_baselines.py                  # Full pipeline
    uv run python scripts/phase2_baselines.py --step tfidf     # TF-IDF only
    uv run python scripts/phase2_baselines.py --step encoder_a # Legal-Swiss-RoBERTa
    uv run python scripts/phase2_baselines.py --step encoder_b # Legal-XLM-R
    uv run python scripts/phase2_baselines.py --step noise     # Label noise analysis

Requires: pip install athena[worldmodel]
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("athena2.phase2")

# MPS safety
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def load_sjp_xl(processed_dir: Path) -> pd.DataFrame:
    """Load processed SJP-XL dataset."""
    path = processed_dir / "sjp_xl.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"SJP-XL not found at {path}. Run phase1_data_foundation.py first."
        )
    df = pd.read_parquet(path)

    # Map string labels to integers if needed (HF dataset stores strings)
    if df["label"].dtype == object or str(df["label"].dtype) == "str":
        label_map = {"dismissal": 0, "approval": 1}
        df["label"] = df["label"].map(label_map)
        unmapped = df["label"].isna().sum()
        if unmapped > 0:
            logger.warning(f"Dropping {unmapped} rows with unmapped labels")
            df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

    logger.info(f"Loaded SJP-XL: {len(df):,} rows")
    return df


def apply_official_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply official SJP-XL temporal splits.

    Official: train≤2015, val 2016-17, test 2018-22.
    """
    train = df[df["year"] <= 2015].copy()
    val = df[df["year"].isin([2016, 2017])].copy()
    test = df[df["year"] >= 2018].copy()

    logger.info(f"Official splits: train={len(train):,}, val={len(val):,}, test={len(test):,}")
    return train, val, test


def step_tfidf(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, output_dir: Path) -> dict:
    """Phase 1.2: TF-IDF + Logistic Regression baseline."""
    from athena2.evaluation.metrics import evaluate, conformal_calibrate
    from athena2.models.baselines import TFIDFBaseline

    logger.info("=" * 60)
    logger.info("PHASE 1.2: TF-IDF Baseline")
    logger.info("=" * 60)

    baseline = TFIDFBaseline(max_features=50000, ngram_range=(1, 3))
    train_metrics = baseline.train(
        train["facts"].tolist(),
        train["label"].values,
    )

    # Evaluate on validation
    val_probs = baseline.predict_proba(val["facts"].tolist())
    threshold = conformal_calibrate(val["label"].values, val_probs)

    # Evaluate on test
    test_probs = baseline.predict_proba(test["facts"].tolist())
    report = evaluate(
        test["label"].values,
        test_probs,
        conformal_threshold=threshold,
        metadata={"model": "TF-IDF + LR", "phase": "1.2"},
    )

    # Per-language breakdown
    for lang in ["de", "fr", "it"]:
        mask = test["language"].values == lang
        if mask.sum() > 0:
            lang_report = evaluate(test["label"].values[mask], test_probs[mask])
            logger.info(f"  {lang}: F1={lang_report.macro_f1:.4f}, ACC={lang_report.accuracy:.4f}")

    # Save
    model_dir = output_dir / "tfidf_baseline"
    baseline.save(model_dir)
    report_md = report.to_markdown()
    (output_dir / "tfidf_report.md").write_text(report_md)

    logger.info(f"\nTF-IDF Results:")
    logger.info(f"  Macro F1: {report.macro_f1:.4f}")
    logger.info(f"  Accuracy: {report.accuracy:.4f}")
    logger.info(f"  ACE: {report.ace:.4f}")
    logger.info(f"  Brier: {report.brier.brier:.4f}")

    return {"tfidf": {"f1": report.macro_f1, "acc": report.accuracy, "ace": report.ace}}


def step_transformer_baseline(
    encoder_name: str,
    encoder_label: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Phase 1.3: Transformer baseline (single encoder)."""
    from athena2.evaluation.metrics import evaluate, conformal_calibrate
    from athena2.models.baselines import TransformerBaseline

    logger.info("=" * 60)
    logger.info(f"PHASE 1.3: Transformer Baseline — {encoder_label}")
    logger.info(f"  Encoder: {encoder_name}")
    logger.info("=" * 60)

    baseline = TransformerBaseline(
        model_name=encoder_name,
        max_length=512,
        batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        epochs=3,
    )

    model_dir = output_dir / f"transformer_{encoder_label}"
    train_metrics = baseline.train(
        train["facts"].tolist(),
        train["label"].values,
        val["facts"].tolist(),
        val["label"].values,
        output_dir=model_dir,
    )

    # Evaluate
    val_probs = baseline.predict_proba(val["facts"].tolist())
    threshold = conformal_calibrate(val["label"].values, val_probs)

    test_probs = baseline.predict_proba(test["facts"].tolist())
    test_logits = baseline.get_logits(test["facts"].tolist())
    report = evaluate(
        test["label"].values,
        test_probs,
        conformal_threshold=threshold,
        metadata={"model": encoder_name, "phase": "1.3", "label": encoder_label},
    )

    # Per-language breakdown
    for lang in ["de", "fr", "it"]:
        mask = test["language"].values == lang
        if mask.sum() > 0:
            lang_report = evaluate(test["label"].values[mask], test_probs[mask])
            logger.info(f"  {lang}: F1={lang_report.macro_f1:.4f}, ACC={lang_report.accuracy:.4f}")

    report_md = report.to_markdown()
    (output_dir / f"transformer_{encoder_label}_report.md").write_text(report_md)

    # Save logits for temperature scaling
    np.save(model_dir / "test_logits.npy", test_logits)
    np.save(model_dir / "test_probs.npy", test_probs)
    np.save(model_dir / "test_labels.npy", test["label"].values)

    logger.info(f"\n{encoder_label} Results:")
    logger.info(f"  Macro F1: {report.macro_f1:.4f}")
    logger.info(f"  Accuracy: {report.accuracy:.4f}")
    logger.info(f"  ACE: {report.ace:.4f}")
    logger.info(f"  Brier: {report.brier.brier:.4f}")

    return {encoder_label: {"f1": report.macro_f1, "acc": report.accuracy, "ace": report.ace}}


def step_noise_analysis(train: pd.DataFrame, output_dir: Path) -> dict:
    """Phase 1.4: Label noise detection using cleanlab."""
    from athena2.data.noise_detection import detect_label_noise, generate_cross_val_probs

    logger.info("=" * 60)
    logger.info("PHASE 1.4: Label Noise Detection")
    logger.info("=" * 60)

    texts = train["facts"].tolist()
    labels = train["label"].values
    languages = train["language"].values if "language" in train.columns else None

    # Generate cross-validated predictions
    logger.info("Generating cross-validated predictions (5-fold)...")
    pred_probs = generate_cross_val_probs(texts, labels, n_folds=5)

    # Detect noise
    report = detect_label_noise(
        labels, pred_probs,
        languages=languages,
        noise_weight=0.5,
    )

    logger.info(report.summary())

    # Save
    noise_dir = output_dir / "noise_analysis"
    noise_dir.mkdir(parents=True, exist_ok=True)
    np.save(noise_dir / "sample_weights.npy", report.sample_weights)
    np.save(noise_dir / "noisy_indices.npy", report.noisy_indices)
    (noise_dir / "noise_report.txt").write_text(report.summary())

    return {"noise_rate": report.noise_rate, "n_noisy": report.n_noisy}


def main():
    parser = argparse.ArgumentParser(description="ATHENA2 Phase 1.2-1.4: Baselines")
    parser.add_argument("--step", choices=["tfidf", "encoder_a", "encoder_b", "noise", "all"],
                        default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--sample", type=int, default=0,
                        help="Subsample N cases per split for smoke testing (0=full)")
    args = parser.parse_args()

    t0 = time.time()

    # Load data
    df = load_sjp_xl(args.data_dir)
    train, val, test = apply_official_splits(df)

    if args.sample > 0:
        train = train.sample(min(args.sample, len(train)), random_state=42)
        val = val.sample(min(args.sample // 2, len(val)), random_state=42)
        test = test.sample(min(args.sample // 2, len(test)), random_state=42)
        logger.info(f"Smoke test: train={len(train)}, val={len(val)}, test={len(test)}")

    results = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.step in ("tfidf", "all"):
        results.update(step_tfidf(train, val, test, args.output_dir))

    if args.step in ("encoder_a", "all"):
        results.update(step_transformer_baseline(
            "joelniklaus/legal-swiss-roberta-large",
            "legal_swiss_roberta",
            train, val, test, args.output_dir,
        ))

    if args.step in ("encoder_b", "all"):
        results.update(step_transformer_baseline(
            "joelniklaus/legal-xlm-roberta-large",
            "legal_xlm_roberta",
            train, val, test, args.output_dir,
        ))

    if args.step in ("noise", "all"):
        results.update(step_noise_analysis(train, args.output_dir))

    # Summary
    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Phase 1.2-1.4 complete in {elapsed/60:.1f} min")
    logger.info(f"{'=' * 60}")
    logger.info(json.dumps(results, indent=2))

    # Save summary
    (args.output_dir / "baseline_results.json").write_text(json.dumps(results, indent=2))

    # Pick winner
    if "legal_swiss_roberta" in results and "legal_xlm_roberta" in results:
        a = results["legal_swiss_roberta"]["f1"]
        b = results["legal_xlm_roberta"]["f1"]
        winner = "legal-swiss-roberta-large" if a >= b else "legal-xlm-roberta-large"
        logger.info(f"\nWINNER: {winner} (F1: {max(a, b):.4f} vs {min(a, b):.4f})")


if __name__ == "__main__":
    main()
