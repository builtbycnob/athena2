"""Baseline models for Swiss Judgment Prediction.

Establishes floor performance to beat:
1. TF-IDF + Logistic Regression (simplest baseline)
2. Fine-tuned XLM-RoBERTa on facts → label
3. Hierarchical encoder for long documents

Published SOTA: ~71% macro F1 (Niklaus et al. 2024, Joint Training + DA).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── TF-IDF Baseline ───────────────────────────────────────────────

@dataclass
class TFIDFBaseline:
    """TF-IDF + Logistic Regression baseline.

    Expected: ~63-64% macro F1 (published SVM baseline).
    """
    max_features: int = 50000
    ngram_range: tuple[int, int] = (1, 3)
    sublinear_tf: bool = True
    min_df: int = 5
    C: float = 1.0
    max_iter: int = 1000

    vectorizer: Any = None
    classifier: Any = None

    def train(self, texts: list[str], labels: np.ndarray) -> dict[str, float]:
        """Train TF-IDF + LR on training data.

        Args:
            texts: List of fact strings.
            labels: Binary labels (0=dismissal, 1=approval).

        Returns:
            Training metrics dict.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        logger.info(f"Training TF-IDF baseline on {len(texts):,} samples...")
        t0 = time.time()

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            min_df=self.min_df,
            strip_accents="unicode",
        )

        X = self.vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF matrix: {X.shape} ({X.nnz:,} non-zero)")

        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            n_jobs=-1,
        )

        self.classifier.fit(X, labels)
        elapsed = time.time() - t0

        # Training accuracy
        train_pred = self.classifier.predict(X)
        train_acc = float(np.mean(train_pred == labels))

        logger.info(f"TF-IDF trained in {elapsed:.1f}s — train accuracy: {train_acc:.3f}")
        return {"train_accuracy": train_acc, "train_time_s": elapsed}

    def predict(self, texts: list[str]) -> np.ndarray:
        """Predict labels for new texts."""
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict probability of class 1 (approval)."""
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        """Save model to disk."""
        import joblib
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path / "vectorizer.joblib")
        joblib.dump(self.classifier, path / "classifier.joblib")

    def load(self, path: Path) -> None:
        """Load model from disk."""
        import joblib
        self.vectorizer = joblib.load(path / "vectorizer.joblib")
        self.classifier = joblib.load(path / "classifier.joblib")


# ── Transformer Baseline ──────────────────────────────────────────

@dataclass
class TransformerBaseline:
    """Fine-tuned transformer on facts → label.

    Uses Legal Swiss RoBERTa or XLM-RoBERTa as backbone.
    Expected: ~68-70% macro F1.
    """
    model_name: str = "xlm-roberta-base"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 5
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    seed: int = 42

    model: Any = None
    tokenizer: Any = None

    def train(
        self,
        train_texts: list[str],
        train_labels: np.ndarray,
        val_texts: list[str] | None = None,
        val_labels: np.ndarray | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, float]:
        """Fine-tune transformer on training data.

        Args:
            train_texts: Training fact strings.
            train_labels: Binary training labels.
            val_texts: Validation texts (optional).
            val_labels: Validation labels (optional).
            output_dir: Directory for checkpoints.

        Returns:
            Training metrics.
        """
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        from torch.utils.data import Dataset

        logger.info(f"Fine-tuning {self.model_name} on {len(train_texts):,} samples...")

        # Set seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Some older models use LayerNorm.gamma/beta instead of weight/bias.
        # Load base model first, fix state dict, then build classifier on top.
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(self.model_name, num_labels=2)
        self.model = AutoModelForSequenceClassification.from_config(config)

        # Load pretrained weights with key remapping
        from safetensors.torch import load_file as load_safetensors
        from huggingface_hub import hf_hub_download
        import os

        try:
            weight_path = hf_hub_download(self.model_name, "model.safetensors")
            pretrained_state = load_safetensors(weight_path)
        except Exception:
            weight_path = hf_hub_download(self.model_name, "pytorch_model.bin")
            pretrained_state = torch.load(weight_path, map_location="cpu", weights_only=True)

        # Remap gamma/beta → weight/bias for LayerNorm compatibility
        remapped = {}
        for k, v in pretrained_state.items():
            new_k = k.replace(".gamma", ".weight").replace(".beta", ".bias")
            remapped[new_k] = v

        # Load into model (strict=False allows missing classifier head)
        missing, unexpected = self.model.load_state_dict(remapped, strict=False)
        if missing:
            classifier_missing = [k for k in missing if "classifier" in k]
            other_missing = [k for k in missing if "classifier" not in k]
            if other_missing:
                logger.warning(f"Missing non-classifier keys: {other_missing}")
            if classifier_missing:
                logger.info(f"Classifier head initialized randomly (expected): {len(classifier_missing)} keys")

        # Device
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Device: {device}")

        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.texts[idx],
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
                }

        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = None
        if val_texts is not None and val_labels is not None:
            val_dataset = TextDataset(val_texts, val_labels, self.tokenizer, self.max_length)

        if output_dir is None:
            output_dir = Path("data/models/transformer_baseline")

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_f1" if val_dataset else None,
            logging_steps=50,
            seed=self.seed,
            dataloader_num_workers=0,  # MPS doesn't support multiprocessing well
        )

        def compute_metrics(eval_pred):
            from athena2.evaluation.metrics import accuracy, macro_f1
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy(labels, preds),
                "f1": macro_f1(labels, preds, labels=[0, 1]),
            }

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics if val_dataset else None,
        )

        t0 = time.time()
        train_result = trainer.train()
        elapsed = time.time() - t0

        logger.info(f"Training complete in {elapsed:.1f}s")
        return {
            "train_loss": train_result.training_loss,
            "train_time_s": elapsed,
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        }

    def predict(self, texts: list[str]) -> np.ndarray:
        """Predict labels."""
        import torch

        self.model.eval()
        device = next(self.model.parameters()).device
        preds = []

        for i in range(0, len(texts), self.batch_size * 2):
            batch = texts[i:i + self.batch_size * 2]
            encoding = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                logits = self.model(**encoding).logits
                preds.extend(logits.argmax(dim=-1).cpu().numpy())

        return np.array(preds)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict probability of class 1 (approval)."""
        import torch

        self.model.eval()
        device = next(self.model.parameters()).device
        probs = []

        for i in range(0, len(texts), self.batch_size * 2):
            batch = texts[i:i + self.batch_size * 2]
            encoding = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                logits = self.model(**encoding).logits
                prob = torch.softmax(logits, dim=-1)[:, 1]
                probs.extend(prob.cpu().numpy())

        return np.array(probs)

    def get_logits(self, texts: list[str]) -> np.ndarray:
        """Get raw logits (for temperature scaling calibration)."""
        import torch

        self.model.eval()
        device = next(self.model.parameters()).device
        all_logits = []

        for i in range(0, len(texts), self.batch_size * 2):
            batch = texts[i:i + self.batch_size * 2]
            encoding = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                logits = self.model(**encoding).logits
                all_logits.extend(logits[:, 1].cpu().numpy())  # Logit for class 1

        return np.array(all_logits)
