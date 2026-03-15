"""Batch inference engine for ATHENA2.

High-throughput prediction on large case datasets.
Target: 10K cases in <60s on M3 Ultra (batch=64).

Supports:
- PyTorch MPS batch inference
- INT8 quantization for production
- Progress tracking and crash-safe checkpointing
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BatchPrediction:
    """Prediction for a single case."""
    decision_id: str
    p_dismissal: float
    p_approval: float
    predicted_verdict: str
    prediction_set_90: list[str]
    prediction_set_95: list[str]
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of batch inference."""
    predictions: list[BatchPrediction]
    total_time_s: float
    cases_per_second: float
    model_name: str
    device: str
    batch_size: int

    def to_csv(self, path: Path) -> None:
        """Save predictions to CSV."""
        import pandas as pd

        rows = []
        for p in self.predictions:
            rows.append({
                "decision_id": p.decision_id,
                "p_dismissal": p.p_dismissal,
                "p_approval": p.p_approval,
                "predicted_verdict": p.predicted_verdict,
                "prediction_set_90": str(p.prediction_set_90),
                "prediction_set_95": str(p.prediction_set_95),
                **{f"feature_{k}": v for k, v in p.features.items()},
            })
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(rows):,} predictions to {path}")


class BatchInferenceEngine:
    """High-throughput batch inference engine.

    Args:
        model: Trained ATHENA2 model (LegalWorldModel instance).
        batch_size: Inference batch size (default 64).
        device: Device for inference (mps/cpu).
        quantize: If True, apply INT8 quantization.
    """

    def __init__(
        self,
        model: Any,  # LegalWorldModel
        batch_size: int = 64,
        device: str = "mps",
        quantize: bool = False,
    ):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.quantize = quantize
        self._prepared = False

    def prepare(self) -> None:
        """Prepare model for inference (move to device, set eval, optionally quantize)."""
        import torch

        if self.model._model is None:
            raise RuntimeError("Model not built. Call model.build() first.")

        self.model._model.eval()

        if self.quantize:
            try:
                self.model._model = torch.quantization.quantize_dynamic(
                    self.model._model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
                logger.info("INT8 dynamic quantization applied")
            except Exception as e:
                logger.warning(f"Quantization failed (MPS may not support it): {e}")

        device = torch.device(self.device)
        self.model._model.to(device)
        self._prepared = True
        logger.info(f"Model prepared on {self.device}")

    def predict_batch(
        self,
        texts: list[str],
        decision_ids: list[str] | None = None,
        conformal_thresholds_90: dict[int, float] | None = None,
        conformal_thresholds_95: dict[int, float] | None = None,
        with_features: bool = True,
        checkpoint_path: Path | None = None,
        checkpoint_every: int = 1000,
    ) -> BatchResult:
        """Run batch inference on a list of texts.

        Args:
            texts: Input fact texts.
            decision_ids: Optional case IDs (for tracking).
            conformal_thresholds_90: 90% coverage conformal thresholds.
            conformal_thresholds_95: 95% coverage conformal thresholds.
            with_features: If True, include intermediate feature predictions.
            checkpoint_path: Path for crash-safe checkpointing.
            checkpoint_every: Checkpoint frequency (in cases).

        Returns:
            BatchResult with all predictions.
        """
        import torch

        if not self._prepared:
            self.prepare()

        if decision_ids is None:
            decision_ids = [f"case_{i}" for i in range(len(texts))]

        predictions = []
        t0 = time.time()
        device = next(self.model._model.parameters()).device

        # Resume from checkpoint if available
        start_idx = 0
        if checkpoint_path and checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text())
            start_idx = checkpoint.get("processed", 0)
            logger.info(f"Resuming from checkpoint at index {start_idx}")

        for i in range(start_idx, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_ids = decision_ids[i:i + self.batch_size]

            encoding = self.model._tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.model.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = self.model._model(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                    return_features=with_features,
                )

                verdict_prob = torch.softmax(outputs["verdict_logits"], dim=-1)

                for j in range(len(batch_texts)):
                    p_dismiss = float(verdict_prob[j, 0])
                    p_approve = float(verdict_prob[j, 1])
                    verdict = "approval" if p_approve > 0.5 else "dismissal"

                    # Conformal prediction sets
                    probs_2d = np.array([[p_dismiss, p_approve]])
                    set_90 = _make_prediction_set(probs_2d[0], conformal_thresholds_90)
                    set_95 = _make_prediction_set(probs_2d[0], conformal_thresholds_95)

                    # Features
                    features = {}
                    if with_features:
                        law_prob = torch.softmax(outputs["law_area_logits"][j], dim=-1)
                        error_prob = torch.softmax(outputs["error_logits"][j], dim=-1)
                        features = {
                            "law_area_max": int(law_prob.argmax()),
                            "law_area_conf": float(law_prob.max()),
                            "has_decisive_error": float(error_prob[1]),
                        }

                    predictions.append(BatchPrediction(
                        decision_id=batch_ids[j],
                        p_dismissal=p_dismiss,
                        p_approval=p_approve,
                        predicted_verdict=verdict,
                        prediction_set_90=set_90,
                        prediction_set_95=set_95,
                        features=features,
                    ))

            # Progress logging
            processed = i + len(batch_texts)
            if processed % 1000 == 0 or processed == len(texts):
                elapsed = time.time() - t0
                rate = processed / max(elapsed, 0.001)
                logger.info(f"  {processed:,}/{len(texts):,} ({rate:.0f} cases/s)")

            # Checkpoint
            if checkpoint_path and processed % checkpoint_every == 0:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_text(json.dumps({"processed": processed}))

        elapsed = time.time() - t0
        rate = len(texts) / max(elapsed, 0.001)

        # Clean up checkpoint
        if checkpoint_path and checkpoint_path.exists():
            checkpoint_path.unlink()

        logger.info(f"Batch inference: {len(texts):,} cases in {elapsed:.1f}s ({rate:.0f}/s)")

        return BatchResult(
            predictions=predictions,
            total_time_s=elapsed,
            cases_per_second=rate,
            model_name=self.model.encoder_name,
            device=self.device,
            batch_size=self.batch_size,
        )


def _make_prediction_set(
    probs: np.ndarray,
    thresholds: dict[int, float] | None,
) -> list[str]:
    """Make a prediction set from probabilities and thresholds."""
    labels = ["dismissal", "approval"]
    if thresholds is None:
        return [labels[int(np.argmax(probs))]]

    pset = []
    for cls in range(len(probs)):
        nonconformity = 1.0 - probs[cls]
        if nonconformity <= thresholds.get(cls, 1.0):
            pset.append(labels[cls])

    if not pset:
        pset = [labels[int(np.argmax(probs))]]

    return pset
