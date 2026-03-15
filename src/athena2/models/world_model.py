"""ATHENA2 Intermediate Reasoning Predictor — Legal Judgment Dynamics.

LUPI-based (Learning Using Privileged Information, Vapnik 2009) multi-task model
for Swiss Federal Supreme Court judgment prediction.

Architecture:
  1. Legal Encoder: Legal-Swiss-RoBERTa-Large (638K Swiss decisions, 340M params)
  2. LUPI Feature Prediction Heads: multi-task intermediate features
     - Law area (17 classes) — from dataset labels
     - Error presence (binary) — from considerations (LUPI privileged info)
     - Reasoning pattern (9 classes) — from considerations (LUPI privileged info)
     - Outcome granular (7 classes) — from considerations (LUPI privileged info)
  3. Dynamics MLP (Intermediate Reasoning Predictor):
     encoder CLS (1024) ⊕ feature logits (~36) ⊕ GAT (64, optional) → verdict
  4. Calibration: BSCE-GRA (training) + temperature/isotonic (post-hoc)
     + class-conditional conformal prediction

The key insight (LUPI): considerations field from SJP-XL provides privileged
supervision for intermediate features at TRAINING time. At INFERENCE, the model
predicts these from facts alone — producing an explainable reasoning chain.

Training signal:
  facts → [encoder] → z
  z → [law_area_head] → law_area_pred  (supervised by law_area label)
  z → [error_head] → error_pred        (supervised by LLM-extracted features — LUPI)
  z → [reasoning_head] → pattern_pred  (supervised by LLM-extracted features — LUPI)
  z → [outcome_head] → outcome_pred    (supervised by LLM-extracted features — LUPI)
  z + features [+ GAT] → [dynamics] → verdict_prob (supervised by label)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Focal Loss (numpy, for evaluation only) ──────────────────────

def focal_loss_numpy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.5,
) -> float:
    """Focal loss for calibrated training (numpy, for evaluation).

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    p_t = np.where(y_true == 1, y_prob, 1 - y_prob)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    loss = -alpha_t * (1 - p_t) ** gamma * np.log(p_t)
    return float(np.mean(loss))


# ── Intermediate Reasoning Predictor (PyTorch) ───────────────────

class LegalWorldModel:
    """LUPI-based multi-task legal judgment model.

    Combines a pre-trained legal encoder with prediction heads for
    intermediate legal features (privileged info at training time)
    and final verdict prediction.

    Terminology:
    - "Intermediate Reasoning Predictor" (not "world model") — honest framing
    - "LUPI feature heads" — considerations are privileged information (Vapnik 2009)
    - "Dynamics MLP" — maps encoder CLS + predicted features [+ GAT] → verdict

    Args:
        encoder_name: HuggingFace model identifier.
        max_length: Maximum input sequence length.
        hidden_size: Encoder hidden dimension.
        dynamics_hidden: Dynamics MLP hidden dimension.
        dynamics_layers: Number of MLP layers.
        dynamics_dropout: Dropout rate in MLP.
        n_law_areas: Number of law area classes (17 for SJP-XL).
        n_reasoning_patterns: Number of reasoning pattern classes.
        n_outcome_granular: Number of granular outcome classes.
        gat_dim: GAT feature dimension (0 to disable).
        use_bsce_gra: Use BSCE-GRA loss instead of focal loss.
    """

    def __init__(
        self,
        encoder_name: str = "joelniklaus/legal-swiss-roberta-large",
        max_length: int = 512,
        hidden_size: int = 1024,
        dynamics_hidden: int = 512,
        dynamics_layers: int = 4,
        dynamics_dropout: float = 0.1,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.5,
        n_law_areas: int = 17,
        n_reasoning_patterns: int = 9,
        n_outcome_granular: int = 7,
        gat_dim: int = 0,
        use_bsce_gra: bool = True,
        feature_head_weights: dict[str, float] | None = None,
    ):
        self.encoder_name = encoder_name
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.dynamics_hidden = dynamics_hidden
        self.dynamics_layers = dynamics_layers
        self.dynamics_dropout = dynamics_dropout
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.n_law_areas = n_law_areas
        self.n_reasoning_patterns = n_reasoning_patterns
        self.n_outcome_granular = n_outcome_granular
        self.gat_dim = gat_dim
        self.use_bsce_gra = use_bsce_gra

        self.feature_head_weights = feature_head_weights or {
            "law_area": 0.3,
            "error_presence": 0.3,
            "reasoning_pattern": 0.1,
            "outcome_granular": 0.3,
        }

        self._model = None
        self._tokenizer = None

    def build(self) -> None:
        """Build the full model architecture.

        Requires PyTorch. Called explicitly to allow configuration before building.
        """
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer

        def _load_encoder_with_remap(model_name: str):
            """Load encoder, remapping LayerNorm gamma/beta → weight/bias if needed."""
            from pathlib import Path

            # Local path (e.g., SupCon pretrained) — load directly
            if Path(model_name).exists():
                return AutoModel.from_pretrained(model_name)

            # HuggingFace model — check for gamma/beta naming and remap
            from transformers import AutoConfig
            from huggingface_hub import hf_hub_download

            enc_config = AutoConfig.from_pretrained(model_name)
            encoder = AutoModel.from_config(enc_config)

            try:
                from safetensors.torch import load_file as load_safetensors
                weight_path = hf_hub_download(model_name, "model.safetensors")
                pretrained_state = load_safetensors(weight_path)
            except Exception:
                weight_path = hf_hub_download(model_name, "pytorch_model.bin")
                pretrained_state = torch.load(weight_path, map_location="cpu", weights_only=True)

            # Remap gamma/beta → weight/bias (no-op if already correct)
            remapped = {
                k.replace(".gamma", ".weight").replace(".beta", ".bias"): v
                for k, v in pretrained_state.items()
            }
            missing, unexpected = encoder.load_state_dict(remapped, strict=False)
            if unexpected:
                logger.debug(f"Unexpected keys during encoder load: {len(unexpected)}")
            return encoder

        logger.info(f"Building Intermediate Reasoning Predictor with encoder: {self.encoder_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)

        class IntermediateReasoningPredictor(nn.Module):
            """LUPI-based multi-task model with Intermediate Reasoning Predictor."""

            def __init__(self, config):
                super().__init__()
                self.encoder = _load_encoder_with_remap(config.encoder_name)
                h = config.hidden_size

                # LUPI Feature prediction heads (multi-task)
                # These predict intermediate features that are supervised by
                # privileged information (considerations) at training time
                self.law_area_head = nn.Linear(h, config.n_law_areas)
                self.error_head = nn.Linear(h, 2)  # Binary: has_decisive_error
                self.reasoning_head = nn.Linear(h, config.n_reasoning_patterns)
                self.outcome_granular_head = nn.Linear(h, config.n_outcome_granular)

                # Feature dimension: probabilities from each head
                feature_dim = (
                    config.n_law_areas +
                    2 +  # error
                    config.n_reasoning_patterns +
                    config.n_outcome_granular
                )

                # Dynamics MLP (Intermediate Reasoning Predictor)
                # Input: encoder CLS + soft features + GAT (optional)
                total_input_dim = h + feature_dim + config.gat_dim

                layers = []
                in_dim = total_input_dim
                # 4-layer MLP: input → 512 → 256 → 128 → 2
                hidden_sizes = [512, 256, 128]
                for i, out_dim in enumerate(hidden_sizes):
                    layers.extend([
                        nn.Linear(in_dim, out_dim),
                        nn.GELU(),
                        nn.LayerNorm(out_dim),
                        nn.Dropout(config.dynamics_dropout),
                    ])
                    in_dim = out_dim

                self.dynamics = nn.Sequential(*layers)
                self.verdict_head = nn.Linear(128, 2)

            def forward(
                self,
                input_ids,
                attention_mask,
                gat_features=None,
                return_features=False,
            ):
                # Encode facts
                encoder_output = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                z = encoder_output.last_hidden_state[:, 0, :]  # CLS token

                # Predict intermediate features (LUPI heads)
                law_area_logits = self.law_area_head(z)
                error_logits = self.error_head(z)
                reasoning_logits = self.reasoning_head(z)
                outcome_logits = self.outcome_granular_head(z)

                # Soft features (probabilities for differentiability)
                law_area_probs = torch.softmax(law_area_logits, dim=-1)
                error_probs = torch.softmax(error_logits, dim=-1)
                reasoning_probs = torch.softmax(reasoning_logits, dim=-1)
                outcome_probs = torch.softmax(outcome_logits, dim=-1)

                features = torch.cat([
                    law_area_probs, error_probs, reasoning_probs, outcome_probs,
                ], dim=-1)

                # Dynamics input: CLS + features + GAT (optional)
                dynamics_components = [z, features]
                if gat_features is not None:
                    dynamics_components.append(gat_features)

                dynamics_input = torch.cat(dynamics_components, dim=-1)
                dynamics_output = self.dynamics(dynamics_input)
                verdict_logits = self.verdict_head(dynamics_output)

                result = {
                    "verdict_logits": verdict_logits,
                    "law_area_logits": law_area_logits,
                    "error_logits": error_logits,
                    "reasoning_logits": reasoning_logits,
                    "outcome_logits": outcome_logits,
                }

                if return_features:
                    result["encoding"] = z
                    result["features"] = features

                return result

        self._model = IntermediateReasoningPredictor(self)
        n_params = sum(p.numel() for p in self._model.parameters())
        n_trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        logger.info(f"Intermediate Reasoning Predictor built: {n_params:,} params ({n_trainable:,} trainable)")

    def compute_loss(
        self,
        outputs: dict,
        verdict_labels: Any,
        law_area_labels: Any = None,
        error_labels: Any = None,
        reasoning_labels: Any = None,
        outcome_labels: Any = None,
        sample_weights: Any = None,
    ) -> dict:
        """Compute multi-task loss with BSCE-GRA for verdict.

        Returns dict with 'total', 'verdict', 'law_area', 'error',
        'reasoning', 'outcome' loss components.
        """
        import torch
        import torch.nn.functional as F

        losses = {}

        # Verdict loss
        verdict_logits = outputs["verdict_logits"]
        if self.use_bsce_gra:
            from athena2.models.bsce_gra import BSCEGRALoss
            bsce = BSCEGRALoss(num_classes=2)
            losses["verdict"] = bsce(verdict_logits, verdict_labels, sample_weights)
        else:
            # Fallback: focal loss
            verdict_probs = torch.softmax(verdict_logits, dim=-1)
            p_t = verdict_probs.gather(1, verdict_labels.unsqueeze(1)).squeeze(1)
            alpha_t = torch.where(
                verdict_labels == 1,
                torch.tensor(self.focal_alpha, device=verdict_logits.device),
                torch.tensor(1 - self.focal_alpha, device=verdict_logits.device),
            )
            focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma
            ce_loss = F.cross_entropy(verdict_logits, verdict_labels, reduction="none")
            if sample_weights is not None:
                ce_loss = ce_loss * sample_weights
            losses["verdict"] = (focal_weight * ce_loss).mean()

        total = losses["verdict"]

        # LUPI feature head losses (standard CE, weighted)
        if law_area_labels is not None:
            mask = law_area_labels >= 0
            if mask.any():
                losses["law_area"] = F.cross_entropy(
                    outputs["law_area_logits"][mask],
                    law_area_labels[mask],
                )
                total = total + self.feature_head_weights["law_area"] * losses["law_area"]

        if error_labels is not None:
            mask = error_labels >= 0
            if mask.any():
                losses["error"] = F.cross_entropy(
                    outputs["error_logits"][mask],
                    error_labels[mask],
                )
                total = total + self.feature_head_weights["error_presence"] * losses["error"]

        if reasoning_labels is not None:
            mask = reasoning_labels >= 0
            if mask.any():
                losses["reasoning"] = F.cross_entropy(
                    outputs["reasoning_logits"][mask],
                    reasoning_labels[mask],
                )
                total = total + self.feature_head_weights["reasoning_pattern"] * losses["reasoning"]

        if outcome_labels is not None:
            mask = outcome_labels >= 0
            if mask.any():
                losses["outcome"] = F.cross_entropy(
                    outputs["outcome_logits"][mask],
                    outcome_labels[mask],
                )
                total = total + self.feature_head_weights["outcome_granular"] * losses["outcome"]

        losses["total"] = total
        return losses

    def predict_proba(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Predict verdict probabilities for texts.

        Returns array of P(approval) for each text.
        """
        import torch

        if self._model is None:
            raise RuntimeError("Model not built. Call build() first.")

        self._model.eval()
        device = next(self._model.parameters()).device
        probs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoding = self._tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = self._model(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                )
                verdict_prob = torch.softmax(outputs["verdict_logits"], dim=-1)[:, 1]
                probs.extend(verdict_prob.cpu().numpy())

        return np.array(probs)

    def predict_with_explanation(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        """Predict with full intermediate feature explanation.

        Returns list of dicts with verdict probability + predicted LUPI features.
        This IS the explanation — faithful by construction, not post-hoc.
        """
        import torch

        if self._model is None:
            raise RuntimeError("Model not built. Call build() first.")

        self._model.eval()
        device = next(self._model.parameters()).device
        results = []

        LAW_AREAS = [
            "public_law", "civil_law", "penal_law", "social_law",
            "tax_law", "insurance_law", "administrative_law", "constitutional_law",
            "family_law", "contract_law", "tort_law", "property_law",
            "criminal_procedure", "civil_procedure", "bankruptcy_law",
            "immigration_law", "other",
        ]
        REASONING = [
            "de_novo_review", "arbitrariness_review", "proportionality_test",
            "balancing_test", "subsumption", "teleological", "systematic",
            "historical", "mixed",
        ]
        OUTCOMES = [
            "full_dismissal", "full_approval", "partial_approval",
            "remand", "inadmissible", "withdrawn", "other",
        ]

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoding = self._tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = self._model(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                    return_features=True,
                )

                verdict_prob = torch.softmax(outputs["verdict_logits"], dim=-1)
                law_area_prob = torch.softmax(outputs["law_area_logits"], dim=-1)
                error_prob = torch.softmax(outputs["error_logits"], dim=-1)
                reasoning_prob = torch.softmax(outputs["reasoning_logits"], dim=-1)
                outcome_prob = torch.softmax(outputs["outcome_logits"], dim=-1)

                for j in range(len(batch)):
                    n_law = min(len(LAW_AREAS), law_area_prob.size(-1))
                    n_reas = min(len(REASONING), reasoning_prob.size(-1))
                    n_out = min(len(OUTCOMES), outcome_prob.size(-1))

                    result = {
                        "p_dismissal": float(verdict_prob[j, 0]),
                        "p_approval": float(verdict_prob[j, 1]),
                        "predicted_verdict": "approval" if verdict_prob[j, 1] > 0.5 else "dismissal",
                        "law_area": {
                            LAW_AREAS[k]: float(law_area_prob[j, k])
                            for k in range(n_law)
                        },
                        "has_decisive_error": float(error_prob[j, 1]),
                        "reasoning_pattern": {
                            REASONING[k]: float(reasoning_prob[j, k])
                            for k in range(n_reas)
                        },
                        "outcome_granular": {
                            OUTCOMES[k]: float(outcome_prob[j, k])
                            for k in range(n_out)
                        },
                    }
                    results.append(result)

        return results

    def counterfactual(
        self,
        text: str,
        perturbation: str,
        n_samples: int = 100,
    ) -> dict[str, Any]:
        """Generate counterfactual prediction via feature perturbation.

        Honest framing: this is feature perturbation, not true counterfactual
        simulation (which would require causal graph knowledge).

        Args:
            text: Original facts text.
            perturbation: Modified facts text.
            n_samples: Not used for deterministic model.

        Returns:
            Dict comparing original vs counterfactual predictions.
        """
        original = self.predict_with_explanation([text])[0]
        perturbed = self.predict_with_explanation([perturbation])[0]

        return {
            "original": original,
            "counterfactual": perturbed,
            "delta_p_approval": perturbed["p_approval"] - original["p_approval"],
            "verdict_changed": original["predicted_verdict"] != perturbed["predicted_verdict"],
        }
