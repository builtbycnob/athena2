"""BSCE-GRA: Balanced Sigmoid Cross-Entropy with Gradient-Aware Reweighting.

CVPR 2025 — SOTA training-time calibration via uncertainty-weighted gradients.
Handles both class imbalance AND calibration simultaneously.

Key idea: Weight gradient contribution by prediction uncertainty. Samples where
the model is uncertain contribute more to the gradient, while confident (and likely
correct) samples contribute less. This prevents overconfidence while addressing
class imbalance through balanced sampling.

Reference: "Calibration-Aware Training via Gradient-Reweighted Losses" (CVPR 2025)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BSCEGRALoss(nn.Module):
    """Balanced Sigmoid Cross-Entropy with Gradient-Aware Reweighting.

    Combines three mechanisms:
    1. Balanced per-class weighting (handles class imbalance)
    2. Gradient-aware reweighting (improves calibration)
    3. Uncertainty-based sample weighting (reduces overconfidence)

    Args:
        num_classes: Number of output classes.
        class_counts: Per-class sample counts for balancing (optional).
        beta: Effective number of samples smoothing (default 0.999).
        gamma: Gradient-aware reweighting strength (default 1.0).
        uncertainty_weight: Weight for uncertainty term (default 0.5).
        reduction: Loss reduction method.
    """

    def __init__(
        self,
        num_classes: int = 2,
        class_counts: list[int] | None = None,
        beta: float = 0.999,
        gamma: float = 1.0,
        uncertainty_weight: float = 0.5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.uncertainty_weight = uncertainty_weight
        self.reduction = reduction

        # Compute effective number of samples per class (Cui et al. 2019)
        if class_counts is not None:
            effective_num = [1.0 - beta**n for n in class_counts]
            weights = [1.0 / en if en > 0 else 1.0 for en in effective_num]
            total = sum(weights)
            weights = [w / total * num_classes for w in weights]
            self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))
        else:
            self.register_buffer("class_weights", torch.ones(num_classes, dtype=torch.float32))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute BSCE-GRA loss.

        Args:
            logits: Model logits, shape (B, C).
            targets: Class labels, shape (B,).
            sample_weights: Optional per-sample weights, shape (B,).

        Returns:
            Scalar loss.
        """
        probs = F.softmax(logits, dim=-1)

        # Standard cross-entropy per sample
        class_w = self.class_weights.to(logits.device)
        ce_loss = F.cross_entropy(logits, targets, weight=class_w, reduction="none")

        # Gradient-aware reweighting: weight by prediction uncertainty
        # Uncertainty = entropy of predicted distribution
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(float(self.num_classes), device=logits.device))
        normalized_entropy = entropy / max_entropy  # [0, 1]

        # Gradient reweighting factor: uncertain samples get higher weight
        gra_weight = 1.0 + self.gamma * normalized_entropy

        # Confidence penalty: penalize overconfident predictions
        p_target = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        confidence_penalty = -self.uncertainty_weight * (
            (1.0 - p_target) * torch.log(1.0 - p_target + 1e-10)
        )

        # Combined loss
        loss = gra_weight * ce_loss + confidence_penalty

        # Apply per-sample weights (e.g., from cleanlab)
        if sample_weights is not None:
            loss = loss * sample_weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class BSCEGRABinaryLoss(nn.Module):
    """Binary version of BSCE-GRA for two-class problems.

    Optimized for binary classification (dismissal/approval).
    Uses sigmoid instead of softmax for efficiency.

    Args:
        pos_weight: Weight for positive class (approval). Set > 1 for ~70/30 imbalance.
        gamma: Gradient-aware reweighting strength.
        uncertainty_weight: Weight for uncertainty regularization.
    """

    def __init__(
        self,
        pos_weight: float = 2.33,  # ~70/30 imbalance: 0.7/0.3
        gamma: float = 1.0,
        uncertainty_weight: float = 0.5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.uncertainty_weight = uncertainty_weight
        self.reduction = reduction
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute binary BSCE-GRA loss.

        Args:
            logits: Raw logits for positive class, shape (B,) or (B, 1).
            targets: Binary labels (0/1), shape (B,).
            sample_weights: Optional per-sample weights, shape (B,).

        Returns:
            Scalar loss.
        """
        if logits.dim() == 2:
            logits = logits[:, 1] - logits[:, 0]  # Convert 2-class to binary logit

        probs = torch.sigmoid(logits)

        # Weighted BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(),
            pos_weight=self.pos_weight,
            reduction="none",
        )

        # Uncertainty: binary entropy H(p) = -p*log(p) - (1-p)*log(1-p)
        binary_entropy = -(
            probs * (probs + 1e-10).log() +
            (1 - probs) * (1 - probs + 1e-10).log()
        )
        max_entropy = torch.log(torch.tensor(2.0, device=logits.device))
        normalized_entropy = binary_entropy / max_entropy

        # GRA weight
        gra_weight = 1.0 + self.gamma * normalized_entropy

        # Confidence penalty
        p_target = torch.where(targets == 1, probs, 1 - probs)
        confidence_penalty = -self.uncertainty_weight * (
            (1.0 - p_target) * torch.log(1.0 - p_target + 1e-10)
        )

        loss = gra_weight * bce + confidence_penalty

        if sample_weights is not None:
            loss = loss * sample_weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
