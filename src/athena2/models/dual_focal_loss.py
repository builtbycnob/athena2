"""Dual Focal Loss — Fallback calibration-aware training loss.

Tao et al. ICML 2023: Combines standard focal loss with a complementary
focal term that penalizes overconfident predictions on both correct AND
incorrect classes.

Used as fallback if BSCE-GRA doesn't converge.

Reference: "Dual Focal Loss for Calibration" (ICML 2023)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualFocalLoss(nn.Module):
    """Dual Focal Loss for calibrated multi-class classification.

    Combines:
    1. Standard focal loss: FL(p_t) = -alpha * (1-p_t)^gamma * log(p_t)
    2. Complement focal: penalizes incorrect class overconfidence

    Args:
        gamma: Focusing parameter (default 2.0).
        alpha: Class balance weight. None = uniform, float = binary, list = per-class.
        complement_gamma: Focusing parameter for complement term (default 1.0).
        complement_weight: Weight for complement term (default 0.5).
        reduction: Loss reduction method.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | list[float] | None = None,
        complement_gamma: float = 1.0,
        complement_weight: float = 0.5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.complement_gamma = complement_gamma
        self.complement_weight = complement_weight
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.register_buffer("alpha", torch.tensor([1 - alpha, alpha]))
            else:
                self.register_buffer("alpha", torch.tensor(alpha))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute dual focal loss.

        Args:
            logits: Shape (B, C).
            targets: Shape (B,), integer class labels.

        Returns:
            Scalar loss.
        """
        probs = F.softmax(logits, dim=-1)
        num_classes = logits.size(1)

        # Target probability
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Standard focal loss
        focal_weight = (1.0 - p_t) ** self.gamma
        ce = F.cross_entropy(logits, targets, reduction="none")

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            focal_loss = alpha_t * focal_weight * ce
        else:
            focal_loss = focal_weight * ce

        # Complement focal: penalize overconfidence on wrong classes
        # Average probability on non-target classes
        one_hot = F.one_hot(targets, num_classes).float()
        complement_probs = probs * (1 - one_hot)  # Zero out target class
        avg_complement = complement_probs.sum(dim=-1) / max(num_classes - 1, 1)

        # Complement focal term: higher loss when wrong classes have high probability
        complement_focal = avg_complement ** self.complement_gamma * (
            -torch.log(1.0 - avg_complement + 1e-10)
        )

        # Combined
        loss = focal_loss + self.complement_weight * complement_focal

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
