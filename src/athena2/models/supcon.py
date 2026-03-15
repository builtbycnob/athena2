"""Supervised Contrastive Learning for legal judgment prediction.

Pre-training phase that learns discriminative embeddings before fine-tuning.
Pulls same-class embeddings together, pushes different-class embeddings apart.

Reference: Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    For each anchor, contrasts against all positives (same class) and
    negatives (different class) in the batch.

    Args:
        temperature: Scaling parameter for similarity (default 0.07).
        base_temperature: Base temperature for normalization (default 0.07).
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute SupCon loss.

        Args:
            features: Normalized embeddings, shape (B, D).
            labels: Class labels, shape (B,).

        Returns:
            Scalar loss.
        """
        device = features.device
        batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature

        # Mask: positive pairs (same label, excluding self)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        # Remove self-contrast
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        mask = mask * (~self_mask).float()

        # For numerical stability, subtract max
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # Mask out self from denominator
        exp_logits = torch.exp(logits) * (~self_mask).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-10)

        # Mean of log-likelihood over positive pairs
        n_positives = mask.sum(dim=1)
        # Avoid division by zero for samples with no positives in batch
        n_positives = torch.clamp(n_positives, min=1.0)
        mean_log_prob = (mask * log_prob).sum(dim=1) / n_positives

        # Loss
        loss = -(self.base_temperature / self.temperature) * mean_log_prob
        return loss.mean()


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning.

    Maps encoder CLS output to a lower-dimensional space for contrastive loss.
    Used only during SupCon pre-training, discarded for fine-tuning.

    Args:
        input_dim: Encoder hidden size (e.g., 1024 for RoBERTa-Large).
        hidden_dim: Intermediate projection dimension.
        output_dim: Final embedding dimension for contrastive loss.
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize."""
        return F.normalize(self.net(x), dim=1)


class SupConPreTrainer:
    """Manages supervised contrastive pre-training phase.

    Usage:
        pretrainer = SupConPreTrainer(encoder, hidden_size=1024)
        pretrainer.train(train_dataloader, epochs=2, lr=5e-5)
        # Discard projection head, use encoder for fine-tuning
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int = 1024,
        projection_dim: int = 128,
        temperature: float = 0.07,
        device: str = "mps",
    ):
        self.encoder = encoder
        self.projection = ProjectionHead(hidden_size, hidden_size // 4, projection_dim).to(device)
        self.criterion = SupConLoss(temperature=temperature)
        self.device = device

    def train_epoch(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Train one epoch of SupCon.

        Args:
            dataloader: Yields input_ids, attention_mask, labels.
            optimizer: Optimizer for encoder + projection head.

        Returns:
            Average loss for the epoch.
        """
        self.encoder.train()
        self.projection.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Encode
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]

            # Project
            projected = self.projection(cls_output)

            # Contrastive loss
            loss = self.criterion(projected, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)
