"""FAMO: Fast Adaptive Multitask Optimization.

Automatic loss weight balancing for multi-task learning. Replaces manual
loss weight tuning with adaptive gradient-based optimization.

Key idea: Maintain a set of task weights that are updated online based on
each task's loss trajectory. Tasks that are falling behind get more weight.

Reference: Liu et al., "FAMO: Fast Adaptive Multitask Optimization" (2024)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FAMO(nn.Module):
    """Fast Adaptive Multitask Optimization.

    Maintains learnable log-scale task weights that balance multiple losses
    adaptively during training.

    Args:
        n_tasks: Number of tasks to balance.
        gamma: Momentum for loss history smoothing (default 0.01).
        min_weight: Minimum task weight to prevent collapse (default 0.01).
        device: Device for weight tensors.
    """

    def __init__(
        self,
        n_tasks: int,
        gamma: float = 0.01,
        min_weight: float = 0.01,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.gamma = gamma
        self.min_weight = min_weight

        # Learnable log-weights (initialized to equal weighting)
        self.log_weights = nn.Parameter(torch.zeros(n_tasks, device=device))

        # Running loss history for each task
        self.register_buffer(
            "loss_history",
            torch.zeros(n_tasks, device=device),
        )
        self.register_buffer(
            "loss_prev",
            torch.zeros(n_tasks, device=device),
        )
        self._initialized = False

    @property
    def weights(self) -> torch.Tensor:
        """Get normalized task weights (sum to n_tasks for scale invariance)."""
        raw = torch.softmax(self.log_weights, dim=0) * self.n_tasks
        return torch.clamp(raw, min=self.min_weight)

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted multi-task loss.

        Args:
            losses: Dict mapping task name → scalar loss tensor.
                    Task order must be consistent across calls.

        Returns:
            Weighted total loss.
        """
        loss_tensors = list(losses.values())
        if len(loss_tensors) != self.n_tasks:
            raise ValueError(
                f"Expected {self.n_tasks} tasks, got {len(loss_tensors)}"
            )

        weights = self.weights
        total = sum(w * l for w, l in zip(weights, loss_tensors))
        return total

    @torch.no_grad()
    def update_weights(self, losses: dict[str, torch.Tensor]) -> dict[str, float]:
        """Update task weights based on loss trajectory.

        Call this AFTER the backward pass each step.

        Args:
            losses: Current per-task losses (detached).

        Returns:
            Dict of current task weights for logging.
        """
        loss_values = torch.stack([l.detach() for l in losses.values()])

        if not self._initialized:
            self.loss_prev.copy_(loss_values)
            self._initialized = True
            return {name: float(w) for name, w in zip(losses.keys(), self.weights)}

        # Loss rate of change (higher = task is struggling)
        loss_delta = loss_values - self.loss_prev

        # Exponential moving average of loss deltas
        self.loss_history.mul_(1 - self.gamma).add_(loss_delta, alpha=self.gamma)

        # Update log_weights: increase weight for tasks with increasing loss
        # Tasks that are improving (negative delta) get less weight
        self.log_weights.data.add_(self.loss_history, alpha=1.0)

        # Update previous losses
        self.loss_prev.copy_(loss_values)

        return {name: float(w) for name, w in zip(losses.keys(), self.weights)}


class FAMOOptimizer:
    """Convenience wrapper that manages FAMO + model optimizer together.

    Usage:
        famo_opt = FAMOOptimizer(model, n_tasks=5, lr=1e-5, famo_lr=1e-3)
        for batch in dataloader:
            losses = model.compute_losses(batch)  # dict of task losses
            total_loss = famo_opt.step(losses)
    """

    def __init__(
        self,
        model: nn.Module,
        n_tasks: int,
        lr: float = 1e-5,
        famo_lr: float = 1e-3,
        weight_decay: float = 0.01,
        gamma: float = 0.01,
        device: str = "cpu",
    ):
        self.famo = FAMO(n_tasks=n_tasks, gamma=gamma, device=device)

        # Separate optimizers for model and FAMO weights
        self.model_optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.famo_optimizer = torch.optim.Adam(
            self.famo.parameters(), lr=famo_lr,
        )

    def step(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute weighted loss, backward, and update both model and FAMO.

        Args:
            losses: Dict of per-task losses.

        Returns:
            (total_loss, task_weights_dict)
        """
        # Forward through FAMO
        total_loss = self.famo(losses)

        # Backward
        self.model_optimizer.zero_grad()
        self.famo_optimizer.zero_grad()
        total_loss.backward()

        # Update model
        self.model_optimizer.step()

        # Update FAMO weights
        self.famo_optimizer.step()
        task_weights = self.famo.update_weights(losses)

        return total_loss.detach(), task_weights
