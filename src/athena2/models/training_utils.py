"""Advanced training utilities for ATHENA2.

Implements bleeding-edge training techniques from 2018-2025:
- LLRD: Layer-wise Learning Rate Decay (Howard & Ruder, ULMFiT 2018)
- SAM: Sharpness-Aware Minimization (Foret et al., ICLR 2021)
- R-Drop: Regularized Dropout (Wu et al., NeurIPS 2021)
- EMA: Exponential Moving Average (Polyak & Juditsky 1992)
- SWA/SWAG: Stochastic Weight Averaging (Maddox et al., NeurIPS 2019)
- Cosine annealing with warm restarts
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


# ── LLRD: Layer-wise Learning Rate Decay ─────────────────────────

def get_llrd_param_groups(
    model: nn.Module,
    base_lr: float = 1e-5,
    decay_factor: float = 0.95,
    weight_decay: float = 0.01,
    no_decay_params: tuple[str, ...] = ("bias", "LayerNorm.weight", "layer_norm.weight"),
) -> list[dict[str, Any]]:
    """Create parameter groups with layer-wise learning rate decay.

    Lower (earlier) transformer layers get smaller learning rates.
    The intuition: lower layers capture general linguistic features,
    higher layers capture task-specific features.

    Args:
        model: Model with .encoder attribute (transformer backbone).
        base_lr: Learning rate for the top layer.
        decay_factor: Multiplicative decay per layer (0.95 = each lower layer gets 5% less LR).
        weight_decay: Weight decay for parameters not in no_decay set.
        no_decay_params: Parameter name patterns that should not have weight decay.

    Returns:
        List of parameter groups for optimizer.
    """
    param_groups = []
    named_params = list(model.named_parameters())

    # Detect number of encoder layers
    n_layers = 0
    for name, _ in named_params:
        if "encoder.layer." in name or "encoder.layers." in name:
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part in ("layer", "layers") and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        n_layers = max(n_layers, layer_idx + 1)
                    except ValueError:
                        pass

    if n_layers == 0:
        # Fallback: no layer structure detected, use uniform LR
        logger.warning("No layer structure detected, using uniform LR")
        return [{"params": [p for _, p in named_params if p.requires_grad], "lr": base_lr}]

    logger.info(f"LLRD: {n_layers} encoder layers, decay={decay_factor}, base_lr={base_lr}")

    # Group parameters by layer depth
    # embeddings → layer 0 → ... → layer N-1 → classifier heads
    layer_lrs = {}
    for i in range(n_layers):
        layer_lrs[i] = base_lr * (decay_factor ** (n_layers - 1 - i))

    for name, param in named_params:
        if not param.requires_grad:
            continue

        # Determine learning rate based on layer
        lr = base_lr  # default for head parameters
        if "embeddings" in name:
            lr = base_lr * (decay_factor ** n_layers)  # Lowest LR
        elif "encoder.layer." in name or "encoder.layers." in name:
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part in ("layer", "layers") and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        lr = layer_lrs[layer_idx]
                    except (ValueError, KeyError):
                        pass
                    break

        # Weight decay grouping
        wd = weight_decay
        if any(nd in name for nd in no_decay_params):
            wd = 0.0

        param_groups.append({
            "params": [param],
            "lr": lr,
            "weight_decay": wd,
            "name": name,
        })

    return param_groups


# ── SAM: Sharpness-Aware Minimization ────────────────────────────

class SAM(Optimizer):
    """Sharpness-Aware Minimization optimizer.

    Seeks parameters that lie in neighborhoods with uniformly low loss,
    leading to better generalization.

    Args:
        base_optimizer: The underlying optimizer (e.g., AdamW).
        rho: Perturbation radius (default 0.05).
        adaptive: Use adaptive SAM (per-parameter scaling).
    """

    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, adaptive: bool = False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self):
        """Compute epsilon (perturbation) and apply to weights."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if group.get("adaptive") else 1.0) * p.grad * scale
                p.add_(e_w)  # Climb to the local maximum
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        """Revert perturbation and apply base optimizer step."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # Revert perturbation
        self.base_optimizer.step()

    @torch.no_grad()
    def step(self, closure=None):
        """Not used directly — call first_step() and second_step() manually."""
        raise NotImplementedError("Use first_step() and second_step()")

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group.get("adaptive") else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none)


# ── R-Drop: Regularized Dropout ─────────────────────────────────

def rdrop_loss(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    loss_fn: nn.Module | None = None,
) -> torch.Tensor:
    """R-Drop: consistency regularization between two forward passes.

    Runs the input through the model twice (with different dropout masks),
    then minimizes KL divergence between the two output distributions.

    Args:
        logits1: First forward pass logits, shape (B, C).
        logits2: Second forward pass logits, shape (B, C).
        targets: Class labels, shape (B,).
        alpha: Weight for KL divergence term.
        loss_fn: Task loss function (default: CrossEntropyLoss).

    Returns:
        Combined loss = CE(logits1) + CE(logits2) + alpha * KL_sym(p1, p2)
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    # Standard task losses from both passes
    ce1 = loss_fn(logits1, targets)
    ce2 = loss_fn(logits2, targets)
    ce_loss = (ce1 + ce2) / 2.0

    # Symmetric KL divergence between the two distributions
    p1 = F.log_softmax(logits1, dim=-1)
    p2 = F.log_softmax(logits2, dim=-1)

    kl_12 = F.kl_div(p1, p2.exp(), reduction="batchmean")
    kl_21 = F.kl_div(p2, p1.exp(), reduction="batchmean")
    kl_loss = (kl_12 + kl_21) / 2.0

    return ce_loss + alpha * kl_loss


# ── EMA: Exponential Moving Average ──────────────────────────────

class EMA:
    """Exponential Moving Average of model parameters.

    Maintains shadow weights that are a running average of model weights.
    Use EMA weights for evaluation (smoother, better generalization).

    Args:
        model: The model to track.
        decay: EMA decay rate (default 0.999). Higher = smoother.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register(model)

    def _register(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights with current model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        """Replace model weights with EMA shadow weights (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore original model weights (after evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ── SWA / SWAG ───────────────────────────────────────────────────

class SWACollector:
    """Stochastic Weight Averaging collector.

    Averages model weights over training epochs for better generalization.
    Start collecting after the model has converged (e.g., last 20% of training).

    Args:
        model: Model to collect weights from.
    """

    def __init__(self, model: nn.Module):
        self.n_models = 0
        self.swa_state: dict[str, torch.Tensor] = {}
        self._init_from_model(model)

    def _init_from_model(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.swa_state[name] = torch.zeros_like(param.data)

    @torch.no_grad()
    def collect(self, model: nn.Module) -> None:
        """Collect current model weights into running average."""
        self.n_models += 1
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.swa_state:
                self.swa_state[name].add_(param.data)

    def apply_swa(self, model: nn.Module) -> None:
        """Apply averaged weights to model."""
        if self.n_models == 0:
            return
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.swa_state:
                param.data.copy_(self.swa_state[name] / self.n_models)
        logger.info(f"SWA applied: averaged {self.n_models} checkpoints")


class SWAGCollector(SWACollector):
    """SWAG: SWA with Gaussian approximation for uncertainty.

    Extends SWA with second-moment statistics to approximate the posterior
    distribution over weights. Enables Bayesian-like uncertainty estimation.

    Args:
        model: Model to collect weights from.
        max_rank: Maximum rank for low-rank covariance approximation (default 20).
    """

    def __init__(self, model: nn.Module, max_rank: int = 20):
        super().__init__(model)
        self.max_rank = max_rank
        self.sq_state: dict[str, torch.Tensor] = {}
        self.deviations: list[dict[str, torch.Tensor]] = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.sq_state[name] = torch.zeros_like(param.data)

    @torch.no_grad()
    def collect(self, model: nn.Module) -> None:
        """Collect weights, squared weights, and deviation vectors."""
        super().collect(model)

        # Second moment for diagonal variance
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.sq_state:
                self.sq_state[name].add_(param.data ** 2)

        # Low-rank deviation
        if len(self.deviations) < self.max_rank:
            deviation = {}
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.swa_state:
                    mean = self.swa_state[name] / self.n_models
                    deviation[name] = param.data - mean
            self.deviations.append(deviation)

    def sample_model(self, model: nn.Module, scale: float = 0.5) -> None:
        """Sample a model from the SWAG posterior approximation.

        Args:
            model: Model to apply sampled weights to.
            scale: Scaling factor for the perturbation (default 0.5).
        """
        if self.n_models < 2:
            self.apply_swa(model)
            return

        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.swa_state:
                continue

            mean = self.swa_state[name] / self.n_models
            sq_mean = self.sq_state[name] / self.n_models
            var = torch.clamp(sq_mean - mean ** 2, min=1e-10)

            # Diagonal perturbation
            z1 = torch.randn_like(param.data)
            diag_sample = mean + scale * torch.sqrt(var) * z1

            # Low-rank perturbation
            if self.deviations:
                z2 = torch.randn(len(self.deviations), device=param.device)
                lr_sample = sum(
                    z * dev[name] for z, dev in zip(z2, self.deviations) if name in dev
                ) / math.sqrt(2.0 * (len(self.deviations) - 1) + 1e-10)
                param.data.copy_(diag_sample + scale * lr_sample)
            else:
                param.data.copy_(diag_sample)


# ── Cosine Annealing with Warm Restarts ──────────────────────────

class CosineAnnealingWarmRestartsWithWarmup(_LRScheduler):
    """Cosine annealing with warm restarts and linear warmup.

    Args:
        optimizer: Wrapped optimizer.
        T_0: Initial restart period (in epochs).
        T_mult: Multiplicative factor for period after each restart.
        eta_min: Minimum learning rate.
        warmup_steps: Number of warmup steps (linear ramp).
        last_epoch: Last epoch index.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 2,
        eta_min: float = 1e-7,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.T_cur = 0
        self.T_i = T_0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(self.warmup_steps, 1)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # Cosine annealing with restarts
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        if epoch >= self.warmup_steps:
            adjusted = epoch - self.warmup_steps
            if adjusted >= self.T_i:
                self.T_cur = 0
                self.T_i = self.T_i * self.T_mult
            else:
                self.T_cur = adjusted

        super().step(epoch)
