from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _softplus_inv(x: torch.Tensor) -> torch.Tensor:
    # inverse of softplus: x = log(1+exp(s)) -> s = log(exp(x)-1)
    # clamp to avoid numerical issues for small x
    eps = torch.finfo(x.dtype).eps
    return torch.log(torch.expm1(x).clamp_min(eps))


@dataclass
class AlphaConfig:
    mode: str  # none|fixed|learn
    entropy_reg: float = 0.0
    s_l2_reg: float = 0.0
    classwise_batch_norm: bool = True


class AlphaWeights(nn.Module):
    """Sample weights α for weighted training.

    Supports:
    - fixed: alpha is a fixed vector (no gradient)
    - learn: alpha is parameterized by s, alpha=softplus(s)

    To prevent collapse, we apply (optional) class-wise normalization *within each batch*:
      sum_{i in batch, y_i=c} alpha_i = count_{i in batch, y_i=c}
    so the average alpha per class in the batch is 1.
    """

    def __init__(
        self,
        n_train: int,
        device: torch.device,
        config: AlphaConfig,
        init_alpha: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.n_train = int(n_train)
        self.config = config

        if config.mode not in {"fixed", "learn"}:
            raise ValueError(f"Unsupported alpha mode: {config.mode}")

        if init_alpha is None:
            init_alpha = torch.ones(self.n_train, dtype=torch.float32)
        if init_alpha.numel() != self.n_train:
            raise ValueError(f"init_alpha has length {init_alpha.numel()} but n_train={self.n_train}")

        init_alpha = init_alpha.detach().float().to(device)

        if config.mode == "fixed":
            self.register_buffer("alpha_fixed", init_alpha)
            self.s = None
        else:
            s0 = _softplus_inv(init_alpha)
            self.s = nn.Parameter(s0)
            self.register_buffer("alpha_fixed", torch.empty(0))

    def has_params(self) -> bool:
        return self.s is not None

    def full_alpha_detached(self, train_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Compute globally class-normalized alpha (no grad) for logging/stats."""
        with torch.no_grad():
            if self.config.mode == "fixed":
                alpha_raw = self.alpha_fixed
            else:
                alpha_raw = F.softplus(self.s)

            y = train_labels.to(alpha_raw.device)
            counts = torch.bincount(y, minlength=num_classes).float().clamp_min(1.0)
            sums = torch.zeros(num_classes, device=alpha_raw.device, dtype=alpha_raw.dtype)
            sums.scatter_add_(0, y, alpha_raw)
            scale = counts / (sums.clamp_min(1e-12))
            alpha_norm = alpha_raw * scale.gather(0, y)
            return alpha_norm.detach().float().cpu()

    def batch_alpha(self, indices: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Return alpha for the current batch (on device), normalized per class within batch if enabled."""
        if self.config.mode == "fixed":
            alpha_raw = self.alpha_fixed.gather(0, indices)
        else:
            alpha_raw = F.softplus(self.s.gather(0, indices))

        if not self.config.classwise_batch_norm:
            return alpha_raw

        y = targets
        counts = torch.bincount(y, minlength=num_classes).float().clamp_min(1.0)
        sums = torch.zeros(num_classes, device=alpha_raw.device, dtype=alpha_raw.dtype)
        sums.scatter_add_(0, y, alpha_raw)
        scale = counts / (sums.clamp_min(1e-12))
        return alpha_raw * scale.gather(0, y)

    def regularizer(self, alpha_batch: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Regularize alpha to avoid collapse.

        - entropy_reg: maximize class-wise entropy within batch
        - s_l2_reg: L2 on s parameters (only in learn mode)
        """
        reg = torch.tensor(0.0, device=alpha_batch.device)

        if self.config.entropy_reg > 0:
            ent_total = torch.tensor(0.0, device=alpha_batch.device)
            for c in range(num_classes):
                mask = targets == c
                if mask.any():
                    a = alpha_batch[mask].clamp_min(1e-12)
                    p = a / a.sum()
                    ent = -(p * p.log()).sum()
                    ent_total = ent_total + ent
            # we want to maximize entropy, so add negative entropy
            reg = reg + (-self.config.entropy_reg) * ent_total

        if self.config.mode == "learn" and self.config.s_l2_reg > 0 and self.s is not None:
            reg = reg + self.config.s_l2_reg * (self.s.pow(2).mean())

        return reg
