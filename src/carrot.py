from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CarrotStats:
    L: Optional[float]
    U: Optional[float]
    num_pos: int
    num_neg: int
    pos_mean: Optional[float] = None
    pos_max: Optional[float] = None
    frac_pos_above_U: Optional[float] = None
    frac_pos_below_L: Optional[float] = None


class CARROT(nn.Module):
    """Class-conditional Adaptive Range Regularization with Order-statistic Thresholds.

    Operates on normalized embeddings z (cosine similarity).

    Notes:
    - Similarity matrix computed in fp32 for stability (esp. AMP).
    - Corridor thresholds L/U are detached (no gradient through quantiles).
    - Handles degenerate batches with no positive pairs.
    """

    def __init__(self, q_hi: float = 0.90, q_lo: float = 0.10, eps: float = 1e-8) -> None:
        super().__init__()
        self.q_hi = float(q_hi)
        self.q_lo = float(q_lo)
        self.eps = float(eps)

    @torch.no_grad()
    def _safe_quantile(self, x: torch.Tensor, q: float) -> Optional[torch.Tensor]:
        if x.numel() == 0:
            return None
        return torch.quantile(x, q)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, CarrotStats]:
        """Compute CARROT regularizer.

        Args:
            z: (B, D) embeddings (float/half). Will be L2-normalized inside.
            y: (B,) int labels.

        Returns:
            reg: scalar tensor
            stats: CarrotStats
        """
        B = int(z.size(0))
        device = z.device

        z = F.normalize(z, dim=1)
        sim = (z.float() @ z.float().t()).clamp(-1.0, 1.0)  # (B, B)

        y = y.view(-1, 1)
        same = y.eq(y.t())
        eye = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = same & (~eye)
        neg_mask = ~same

        pos_sims = sim[pos_mask]
        neg_sims = sim[neg_mask]

        if pos_sims.numel() == 0 or neg_sims.numel() == 0:
            reg0 = sim.new_tensor(0.0)
            return (
                reg0,
                CarrotStats(
                    L=None,
                    U=None,
                    num_pos=int(pos_sims.numel()),
                    num_neg=int(neg_sims.numel()),
                ),
            )

        q_hi = self._safe_quantile(neg_sims, self.q_hi)
        q_lo = self._safe_quantile(neg_sims, self.q_lo)
        if q_hi is None or q_lo is None:
            reg0 = sim.new_tensor(0.0)
            return (
                reg0,
                CarrotStats(
                    L=None,
                    U=None,
                    num_pos=int(pos_sims.numel()),
                    num_neg=int(neg_sims.numel()),
                ),
            )

        width = (q_hi - q_lo).clamp_min(0.0)
        L = q_hi
        U = 1.0 - width

        L = L.detach()
        U = U.detach()

        # ensure U > L; clamp to sensible range
        min_u = float((L + 1e-3).clamp_max(0.999).item())
        U = torch.clamp(U, min=min_u, max=0.999)

        low = F.relu(L - pos_sims)
        high = F.relu(pos_sims - U)
        reg = (low * low + high * high).mean()

        stats = CarrotStats(
            L=float(L.item()),
            U=float(U.item()),
            num_pos=int(pos_sims.numel()),
            num_neg=int(neg_sims.numel()),
            pos_mean=float(pos_sims.mean().item()),
            pos_max=float(pos_sims.max().item()),
            frac_pos_above_U=float((pos_sims > U).float().mean().item()),
            frac_pos_below_L=float((pos_sims < L).float().mean().item()),
        )
        return reg, stats


def grad_balanced_total_loss(
    loss_base: torch.Tensor,
    reg: torch.Tensor,
    z: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute total = loss_base + alpha * reg with alpha chosen by gradient norm balancing.

    alpha := ||d loss_base / d z||_2 / (||d reg / d z||_2 + eps)

    alpha is detached to avoid higher-order dynamics.

    Returns:
        total: scalar tensor
        alpha: scalar tensor (detached)
    """
    if not reg.requires_grad:
        alpha0 = loss_base.detach().new_tensor(0.0)
        return loss_base, alpha0

    g_base = torch.autograd.grad(loss_base, z, retain_graph=True, create_graph=False)[0]
    g_reg = torch.autograd.grad(reg, z, retain_graph=True, create_graph=False)[0]

    nb = g_base.norm(p=2)
    nr = g_reg.norm(p=2)

    # If reg gradient collapses to ~0, don't let alpha explode.
    alpha = torch.where(nr > eps, nb / (nr + eps), nb.detach().new_tensor(0.0)).detach()

    total = loss_base + alpha * reg
    return total, alpha
