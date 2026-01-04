from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CarrotStats:
    """Logging stats for CARROT operator.

    All fields are batch-level aggregates computed from in-batch class statistics.
    """

    classes_in_batch: int
    gamma_mean: Optional[float] = None
    gamma_max: Optional[float] = None
    frac_gamma_gt_1: Optional[float] = None
    r_mean: Optional[float] = None
    m_mean: Optional[float] = None


def carrot_operator(
    z: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 1e-12,
    detach_stats: bool = True,
) -> Tuple[torch.Tensor, CarrotStats]:
    """CARROT operator (parameter-free class-conditional expansion).

    Implements imp.md / idea.md core:
      mu_c = mean(z_i)
      r_c  = RMS radius
      m_c  = min_{c'!=c} ||mu_c - mu_{c'}||
      gamma_c = max(1, m_c / (2 r_c + eps))
      z_plus = mu + gamma (z - mu)

    Args:
        z: (B, D) embeddings
        y: (B,) int64 labels
        eps: numerical stability
        detach_stats: if True, stop-grad through mu/gamma (recommended)

    Returns:
        z_plus: (B, D)
        stats: CarrotStats
    """
    assert z.ndim == 2 and y.ndim == 1
    y = y.to(torch.long)
    B, D = z.shape
    device = z.device

    # Normalize for stable geometry (imp.md recommendation).
    z = F.normalize(z, dim=1)

    classes, inv = torch.unique(y, sorted=True, return_inverse=True)
    C = int(classes.numel())

    # Degenerate: batch accidentally has <2 classes.
    if C < 2:
        return z, CarrotStats(classes_in_batch=C)

    counts = torch.bincount(inv, minlength=C).clamp_min(1)  # (C,)

    mu = torch.zeros(C, D, device=device, dtype=z.dtype)
    mu.index_add_(0, inv, z)
    mu = mu / counts.unsqueeze(1)

    diff = z - mu[inv]
    sqnorm = (diff * diff).sum(dim=1)  # (B,)
    r2_sum = torch.zeros(C, device=device, dtype=z.dtype)
    r2_sum.index_add_(0, inv, sqnorm)
    r = torch.sqrt(r2_sum / counts + eps)  # (C,)

    dist_cc = torch.cdist(mu, mu, p=2)  # (C, C)
    dist_cc.fill_diagonal_(float("inf"))
    m = dist_cc.min(dim=1).values  # (C,)

    gamma = torch.clamp(m / (2.0 * r + eps), min=1.0)  # (C,)

    if detach_stats:
        mu = mu.detach()
        gamma = gamma.detach()

    z_plus = mu[inv] + gamma[inv].unsqueeze(1) * (z - mu[inv])

    gamma_mean = float(gamma.mean().item())
    gamma_max = float(gamma.max().item())
    frac_gamma_gt_1 = float((gamma > 1.0 + 1e-12).float().mean().item())
    r_mean = float(r.mean().item())
    m_mean = float(m.mean().item())

    stats = CarrotStats(
        classes_in_batch=C,
        gamma_mean=gamma_mean,
        gamma_max=gamma_max,
        frac_gamma_gt_1=frac_gamma_gt_1,
        r_mean=r_mean,
        m_mean=m_mean,
    )
    return z_plus, stats


class CARROT(nn.Module):
    """CARROT operator module.

    This module only computes the operator output `z_plus` from in-batch class statistics.
    Regularization (e.g., logit consistency KL) is computed in the training loop so it can
    reuse the task head without coupling to the model.
    """

    def __init__(self, eps: float = 1e-12, detach_stats: bool = True) -> None:
        super().__init__()
        self.eps = float(eps)
        self.detach_stats = bool(detach_stats)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, CarrotStats]:
        return carrot_operator(z, y, eps=self.eps, detach_stats=self.detach_stats)


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


def logits_grad_balanced_total_loss(
    loss_base: torch.Tensor,
    reg: torch.Tensor,
    logits: torch.Tensor,
    logits_plus: torch.Tensor,
    targets: torch.Tensor,
    *,
    T: float = 1.0,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fast gradient-norm balancing in logit space (approximation).

    The original imp.md-style balancing chooses:
      alpha := ||d loss_base / d z|| / (||d reg / d z|| + eps)

    Computing d/dz requires extra autograd.grad calls (slow). This helper instead
    balances gradients w.r.t. logits, which has a cheap closed form:
      d CE(logits,y)/d logits = (softmax(logits) - onehot(y)) / B
      d KL(q||p)/d logits     = T * (p - q) / B   (when reg is built as in main.py)

    This removes the extra backward-like passes while keeping a similar adaptive
    weighting behavior.

    Returns:
        total: scalar tensor
        alpha: scalar tensor (detached)
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape (B, C); got {tuple(logits.shape)}")
    if logits_plus.shape != logits.shape:
        raise ValueError(
            f"Expected logits_plus same shape as logits; got {tuple(logits_plus.shape)} vs {tuple(logits.shape)}"
        )

    B, C = logits.shape
    if B == 0:
        alpha0 = loss_base.detach().new_tensor(0.0)
        return loss_base, alpha0

    # Base gradient (CE mean reduction).
    p_base = F.softmax(logits, dim=1)
    onehot = F.one_hot(targets.to(torch.long), num_classes=C).to(dtype=logits.dtype)
    g_base = (p_base - onehot) / float(B)

    # Reg gradient for reg := kl_div(log_softmax(logits/T), softmax(logits_plus/T), batchmean) * T^2
    p = F.softmax(logits / float(T), dim=1)
    q = F.softmax(logits_plus / float(T), dim=1)
    g_reg = (float(T) * (p - q)) / float(B)

    nb = g_base.norm(p=2)
    nr = g_reg.norm(p=2)
    alpha = torch.where(nr > eps, nb / (nr + eps), nb.detach().new_tensor(0.0)).detach()

    total = loss_base + alpha * reg
    return total, alpha
