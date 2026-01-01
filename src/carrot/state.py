from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CarrotHyperParams:
    alpha: float = 10.0
    topm: int = 20
    conf_topk: int = 0
    use_mahalanobis: bool = False


class CarrotState(nn.Module):
    """CARROT state (EMA prototypes + EMA soft confusion) and differentiable loss.

    Notes on gradients:
    - We update `mu/conf` under `torch.no_grad()` (EMA statistics).
    - For the loss used in training, we compute *batch prototypes* from current
      features so the loss is differentiable w.r.t. the backbone.
    - Confusion weights (omega) come from EMA `conf`.
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        device: Optional[torch.device] = None,
        mu_momentum: float = 0.05,
        conf_momentum: float = 0.05,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        K, d = int(num_classes), int(feat_dim)
        self.K = K
        self.d = d
        self.mu_m = float(mu_momentum)
        self.conf_m = float(conf_momentum)
        self.eps = float(eps)

        self.register_buffer("mu", torch.zeros(K, d, device=device))
        self.register_buffer("mu_count", torch.zeros(K, device=device))
        self.register_buffer("conf", torch.zeros(K, K, device=device))

        # Optional: Mahalanobis (kept for future extension; MVP uses Euclidean)
        self.register_buffer("inv_cov", torch.eye(d, device=device))

    @torch.no_grad()
    def update_mu(self, feats: torch.Tensor, y: torch.Tensor) -> None:
        """EMA update for class prototypes.

        feats: [B, d] (recommended: already L2-normalized)
        y: [B]
        """
        K = self.K
        B, d = feats.shape
        if d != self.d:
            raise ValueError(f"Feature dim mismatch: got {d}, expected {self.d}")

        sums = torch.zeros(K, d, device=feats.device, dtype=feats.dtype)
        cnts = torch.zeros(K, device=feats.device, dtype=feats.dtype)

        sums.index_add_(0, y, feats)
        cnts.index_add_(0, y, torch.ones(B, device=feats.device, dtype=feats.dtype))

        mask = cnts > 0
        if not torch.any(mask):
            return

        batch_mu = torch.zeros_like(self.mu)
        batch_mu[mask] = sums[mask] / cnts[mask].unsqueeze(1)

        m = self.mu_m
        self.mu[mask] = (1.0 - m) * self.mu[mask] + m * batch_mu[mask]
        self.mu_count[mask] += cnts[mask]

        # Keep prototypes normalized (in-place to preserve buffer registration)
        self.mu.copy_(F.normalize(self.mu, dim=-1))

    @torch.no_grad()
    def update_conf(self, logits: torch.Tensor, y: torch.Tensor, topk: int = 0) -> None:
        """EMA update for soft confusion.

        logits: [B, K]
        y: [B]
        topk=0 uses full probs; >0 uses only top-k probs per sample (faster).
        """
        K = self.K
        if logits.ndim != 2 or logits.size(1) != K:
            raise ValueError(f"Logits shape must be [B, {K}], got {tuple(logits.shape)}")

        p = F.softmax(logits, dim=-1)  # [B, K]
        B = logits.size(0)
        delta = torch.zeros(K, K, device=logits.device, dtype=logits.dtype)

        if topk and topk < K:
            vals, idx = torch.topk(p, k=int(topk), dim=-1)  # [B, topk]
            row_idx = y.view(-1, 1).expand_as(idx).reshape(-1)
            col_idx = idx.reshape(-1)
            v = vals.reshape(-1)
            delta.index_put_((row_idx, col_idx), v, accumulate=True)
        else:
            delta.index_add_(0, y, p)

        # Normalize by count per class (stabilize scale)
        cnts = torch.bincount(y, minlength=K).to(dtype=logits.dtype, device=logits.device)
        cnts = torch.clamp(cnts, min=1.0)
        delta = delta / cnts.unsqueeze(1)

        m = self.conf_m
        self.conf.mul_(1.0 - m).add_(delta, alpha=m)
        self.conf.fill_diagonal_(0.0)

    def _omega(self) -> torch.Tensor:
        omega = 0.5 * (self.conf + self.conf.t())
        omega.fill_diagonal_(0.0)
        return omega

    def carrot_loss_from_batch(
        self,
        feats: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: float = 10.0,
        topm: int = 20,
        use_mahalanobis: bool = False,
    ) -> torch.Tensor:
        """Compute differentiable CARROT loss using *batch prototypes*.

        - Confusion weights come from EMA `conf`.
        - Prototypes are computed from current batch features (keeps gradient).
        """
        if feats.ndim != 2:
            raise ValueError(f"feats must be [B, d], got {tuple(feats.shape)}")

        B, d = feats.shape
        if d != self.d:
            raise ValueError(f"Feature dim mismatch: got {d}, expected {self.d}")

        classes, inv = torch.unique(y, sorted=True, return_inverse=True)
        C = int(classes.numel())
        if C < 2:
            return feats.new_tensor(0.0)

        sums = torch.zeros(C, d, device=feats.device, dtype=feats.dtype)
        cnts = torch.zeros(C, device=feats.device, dtype=feats.dtype)
        sums.index_add_(0, inv, feats)
        cnts.index_add_(0, inv, torch.ones(B, device=feats.device, dtype=feats.dtype))
        cnts = torch.clamp(cnts, min=1.0)

        proto = sums / cnts.unsqueeze(1)
        proto = F.normalize(proto, dim=-1)

        omega = self._omega()  # [K, K]
        omega_sub = omega.index_select(0, classes).index_select(1, classes)  # [C, C]
        omega_sub.fill_diagonal_(0.0)

        # Build pairs (i<j) for top-m neighbors within this batch's classes
        if topm >= C - 1:
            ii, jj = torch.triu_indices(C, C, offset=1, device=feats.device)
        else:
            pairs_i = []
            pairs_j = []
            k = int(min(max(topm, 1), C - 1))
            for i in range(C):
                row = omega_sub[i]
                js = torch.topk(row, k=k, largest=True).indices
                for j in js.tolist():
                    if i < j:
                        pairs_i.append(i)
                        pairs_j.append(j)
            if len(pairs_i) == 0:
                return feats.new_tensor(0.0)
            ii = torch.tensor(pairs_i, device=feats.device, dtype=torch.long)
            jj = torch.tensor(pairs_j, device=feats.device, dtype=torch.long)

        diff = proto[ii] - proto[jj]  # [P, d]
        if use_mahalanobis:
            inv_cov = self.inv_cov
            d2 = torch.einsum("pd,dd,pd->p", diff, inv_cov, diff)
        else:
            d2 = (diff * diff).sum(dim=-1)

        w = omega_sub[ii, jj]
        loss = (w * torch.exp(-float(alpha) * d2)).mean()
        return loss
