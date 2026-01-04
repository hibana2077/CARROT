from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DSOTConfig:
    k: int = 16
    eps: float = 0.10
    sinkhorn_iters: int = 20
    lambda_pos: float = 0.10
    self_loop_alpha: float = 0.20
    cost_normalize: bool = True


class DSOTGraphBuilder(nn.Module):
    """DSOT / Sinkhorn-based graph builder.

    Input:
      x: (B,N,D) node features
      pos: (N,2) positions in [0,1], row-major (y,x)

    Output:
      edge_index: (2, B*N*k)
      edge_weight: (B*N*k,)
      batch: (B*N,) graph id per node
    """

    def __init__(
        self,
        *,
        k: int = 16,
        eps: float = 0.10,
        sinkhorn_iters: int = 20,
        lambda_pos: float = 0.10,
        self_loop_alpha: float = 0.20,
        cost_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.eps = float(eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.lambda_pos = float(lambda_pos)
        self.self_loop_alpha = float(self_loop_alpha)
        self.cost_normalize = bool(cost_normalize)

    @staticmethod
    def make_grid_pos(H: int, W: int, device=None, dtype=torch.float32) -> torch.Tensor:
        ys = torch.linspace(0.0, 1.0, steps=int(H), device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, steps=int(W), device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)  # (N,2)
        return pos

    @staticmethod
    def pairwise_sqdist_pos(pos: torch.Tensor) -> torch.Tensor:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        a2 = (pos * pos).sum(dim=-1, keepdim=True)  # (N,1)
        dist2 = a2 + a2.transpose(0, 1) - 2.0 * (pos @ pos.transpose(0, 1))
        return torch.clamp(dist2, min=0.0)

    @staticmethod
    def log_sinkhorn_uniform(logK: torch.Tensor, iters: int) -> torch.Tensor:
        """Uniform-marginal log-domain Sinkhorn.

        logK: (B,N,N) where K = exp(logK)
        returns P: (B,N,N)
        """

        B, N, _ = logK.shape
        log_r = -math.log(N)
        log_c = -math.log(N)

        logu = torch.zeros((B, N), device=logK.device, dtype=logK.dtype)
        logv = torch.zeros((B, N), device=logK.device, dtype=logK.dtype)

        for _ in range(int(iters)):
            logu = log_r - torch.logsumexp(logK + logv[:, None, :], dim=-1)
            logv = log_c - torch.logsumexp(logK.transpose(1, 2) + logu[:, None, :], dim=-1)

        logP = logu[:, :, None] + logK + logv[:, None, :]
        return torch.exp(logP)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        device = x.device

        # 1) feature cost: 1 - cosine
        x = F.normalize(x, p=2, dim=-1)
        sim = torch.matmul(x, x.transpose(1, 2))  # (B,N,N)
        cost_feat = 1.0 - sim

        # 2) position cost
        pos_dist2 = self.pairwise_sqdist_pos(pos).to(device=device, dtype=cost_feat.dtype)
        cost = cost_feat + self.lambda_pos * pos_dist2[None, :, :]

        if self.cost_normalize:
            cost = cost / (cost.mean(dim=(1, 2), keepdim=True) + 1e-8)

        # 3) Sinkhorn in log-domain; keep fp32 for stability
        logK = -cost / self.eps
        P = self.log_sinkhorn_uniform(logK.float(), self.sinkhorn_iters).to(dtype=cost.dtype)

        # 4) symmetrize adjacency
        A = 0.5 * (P + P.transpose(1, 2))

        # 5) self-loop bias
        if self.self_loop_alpha > 0:
            eye = torch.eye(N, device=device, dtype=A.dtype)[None, :, :]
            A = A + self.self_loop_alpha * eye

        # 6) top-k per row
        k = min(self.k, N)
        vals, idx = torch.topk(A, k=k, dim=-1)  # (B,N,k)

        # row-normalize
        vals = vals / (vals.sum(dim=-1, keepdim=True) + 1e-12)

        offsets = (torch.arange(B, device=device) * N)[:, None, None]
        src = torch.arange(N, device=device)[None, :, None].expand(B, N, k) + offsets
        dst = idx + offsets

        edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)
        edge_weight = vals.reshape(-1)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        return edge_index, edge_weight, batch
