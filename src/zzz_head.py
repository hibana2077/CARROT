from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZZZHead(nn.Module):
    """ZZZ: Z-Normalized Z-Subspace Zooming classifier head.

    Per-class low-rank covariance:
        Σ_c = U_c diag(v_c) U_c^T + σ^2 I

    Logits are Gaussian log-likelihood (up to constants):
        s_c(x) = -1/2 (x-μ_c)^T Σ_c^{-1} (x-μ_c) - 1/2 log|Σ_c| + b_c

    Notes:
    - Features and class means are L2-normalized (cosine-like geometry).
    - U_c is obtained by QR retraction from an unconstrained parameter A_c.
    - Uses Woodbury + determinant lemma; complexity ~ O(B*C*r).
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        rank: int = 8,
        v_min: float = 1e-4,
        sigma2_min: float = 1e-4,
    ) -> None:
        super().__init__()
        self.C = int(num_classes)
        self.d = int(feat_dim)
        self.r = int(rank)
        if self.C <= 0:
            raise ValueError("num_classes must be positive")
        if self.d <= 0:
            raise ValueError("feat_dim must be positive")
        if self.r < 0:
            raise ValueError("rank must be non-negative")

        self.v_min = float(v_min)
        self.sigma2_min = float(sigma2_min)

        self.mu = nn.Parameter(torch.randn(self.C, self.d) * 0.02)
        if self.r > 0:
            self.A = nn.Parameter(torch.randn(self.C, self.d, self.r) * 0.02)
            self.v_raw = nn.Parameter(torch.zeros(self.C, self.r))
        else:
            self.register_parameter("A", None)
            self.register_parameter("v_raw", None)

        self.sigma2_raw = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.zeros(self.C))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Compute class logits.

        feats: (B, d)
        returns: (B, C)
        """
        if feats.ndim != 2 or feats.shape[-1] != self.d:
            raise ValueError(f"Expected feats of shape (B, {self.d}), got {tuple(feats.shape)}")

        x = F.normalize(feats, dim=-1)
        mu = F.normalize(self.mu, dim=-1)

        sigma2 = F.softplus(self.sigma2_raw) + self.sigma2_min
        alpha = 1.0 / sigma2

        # ||x-mu||^2 = ||x||^2 + ||mu||^2 - 2 x^T mu. With normalization, norms are 1.
        sim = x @ mu.T  # (B, C)
        delta2 = (2.0 - 2.0 * sim).clamp_min(0.0)  # (B, C)

        if self.r == 0:
            # Degenerates to cosine-like head (up to affine scaling).
            logdet = self.d * torch.log(sigma2)  # scalar
            maha = alpha * delta2
            return -0.5 * (maha + logdet) + self.bias

        # U via QR retraction: A (C,d,r) -> U (C,d,r)
        U, _ = torch.linalg.qr(self.A, mode="reduced")

        v = F.softplus(self.v_raw) + self.v_min  # (C, r)

        # t = (x - mu)^T U = x^T U - mu^T U
        xU = torch.einsum("bd,cdr->bcr", x, U)  # (B, C, r)
        muU = torch.einsum("cd,cdr->cr", mu, U)  # (C, r)
        t = xU - muU.unsqueeze(0)  # (B, C, r)

        # Woodbury simplification for diagonal V.
        Minv = v / (1.0 + alpha * v)  # (C, r)

        maha = alpha * delta2 - (alpha * alpha) * (t * t * Minv.unsqueeze(0)).sum(dim=-1)  # (B, C)

        # log|Σ| = d log(sigma2) + sum_i log(1 + v_i / sigma2)
        logdet = (self.d * torch.log(sigma2)) + torch.log1p(alpha * v).sum(dim=-1)  # (C,)

        logits = -0.5 * (maha + logdet.unsqueeze(0)) + self.bias.unsqueeze(0)
        return logits
