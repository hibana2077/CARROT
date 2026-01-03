import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class CARROTFeatureAug(nn.Module):
    """CARROT feature-space augmentation (sample-level, top-K confusing classes).

    Implementation matches the description in docs/imp.md:
    - Build discriminative subspace from classifier weight differences (w_y - w_k).
    - Inject Gaussian noise only in nuisance subspace (orthogonal complement).
    - Choose noise scale alpha via a conservative margin chance-constraint bound.

    Notes:
    - This implementation is intentionally simple (loop over samples) to stay faithful to the spec.
    - If the discriminative subspace is degenerate for a sample, the sample is skipped.
    """

    def __init__(
        self,
        topk: int = 10,
        gamma: float = 0.0,
        delta: float = 0.05,
        aug_per_sample: int = 1,
        aug_weight: float = 1.0,
        eps: float = 1e-8,
        qr_rank_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.topk = int(topk)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.aug_per_sample = int(aug_per_sample)
        self.aug_weight = float(aug_weight)
        self.eps = float(eps)
        self.qr_rank_eps = float(qr_rank_eps)

    @torch.no_grad()
    def _build_D_basis(self, w_y: torch.Tensor, w_ks: torch.Tensor) -> Optional[torch.Tensor]:
        """Return an orthonormal basis for the discriminative subspace.

        w_y: (D,)
        w_ks: (K, D)
        returns: Q[:, keep] shape (D, r) or None if degenerate
        """
        # A = [w_y - w_k]  -> (K, D)
        A = (w_y.unsqueeze(0) - w_ks)  # (K, D)
        # QR on A^T to get basis for span(A^T)
        Q, R = torch.linalg.qr(A.T, mode="reduced")  # Q: (D, r)

        if R.ndim != 2 or R.numel() == 0:
            return None

        diag = torch.abs(torch.diag(R))
        keep = diag > self.qr_rank_eps
        if keep.numel() == 0 or int(keep.sum().item()) == 0:
            return None
        return Q[:, keep]

    def forward(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        logits: torch.Tensor,
        W: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Generate augmented feature samples.

        z: (B, D) embedding (typically post-norm)
        y: (B,) labels
        logits: (B, C) logits (used only to pick top-K confusing classes)
        W: (C, D) classifier weights
        """
        device = z.device
        B, D = z.shape
        C = W.shape[0]
        if C <= 1 or self.aug_per_sample <= 0:
            return None, None

        # top-K confusing classes per sample (exclude true class)
        with torch.no_grad():
            logits_masked = logits.clone()
            logits_masked[torch.arange(B, device=device), y] = -1e9
            k = min(self.topk, C - 1)
            topk_idx = torch.topk(logits_masked, k=k, dim=1).indices  # (B, K)

        z_out = []
        y_out = []

        # log(1/delta)
        log_term = math.log(1.0 / max(self.delta, 1e-12))

        for i in range(B):
            yi = int(y[i].item())
            w_y = W[yi]  # (D,)
            w_ks = W[topk_idx[i]]  # (K, D)

            with torch.no_grad():
                D_basis = self._build_D_basis(w_y.detach(), w_ks.detach())
            if D_basis is None:
                continue

            # nuisance projection: proj_N(v) = v - D (D^T v)
            def proj_N(v: torch.Tensor) -> torch.Tensor:
                return v - D_basis @ (D_basis.T @ v)

            with torch.no_grad():
                zi = z[i].detach()
                diffs = (w_y.unsqueeze(0) - w_ks)  # (K, D)
                margins = diffs @ zi  # (K,)

                # ||proj_N(w_y - w_k)||^2
                diffs_N = torch.stack([proj_N(diffs[j]) for j in range(diffs.shape[0])], dim=0)
                pn2 = diffs_N.pow(2).sum(dim=1).clamp_min(self.eps)  # (K,)

                slack = (margins - self.gamma).clamp_min(0.0)
                alpha_k = (slack * slack) / (2.0 * pn2 * log_term + self.eps)
                alpha = float(torch.min(alpha_k).item())

            if not math.isfinite(alpha) or alpha <= 0.0:
                continue

            scale = math.sqrt(alpha)
            for _ in range(self.aug_per_sample):
                eps = torch.randn(D, device=device)
                epsN = proj_N(eps)
                z_tilde = z[i] + scale * epsN
                z_out.append(z_tilde)
                y_out.append(y[i])

        if len(z_out) == 0:
            return None, None

        z_aug = torch.stack(z_out, dim=0)
        y_aug = torch.stack(y_out, dim=0).long()
        return z_aug, y_aug
