import torch
import torch.nn as nn
from .regions import RegionSet

class RegionGraphBuilder:
    """
    Constructs the region-relation graph (W, L) from RegionSet.
    """
    def __init__(
        self,
        sigma_s: float = 0.5,
        sigma_f: float = 1.0,
        *,
        feature_norm: bool = False,
        feature_metric: str = 'l2',
        adaptive_sigma: bool = False,
        sigma_percentile: float = 0.5,
        knn_k: int = 0,
        force_fp32: bool = False,
        add_self_loops: bool = True,
        eps: float = 1e-8,
    ):
        self.sigma_s = float(sigma_s)
        self.sigma_f = float(sigma_f)
        self.feature_norm = bool(feature_norm)
        self.feature_metric = str(feature_metric).lower()
        self.adaptive_sigma = bool(adaptive_sigma)
        self.sigma_percentile = float(sigma_percentile)
        self.knn_k = int(knn_k)
        self.force_fp32 = bool(force_fp32)
        self.add_self_loops = bool(add_self_loops)
        self.eps = float(eps)

    def _maybe_fp32(self, x: torch.Tensor) -> torch.Tensor:
        if not self.force_fp32:
            return x
        # Keep computation stable: do distances/exp in float32
        return x.float() if x.dtype != torch.float32 else x

    def _feature_preprocess(self, H: torch.Tensor) -> torch.Tensor:
        if not self.feature_norm:
            return H
        return H / (torch.linalg.vector_norm(H, ord=2, dim=-1, keepdim=True) + self.eps)

    def _adaptive_sigma_from_dist(self, dist: torch.Tensor, base: float) -> torch.Tensor:
        """Compute per-batch sigma from off-diagonal distance percentiles.

        Args:
            dist: (B, N, N)
            base: multiplier (keeps existing sigma meaning usable)
        Returns:
            sigma_eff: (B, 1, 1)
        """
        if not self.adaptive_sigma:
            return torch.full((dist.size(0), 1, 1), float(base), device=dist.device, dtype=dist.dtype)

        B, N, _ = dist.shape
        off_mask = ~torch.eye(N, device=dist.device, dtype=torch.bool)
        flat = dist[:, off_mask]  # (B, N*(N-1))
        q = float(self.sigma_percentile)
        q = min(max(q, 0.0), 1.0)
        # quantile expects float; compute in current dtype (likely fp32 after _maybe_fp32)
        scale = torch.quantile(flat, q=q, dim=1, keepdim=True)  # (B, 1)
        sigma_eff = (scale * float(base)).clamp_min(self.eps).view(B, 1, 1)
        return sigma_eff

    def _apply_knn(self, W: torch.Tensor) -> torch.Tensor:
        """Keep top-k neighbors per node (row-wise) and symmetrize."""
        k = int(self.knn_k)
        if k <= 0:
            return W

        B, N, _ = W.shape
        k = min(k, max(N - 1, 1))

        # Exclude diagonal for neighbor selection
        W_off = W.clone()
        diag_idx = torch.arange(N, device=W.device)
        W_off[:, diag_idx, diag_idx] = float('-inf')

        topk_idx = torch.topk(W_off, k=k, dim=-1).indices  # (B, N, k)
        mask = torch.zeros((B, N, N), device=W.device, dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk_idx, value=True)
        # Symmetrize so graph is undirected-ish
        mask = mask | mask.transpose(1, 2)

        if self.add_self_loops:
            mask[:, diag_idx, diag_idx] = True

        return W * mask.to(W.dtype)

    def build(self, regions: RegionSet):
        """
        Args:
            regions: RegionSet with features (B, N, D) and positions (N, 2)
        Returns:
            W: (B, N, N) weighted adjacency matrix
            L: (B, N, N) normalized Laplacian matrix
        """
        H = regions.features  # (B, N, D)
        P = regions.positions # (N, 2)

        B, N, D = H.shape

        H = self._maybe_fp32(H)
        P = self._maybe_fp32(P)
        
        # Expand P to (B, N, 2) for batch processing
        if P.dim() == 2:
            P = P.unsqueeze(0).expand(B, -1, -1)
        
        # 1. Pairwise spatial distance
        # dist_s: (B, N, N)
        dist_s = torch.cdist(P, P, p=2)
        sigma_s_eff = self._adaptive_sigma_from_dist(dist_s, self.sigma_s).to(dist_s.dtype)
        W_s = torch.exp(- (dist_s ** 2) / (sigma_s_eff ** 2))

        # 2. Pairwise feature distance
        # dist_f: (B, N, N)
        H_used = self._feature_preprocess(H)
        if self.feature_metric in ('cos', 'cosine'):
            # cosine distance in [0, 2] when inputs are normalized
            if not self.feature_norm:
                # If user requested cosine but didn't enable normalization, do it implicitly.
                H_used = self._feature_preprocess(H_used)
            sim = torch.bmm(H_used, H_used.transpose(1, 2)).clamp(-1.0, 1.0)
            dist_f = (1.0 - sim).clamp_min(0.0)
        elif self.feature_metric in ('l2', 'euclidean'):
            dist_f = torch.cdist(H_used, H_used, p=2)
        else:
            raise ValueError(f"Unknown feature_metric: {self.feature_metric} (use 'l2' or 'cosine')")

        sigma_f_eff = self._adaptive_sigma_from_dist(dist_f, self.sigma_f).to(dist_f.dtype)
        W_f = torch.exp(- (dist_f ** 2) / (sigma_f_eff ** 2))

        # 3. Combined W (Element-wise multiplication)
        W = W_s * W_f  # (B, N, N)

        if not self.add_self_loops:
            diag_idx = torch.arange(N, device=W.device)
            W[:, diag_idx, diag_idx] = 0.0

        # Optional: kNN sparsification (C)
        W = self._apply_knn(W)

        # 4. Normalized Laplacian L = I - D^-1/2 W D^-1/2
        # Degree matrix D: D_ii = sum_j W_ij
        D_diag = W.sum(dim=-1)  # (B, N)
        
        # Avoid division by zero
        D_inv_sqrt = torch.pow(D_diag + 1e-6, -0.5)  # (B, N)
        D_inv_sqrt_mat = torch.diag_embed(D_inv_sqrt) # (B, N, N)

        I = torch.eye(N, device=W.device).unsqueeze(0).expand(B, -1, -1)
        
        # L = I - D^-1/2 W D^-1/2
        # Note: bmm is batch matrix multiplication
        L = I - torch.bmm(torch.bmm(D_inv_sqrt_mat, W), D_inv_sqrt_mat)

        return W, L
