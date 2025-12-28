import torch
import torch.nn as nn
from .regions import RegionSet

class RegionGraphBuilder:
    """
    Constructs the region-relation graph (W, L) from RegionSet.
    """
    def __init__(self, sigma_s: float = 0.5, sigma_f: float = 1.0):
        self.sigma_s = sigma_s
        self.sigma_f = sigma_f

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
        
        # Expand P to (B, N, 2) for batch processing
        if P.dim() == 2:
            P = P.unsqueeze(0).expand(B, -1, -1)
        
        # 1. Pairwise spatial distance
        # dist_s: (B, N, N)
        dist_s = torch.cdist(P, P, p=2) 
        W_s = torch.exp(- (dist_s ** 2) / (self.sigma_s ** 2))

        # 2. Pairwise feature distance
        # dist_f: (B, N, N)
        dist_f = torch.cdist(H, H, p=2)
        W_f = torch.exp(- (dist_f ** 2) / (self.sigma_f ** 2))

        # 3. Combined W (Element-wise multiplication)
        W = W_s * W_f # (B, N, N)

        # 4. Normalized Laplacian L = I - D^-1/2 W D^-1/2
        # Degree matrix D: D_ii = sum_j W_ij
        D_diag = W.sum(dim=-1) # (B, N)
        
        # Avoid division by zero
        D_inv_sqrt = torch.pow(D_diag + 1e-6, -0.5) # (B, N)
        D_inv_sqrt_mat = torch.diag_embed(D_inv_sqrt) # (B, N, N)

        I = torch.eye(N, device=W.device).unsqueeze(0).expand(B, -1, -1)
        
        # L = I - D^-1/2 W D^-1/2
        # Note: bmm is batch matrix multiplication
        L = I - torch.bmm(torch.bmm(D_inv_sqrt_mat, W), D_inv_sqrt_mat)

        return W, L
