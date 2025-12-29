import torch
import torch.nn as nn

class GraphReadout(nn.Module):
    """
    Aggregates node features to graph/image embedding.
    """
    def __init__(self, method: str = 'mean', top_k: int = 8, eps: float = 1e-8):
        super().__init__()
        self.method = method
        self.top_k = int(top_k)
        self.eps = float(eps)

    def forward(
        self,
        H_prime: torch.Tensor,
        W: torch.Tensor | None = None,
        H: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            H_prime: (B, N, D)
            W: (B, N, N) optional adjacency (required for degree-weighted)
            H: (B, N, D) optional original (pre-diffusion) node features (required for residual readouts)
        Returns:
            g: (B, D) or (B, 2D) depending on method
        """
        method = (self.method or 'mean').lower()

        if method == 'mean':
            return H_prime.mean(dim=1)
        elif method == 'sum':
            return H_prime.sum(dim=1)
        elif method == 'max':
            return H_prime.max(dim=1)[0]
        elif method in (
            'mean_max',
            'mean+max',
            'mean_max_concat',
            'meanmax',
            'mean-max',
        ):
            mean = H_prime.mean(dim=1)
            mx = H_prime.max(dim=1)[0]
            return torch.cat([mean, mx], dim=-1)
        elif method in (
            'mean_var',
            'mean+var',
            'mean_var_concat',
            'meanvar',
            'mean-var',
        ):
            mean = H_prime.mean(dim=1)
            var = H_prime.var(dim=1, unbiased=False)
            return torch.cat([mean, var], dim=-1)
        elif method in (
            'residual',
            'mean_residual',
            'mean_delta',
            'mean_h_and_mean_delta',
        ):
            if H is None:
                raise ValueError("Readout method 'residual' requires H (original features) to be provided.")
            return torch.cat([H.mean(dim=1), (H_prime - H).mean(dim=1)], dim=-1)
        elif method in (
            'residual_abs',
            'abs_residual',
            'mean_abs_residual',
            'mean_abs_delta',
            'mean_h_and_mean_abs_delta',
        ):
            if H is None:
                raise ValueError("Readout method 'residual_abs' requires H (original features) to be provided.")
            return torch.cat([H.mean(dim=1), (H_prime - H).abs().mean(dim=1)], dim=-1)
        elif method in ('topk', 'top-k', 'top_k'):
            # Select K nodes with largest feature norm and average their features.
            B, N, D = H_prime.shape
            k = min(max(self.top_k, 1), N)
            scores = torch.linalg.vector_norm(H_prime, ord=2, dim=-1)  # (B, N)
            topk_idx = torch.topk(scores, k=k, dim=1).indices  # (B, k)
            gathered = torch.gather(H_prime, dim=1, index=topk_idx.unsqueeze(-1).expand(-1, -1, D))
            return gathered.mean(dim=1)
        elif method in ('degree', 'degree_weighted', 'centrality', 'centrality_weighted'):
            if W is None:
                raise ValueError("Readout method 'degree' requires W (adjacency) to be provided.")
            deg = W.sum(dim=-1)  # (B, N)
            w = deg / (deg.sum(dim=1, keepdim=True) + self.eps)  # (B, N)
            return (H_prime * w.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f"Unknown readout method: {self.method}")
