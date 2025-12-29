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

    def forward(self, H_prime: torch.Tensor, W: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            H_prime: (B, N, D)
            W: (B, N, N) optional adjacency (required for degree-weighted)
        Returns:
            g: (B, D)
        """
        if self.method == 'mean':
            return H_prime.mean(dim=1)
        elif self.method == 'sum':
            return H_prime.sum(dim=1)
        elif self.method == 'max':
            return H_prime.max(dim=1)[0]
        elif self.method in ('topk', 'top-k', 'top_k'):
            # Select K nodes with largest feature norm and average their features.
            B, N, D = H_prime.shape
            k = min(max(self.top_k, 1), N)
            scores = torch.linalg.vector_norm(H_prime, ord=2, dim=-1)  # (B, N)
            topk_idx = torch.topk(scores, k=k, dim=1).indices  # (B, k)
            gathered = torch.gather(H_prime, dim=1, index=topk_idx.unsqueeze(-1).expand(-1, -1, D))
            return gathered.mean(dim=1)
        elif self.method in ('degree', 'degree_weighted', 'centrality', 'centrality_weighted'):
            if W is None:
                raise ValueError("Readout method 'degree' requires W (adjacency) to be provided.")
            deg = W.sum(dim=-1)  # (B, N)
            w = deg / (deg.sum(dim=1, keepdim=True) + self.eps)  # (B, N)
            return (H_prime * w.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f"Unknown readout method: {self.method}")
