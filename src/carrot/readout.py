import torch
import torch.nn as nn

class GraphReadout(nn.Module):
    """
    Aggregates node features to graph/image embedding.
    """
    def __init__(self, method: str = 'mean'):
        super().__init__()
        self.method = method

    def forward(self, H_prime: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_prime: (B, N, D)
        Returns:
            g: (B, D)
        """
        if self.method == 'mean':
            return H_prime.mean(dim=1)
        elif self.method == 'sum':
            return H_prime.sum(dim=1)
        elif self.method == 'max':
            return H_prime.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown readout method: {self.method}")
