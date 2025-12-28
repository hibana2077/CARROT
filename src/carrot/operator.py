import torch

class DiffusionOperator:
    """
    Applies diffusion operator on the graph.
    H' = exp(-t L) @ H
    """
    def __init__(self, t: float):
        self.t = t

    def forward(self, H: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (B, N, D) Node features
            L: (B, N, N) Graph Laplacian
        Returns:
            H_prime: (B, N, D) Diffused features
        """
        # matrix_exp supports batching
        # M = -t * L
        M = -self.t * L
        
        # exp(M)
        exp_M = torch.matrix_exp(M) # (B, N, N)
        
        # H' = exp(-tL) H
        H_prime = torch.bmm(exp_M, H) # (B, N, D)
        
        return H_prime
