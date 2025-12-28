import torch
import numpy as np

class RidgeHead:
    """
    Closed-form Ridge Regression Head.
    Solves W* = (G^T G + n*lambda*I)^(-1) G^T Y
    """
    def __init__(self, lambda_reg: float = 1.0):
        self.lambda_reg = lambda_reg
        self.W_star = None # (D, C)
        self.G_train = None # (N_train, D)
        self.Y_train = None # (N_train, C)

    def fit(self, G_train: torch.Tensor, Y_train: torch.Tensor):
        """
        Args:
            G_train: (N_train, D) Training embeddings
            Y_train: (N_train,) or (N_train, C) Training labels
        """
        # If Y_train is labels, convert to one-hot
        if Y_train.dim() == 1:
            num_classes = int(Y_train.max().item()) + 1
            Y_onehot = torch.zeros(Y_train.size(0), num_classes, device=Y_train.device)
            Y_onehot.scatter_(1, Y_train.unsqueeze(1), 1)
            Y_train = Y_onehot
        
        self.G_train = G_train
        self.Y_train = Y_train
        
        N_train, D = G_train.shape
        
        # W* = (G^T G + n*lambda*I)^(-1) G^T Y
        # G^T G: (D, D)
        GTG = torch.matmul(G_train.T, G_train)
        I = torch.eye(D, device=G_train.device)
        
        # Regularization term: n * lambda * I
        reg_term = N_train * self.lambda_reg * I
        
        # Inverse term
        inv_term = torch.linalg.inv(GTG + reg_term)
        
        # (D, D) @ (D, N) @ (N, C) -> (D, C)
        self.W_star = torch.matmul(torch.matmul(inv_term, G_train.T), Y_train)
        
    def predict(self, g_test: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g_test: (B, D)
        Returns:
            logits: (B, C)
        """
        if self.W_star is None:
            raise RuntimeError("Head not fitted yet.")
        
        return torch.matmul(g_test, self.W_star)
