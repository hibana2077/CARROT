import torch

def training_contribution(g_test: torch.Tensor, G_train: torch.Tensor, Y_train: torch.Tensor, lambda_reg: float):
    """
    Calculate contribution of each training sample to the prediction of g_test.
    
    f(x) = x^T W* 
         = x^T (G^T G + n lambda I)^-1 G^T Y
         = (x^T M G^T) Y
         = sum_i (x^T M g_i) y_i
         
    Contribution weight of sample i is alpha_i = (x^T M g_i)
    
    Args:
        g_test: (D,) or (1, D) Test image embedding
        G_train: (N_train, D) Training embeddings
        Y_train: (N_train, C) Training labels (one-hot)
        lambda_reg: Regularization strength
        
    Returns:
        influence: (1, N_train) The weight of each training sample
    """
    if g_test.dim() == 1:
        g_test = g_test.unsqueeze(0)
        
    N_train, D = G_train.shape
    
    # M = (G^T G + n lambda I)^-1
    GTG = torch.matmul(G_train.T, G_train)
    I = torch.eye(D, device=G_train.device)
    reg_term = N_train * lambda_reg * I
    M = torch.linalg.inv(GTG + reg_term) # (D, D)
    
    # x^T M: (1, D)
    xtM = torch.matmul(g_test, M)
    
    # Influence vector: (1, D) @ (D, N_train) -> (1, N_train)
    influence = torch.matmul(xtM, G_train.T) 
    
    return influence
