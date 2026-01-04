import torch
import torch.nn as nn
from typing import Dict, Tuple

def _trace_cov_from_centered(xc: torch.Tensor) -> torch.Tensor:
    """
    xc: [n, d] centered
    trace(cov) = sum_j Var_j = mean over samples of squared centered coords, summed over dims.
    """
    n = xc.shape[0]
    # cov diag = mean(xc^2) ; trace = sum(diag)
    return (xc.pow(2).sum(dim=1).mean())  # equivalent to mean over samples of ||xc||^2

def _effective_rank_from_centered(xc: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Effective rank based on covariance eigenvalues.
    For xc [n, d], eigenvalues of cov are (s^2)/n where s are singular values of xc.
    We compute entropy of normalized eigenvalues -> erank = exp(H).
    """
    n, d = xc.shape
    # SVD on [n, d] (usually n << d), take singular values
    s = torch.linalg.svdvals(xc)  # [min(n, d)]
    lam = (s * s) / max(n, 1)     # eigenvalues (up to rank)
    lam_sum = lam.sum().clamp_min(eps)
    p = (lam / lam_sum).clamp_min(eps)
    H = -(p * torch.log(p)).sum()
    erank = torch.exp(H)
    return erank  # in [1, min(n,d)]

def carrot_regularizer(
    z: torch.Tensor,  # [B, D]
    y: torch.Tensor,  # [B]
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Returns:
      R = R_var + R_rank
      stats for logging
    """
    assert z.dim() == 2 and y.dim() == 1 and z.shape[0] == y.shape[0]
    B, D = z.shape

    # 建議用 float32 計算統計量，尤其是 SVD（AMP 下避免 half 精度不穩）
    zf = z.float()
    yf = y

    # batch total scatter (for self-normalization)
    mu_b = zf.mean(dim=0, keepdim=True)
    xc_b = zf - mu_b
    S_bar = _trace_cov_from_centered(xc_b).clamp_min(eps)

    classes = torch.unique(yf)
    r_var_terms = []
    r_rank_terms = []

    used = 0
    min_r = float("inf")
    min_er = float("inf")

    for c in classes:
        idx = (yf == c).nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n < 2:
            continue  # 沒法估 cov，就跳過（不然會很 noisy）
        used += 1

        X = zf.index_select(0, idx)                  # [n, D]
        mu = X.mean(dim=0, keepdim=True)
        xc = X - mu                                  # centered

        # variance barrier
        S_c = _trace_cov_from_centered(xc).clamp_min(eps)
        r_c = (S_c / S_bar).clamp_min(eps)
        r_var_terms.append(-torch.log(r_c))

        # rank barrier
        erank = _effective_rank_from_centered(xc, eps=eps)   # [1, min(n,D)]
        er_norm = (erank / D).clamp_min(eps)
        r_rank_terms.append(-torch.log(er_norm))

        min_r = min(min_r, float(r_c.detach().cpu()))
        min_er = min(min_er, float(er_norm.detach().cpu()))

    if used == 0:
        R_var = zf.new_tensor(0.0)
        R_rank = zf.new_tensor(0.0)
    else:
        R_var = torch.stack(r_var_terms).mean()
        R_rank = torch.stack(r_rank_terms).mean()

    R = R_var + R_rank
    stats = {
        "carrot/R": float(R.detach().cpu()),
        "carrot/R_var": float(R_var.detach().cpu()),
        "carrot/R_rank": float(R_rank.detach().cpu()),
        "carrot/used_classes": used,
        "carrot/min_r": (min_r if used > 0 else float("nan")),
        "carrot/min_erank_norm": (min_er if used > 0 else float("nan")),
    }
    return R, stats
