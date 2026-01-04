from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class FeatureCatcher:
    """Forward-hook based feature capture.

    Stores raw module outputs for later regularization.
    """

    def __init__(self) -> None:
        self.feats: Dict[str, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, name: str):
        def fn(_module: nn.Module, _args, output):
            # Some timm blocks return tuples; keep the first tensor.
            if isinstance(output, (tuple, list)):
                output = output[0]
            if torch.is_tensor(output):
                self.feats[name] = output

        return fn

    def add(self, module: nn.Module, name: str) -> None:
        h = module.register_forward_hook(self._hook(name))
        self.handles.append(h)

    def clear(self) -> None:
        self.feats.clear()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def _to_vec(feat: torch.Tensor) -> torch.Tensor:
    # CNN: (B,C,H,W) -> GAP -> (B,C)
    if feat.dim() == 4:
        return feat.mean(dim=(2, 3))
    # ViT: (B,N,D) -> CLS -> (B,D)
    if feat.dim() == 3:
        return feat[:, 0, :]
    # Already (B,D)
    if feat.dim() == 2:
        return feat
    raise ValueError(f"Unsupported feature shape: {tuple(feat.shape)}")


def _class_stats_diag(
    z: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-class mean and diagonal variance in-batch.

    Returns: classes, mu, var, counts, inv
    """

    classes, inv = torch.unique(y, sorted=True, return_inverse=True)
    K = int(classes.numel())
    B, D = z.shape

    counts = torch.zeros(K, device=z.device, dtype=z.dtype)
    ones = torch.ones(B, device=z.device, dtype=z.dtype)
    counts.index_add_(0, inv, ones)

    sum_z = torch.zeros(K, D, device=z.device, dtype=z.dtype)
    sum_z2 = torch.zeros(K, D, device=z.device, dtype=z.dtype)
    sum_z.index_add_(0, inv, z)
    sum_z2.index_add_(0, inv, z * z)

    denom = counts[:, None].clamp_min(1.0)
    mu = sum_z / denom
    var = (sum_z2 / denom) - mu * mu
    var = var.clamp_min(eps)

    return classes, mu, var, counts, inv


@torch.no_grad()
def _confusion_alpha(
    logits: torch.Tensor,
    y: torch.Tensor,
    classes: torch.Tensor,
    inv: torch.Tensor,
) -> torch.Tensor:
    """Compute symmetric confusion weights alpha_{cd} on the K classes in this batch."""

    p = torch.softmax(logits, dim=1)  # (B,C_total)
    pK = p[:, classes]  # (B,K)
    B, K = pK.shape

    counts = torch.zeros(K, device=logits.device, dtype=pK.dtype)
    ones = torch.ones(B, device=logits.device, dtype=pK.dtype)
    counts.index_add_(0, inv, ones)

    sum_probs = torch.zeros(K, K, device=logits.device, dtype=pK.dtype)
    sum_probs.index_add_(0, inv, pK)

    mean_probs = sum_probs / counts[:, None].clamp_min(1.0)

    alpha = 0.5 * (mean_probs + mean_probs.t())
    alpha.fill_diagonal_(0.0)
    return alpha


def _bhattacharyya_diag(
    mu: torch.Tensor,
    var: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bhattacharyya distance for diagonal Gaussians.

    Returns (D, rho=exp(-D)) of shape (K,K).
    """

    mu_i = mu[:, None, :]  # (K,1,D)
    mu_j = mu[None, :, :]  # (1,K,D)
    diff = mu_i - mu_j  # (K,K,D)

    var_i = var[:, None, :]
    var_j = var[None, :, :]
    var_avg = 0.5 * (var_i + var_j) + eps

    term1 = 0.125 * (diff * diff / var_avg).sum(dim=-1)

    log_var = torch.log(var + eps)
    log_var_i = log_var[:, None, :].sum(dim=-1)
    log_var_j = log_var[None, :, :].sum(dim=-1)
    log_var_avg = torch.log(var_avg).sum(dim=-1)

    term2 = 0.5 * (log_var_avg - 0.5 * (log_var_i + log_var_j))

    D = term1 + term2
    D = D.clamp_min(0.0)
    rho = torch.exp(-D)
    return D, rho


def _bhat_confusion_loss(
    mu: torch.Tensor,
    var: torch.Tensor,
    alpha: torch.Tensor,
    top_m: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    K = int(mu.size(0))
    if K < 2:
        return mu.new_tensor(0.0)

    _, rho = _bhattacharyya_diag(mu, var, eps=eps)

    triu = torch.triu_indices(K, K, offset=1, device=mu.device)
    a = alpha[triu[0], triu[1]]
    r = rho[triu[0], triu[1]]

    if int(a.numel()) == 0:
        return mu.new_tensor(0.0)

    if top_m is not None and int(a.numel()) > int(top_m):
        a_top, idx = torch.topk(a, k=int(top_m), largest=True)
        a = a_top
        r = r[idx]

    denom = a.sum().clamp_min(eps)
    return (a * r).sum() / denom


class ConfusionWeightedBhatReg(nn.Module):
    """Multi-layer confusion-weighted Bhattacharyya regularizer (diag covariance)."""

    def __init__(self, layer_names: Sequence[str], top_m: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.layer_names = list(layer_names)
        self.top_m = int(top_m)
        self.eps = float(eps)

    def forward(self, feats: Dict[str, torch.Tensor], logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total = logits.new_tensor(0.0)
        used = 0

        for name in self.layer_names:
            if name not in feats:
                continue

            z = _to_vec(feats[name]).float()  # stats in fp32

            # First pass: find classes with >=2 samples
            classes, _mu, _var, counts, _inv = _class_stats_diag(z, y, eps=self.eps)
            mask = counts >= 2
            if int(mask.sum().item()) < 2:
                continue

            kept_classes = classes[mask]
            keep = torch.isin(y, kept_classes)
            if int(keep.sum().item()) < 4:
                continue

            z2 = z[keep]
            y2 = y[keep]
            logits2 = logits[keep]

            classes2, mu2, var2, _counts2, inv2 = _class_stats_diag(z2, y2, eps=self.eps)
            if int(classes2.numel()) < 2:
                continue

            alpha = _confusion_alpha(logits2, y2, classes2, inv2)
            loss_layer = _bhat_confusion_loss(mu2, var2, alpha, top_m=self.top_m, eps=self.eps)

            total = total + loss_layer
            used += 1

        if used == 0:
            return logits.new_tensor(0.0)
        return total / float(used)


def resolve_module(root: nn.Module, path: str) -> nn.Module:
    """Resolve a module from a dotted path with optional integer indexing.

    Examples:
      - "blocks.11"
      - "layer3.-1" (negative indices allowed)
    """

    cur: object = root
    for part in [p for p in path.split(".") if p]:
        if isinstance(cur, nn.Module) and hasattr(cur, part):
            cur = getattr(cur, part)
            continue

        # Try integer index into sequences (Sequential/ModuleList/etc.)
        try:
            idx = int(part)
        except ValueError as e:
            raise ValueError(f"Cannot resolve '{part}' in path '{path}'") from e

        if hasattr(cur, "__getitem__"):
            cur = cur[idx]
        else:
            raise ValueError(f"Object at '{part}' in path '{path}' is not indexable")

    if not isinstance(cur, nn.Module):
        raise ValueError(f"Resolved object for '{path}' is not an nn.Module")
    return cur


def default_bhat_layer_paths(backbone: nn.Module) -> List[str]:
    """Heuristic default layers for common timm backbones."""

    # ViT-like
    if hasattr(backbone, "blocks") and hasattr(backbone.blocks, "__len__"):
        n = len(backbone.blocks)
        k = min(4, n)
        return [f"blocks.{i}" for i in range(n - k, n)]

    # ResNet-like
    if all(hasattr(backbone, x) for x in ("layer2", "layer3", "layer4")):
        return ["layer2.-1", "layer3.-1", "layer4.-1"]

    return []
