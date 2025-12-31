from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    feats_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    for batch in loader:
        if len(batch) == 2:
            images, targets = batch
        else:
            images, targets, _indices = batch
        images = images.to(device, non_blocking=True)
        feats = backbone(images)
        feats_list.append(feats.detach().cpu())
        labels_list.append(targets.detach().cpu())
    return torch.cat(feats_list, dim=0), torch.cat(labels_list, dim=0)


@torch.no_grad()
def compute_representer_values(
    head: nn.Module,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    lambda_l2: float,
) -> torch.Tensor:
    if lambda_l2 <= 0:
        raise ValueError("lambda_l2 must be > 0 for representer values.")

    device = next(head.parameters()).device
    h = train_feats.to(device)
    y = train_labels.to(device)
    logits = head(h)
    p = logits.softmax(dim=1)
    onehot = F.one_hot(y, num_classes=logits.size(1)).float()
    g = p - onehot
    n = float(h.size(0))
    r = -(1.0 / (2.0 * lambda_l2 * n)) * g
    return r.detach().cpu()


@torch.no_grad()
def representer_topk(
    train_feats: torch.Tensor,
    representer_r: torch.Tensor,
    query_feats: torch.Tensor,
    class_index: int,
    top_k: int,
    similarity: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    H = train_feats.float()
    q = query_feats.float()
    if similarity == "cosine":
        H = F.normalize(H, dim=1)
        q = F.normalize(q, dim=1)
    sim = (H @ q.T).squeeze(1)
    contrib = representer_r[:, class_index].float() * sim
    support = torch.topk(contrib, k=top_k, largest=True).indices
    inhibit = torch.topk(contrib, k=top_k, largest=False).indices
    return support, inhibit
