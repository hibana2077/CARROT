from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def top1_correct(logits: torch.Tensor, targets: torch.Tensor) -> int:
    preds = torch.argmax(logits, dim=1)
    return int((preds == targets).sum().item())


def balanced_accuracy(targets: torch.Tensor, preds: torch.Tensor) -> float:
    unique_classes = torch.unique(targets)
    recalls = []
    for c in unique_classes:
        mask = targets == c
        if int(mask.sum().item()) == 0:
            continue
        recalls.append(float((preds[mask] == c).float().mean().item()))
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def macro_f1(targets: torch.Tensor, preds: torch.Tensor) -> float:
    """Macro-F1 over classes present in targets."""
    targets = targets.to(torch.long)
    preds = preds.to(torch.long)

    unique_classes = torch.unique(targets)
    if unique_classes.numel() == 0:
        return 0.0

    f1s = []
    for c in unique_classes:
        c = int(c.item())
        tp = int(((preds == c) & (targets == c)).sum().item())
        fp = int(((preds == c) & (targets != c)).sum().item())
        fn = int(((preds != c) & (targets == c)).sum().item())
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2.0 * tp) / float(denom))

    return float(np.mean(f1s))


def top_k_confusion_pairs_error(
    targets: torch.Tensor,
    preds: torch.Tensor,
    num_classes: int,
    k: int = 20,
) -> float:
    """Error mass on the top-k most frequent off-diagonal confusion pairs."""
    if targets.numel() == 0:
        return 0.0

    targets = targets.to(torch.long)
    preds = preds.to(torch.long)

    indices = targets * int(num_classes) + preds
    cm_flat = torch.bincount(indices, minlength=int(num_classes) ** 2)
    cm = cm_flat.view(int(num_classes), int(num_classes))
    cm.fill_diagonal_(0)

    flat = cm.flatten()
    if flat.numel() == 0:
        return 0.0
    k = min(int(k), int(flat.numel()))
    top_vals, _ = torch.topk(flat, k=k, largest=True)
    return float(top_vals.sum().item() / float(targets.numel()))


def concentration_metrics(features: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    """Compute intra-class similarity and estimated concentration (kappa)."""
    features = F.normalize(features, p=2, dim=1)
    unique_classes = torch.unique(targets)

    intra_sims = []
    concentrations = []
    for c in unique_classes:
        mask = targets == c
        feats_c = features[mask]
        if feats_c.size(0) < 2:
            continue

        R = feats_c.mean(dim=0)
        R_norm = R.norm().item()
        intra_sims.append(R_norm)

        D = feats_c.size(1)
        if R_norm >= 0.999:
            kappa = 100.0
        else:
            kappa = (R_norm * D - R_norm**3) / (1 - R_norm**2)
        concentrations.append(kappa)

    if not intra_sims:
        return 0.0, 0.0
    return float(np.mean(intra_sims)), float(np.mean(concentrations))
