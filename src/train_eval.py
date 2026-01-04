from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    # When running as a script: `python src/main.py`
    from metrics.calibration import ece_from_logits
    from metrics.classification import (
        balanced_accuracy,
        concentration_metrics,
        macro_f1,
        top1_correct,
        top_k_confusion_pairs_error,
    )
except ModuleNotFoundError:
    # When running as a module: `python -m src.main`
    from .metrics.calibration import ece_from_logits
    from .metrics.classification import (
        balanced_accuracy,
        concentration_metrics,
        macro_f1,
        top1_correct,
        top_k_confusion_pairs_error,
    )


@dataclass
class EpochStats:
    loss: float
    acc: float
    macro_f1: Optional[float] = None
    balanced_acc: Optional[float] = None
    top_k_conf_pairs_error: Optional[float] = None
    nll: Optional[float] = None
    ece: Optional[float] = None
    intra_sim: Optional[float] = None
    concentration: Optional[float] = None


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> EpochStats:
    model.train()

    total_samples = 0
    total_loss = 0.0
    total_correct = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, _z = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_correct += top1_correct(logits, targets)

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return EpochStats(loss=avg_loss, acc=avg_acc)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    top_k_pairs: int = 20,
) -> EpochStats:
    model.eval()

    total_samples = 0
    total_loss = 0.0
    total_correct = 0
    total_nll = 0.0
    total_ece_weighted = 0.0

    all_features = []
    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits, z = model(images)
        loss = criterion(logits, targets)
        nll_sum = float(F.cross_entropy(logits, targets, reduction="sum").item())
        ece = float(ece_from_logits(logits, targets).item())
        preds = torch.argmax(logits, dim=1)

        batch_size = targets.size(0)
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == targets).sum().item())
        total_nll += nll_sum
        total_ece_weighted += ece * batch_size

        all_features.append(z.detach().cpu())
        all_targets.append(targets.detach().cpu())
        all_preds.append(preds.detach().cpu())

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    avg_nll = total_nll / max(1, total_samples)
    avg_ece = total_ece_weighted / max(1, total_samples)

    intra_sim, concentration = 0.0, 0.0
    balanced_acc = 0.0
    macro = 0.0
    top_k_conf_pairs_error = 0.0

    if all_features:
        feats = torch.cat(all_features, dim=0)
        ys = torch.cat(all_targets, dim=0)
        ps = torch.cat(all_preds, dim=0)

        intra_sim, concentration = concentration_metrics(feats, ys)
        balanced_acc = balanced_accuracy(ys, ps)
        macro = macro_f1(ys, ps)
        num_classes = int(getattr(model, "num_classes", int(ps.max().item()) + 1))
        top_k_conf_pairs_error = top_k_confusion_pairs_error(ys, ps, num_classes, k=top_k_pairs)

    return EpochStats(
        loss=avg_loss,
        acc=avg_acc,
        macro_f1=macro,
        balanced_acc=balanced_acc,
        top_k_conf_pairs_error=top_k_conf_pairs_error,
        nll=avg_nll,
        ece=avg_ece,
        intra_sim=intra_sim,
        concentration=concentration,
    )
