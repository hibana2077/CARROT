from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.exp.common import accuracy_top1, avg_meter_compute, avg_meter_update


def train_one_epoch(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_l2: float,
    freeze_backbone: bool,
    alpha: Optional[torch.Tensor],
) -> Tuple[float, float]:
    if freeze_backbone:
        backbone.eval()
    else:
        backbone.train()
    head.train()

    meter: Dict[str, float] = {}
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(not freeze_backbone):
            feats = backbone(images)
        logits = head(feats)

        ce = F.cross_entropy(logits, targets, reduction="none")
        if alpha is not None:
            raise RuntimeError(
                "alpha training requires dataset indices. Use alpha_mode=none for now, "
                "or extend the dataset to return (image, label, index)."
            )
        loss_data = ce.mean()

        loss_l2 = 0.5 * lambda_l2 * (
            head.weight.pow(2).sum() + (head.bias.pow(2).sum() if head.bias is not None else 0.0)
        )
        loss = loss_data + loss_l2

        loss.backward()
        optimizer.step()

        bsz = targets.size(0)
        avg_meter_update(meter, "loss_sum", loss_data.detach().item() * bsz)
        avg_meter_update(meter, "acc1_sum", accuracy_top1(logits.detach(), targets) * bsz)
        total += bsz

    avg = avg_meter_compute(meter, total)
    return float(avg["loss_sum"]), float(avg["acc1_sum"])


@torch.no_grad()
def evaluate(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    backbone.eval()
    head.eval()

    meter: Dict[str, float] = {}
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        feats = backbone(images)
        logits = head(feats)
        loss = F.cross_entropy(logits, targets, reduction="mean")

        bsz = targets.size(0)
        avg_meter_update(meter, "loss_sum", loss.item() * bsz)
        avg_meter_update(meter, "acc1_sum", accuracy_top1(logits, targets) * bsz)
        total += bsz

    avg = avg_meter_compute(meter, total)
    return float(avg["loss_sum"]), float(avg["acc1_sum"])
