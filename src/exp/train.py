from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.exp.common import accuracy_top1, avg_meter_compute, avg_meter_update
from src.exp.modeling import freeze_batchnorm_stats


def train_one_epoch(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_l2: float,
    warmup: bool,
) -> Tuple[float, float]:
    # Warm-up: backbone is fully trainable; otherwise it may be partially trainable.
    # If no backbone params require grad, eval mode avoids dropout noise.
    backbone_trainable = any(p.requires_grad for p in backbone.parameters())
    backbone.train(backbone_trainable)
    if not warmup:
        # If the backbone is partially trainable, keep BN running stats fixed.
        freeze_batchnorm_stats(backbone)
    head.train()

    meter: Dict[str, float] = {}
    total = 0

    for batch in loader:
        if len(batch) == 2:
            images, targets = batch
        else:
            images, targets, _indices = batch

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(backbone_trainable):
            feats = backbone(images)
        logits = head(feats)

        # Clean CE (no alpha weighting). Per docs: warm-up uses clean CE.
        # We also keep training clean CE in later stages unless you reintroduce reweighting.
        loss_data = F.cross_entropy(logits, targets, reduction="mean")

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
    for batch in loader:
        if len(batch) == 2:
            images, targets = batch
        else:
            images, targets, _indices = batch

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
