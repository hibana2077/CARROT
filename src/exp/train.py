from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.exp.common import accuracy_top1, avg_meter_compute, avg_meter_update
from src.exp.modeling import freeze_batchnorm_stats


def _linear_forward(feats: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    logits = feats @ weight.t()
    if bias is not None:
        logits = logits + bias
    return logits


def train_one_epoch(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_l2: float,
    warmup: bool,
    val_loader: Optional[DataLoader] = None,
    alpha_weights: Optional[object] = None,
    alpha_optimizer: Optional[torch.optim.Optimizer] = None,
    alpha_inner_lr: Optional[float] = None,
    num_classes: Optional[int] = None,
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

    alpha_on = alpha_weights is not None
    if alpha_on:
        if val_loader is None:
            raise ValueError("val_loader is required when alpha_weights is provided")
        if alpha_optimizer is None:
            raise ValueError("alpha_optimizer is required when alpha_weights is provided")
        if num_classes is None:
            raise ValueError("num_classes is required when alpha_weights is provided")
        if warmup:
            raise ValueError("alpha reweighting is not supported in warmup stage (use warmup_epochs=0)")
        if any(p.requires_grad for p in backbone.parameters()):
            raise ValueError(
                "alpha 1-step unroll requires a frozen backbone (set train_last_n=0 in focus stage)"
            )
        if not isinstance(head, nn.Linear):
            raise ValueError("alpha 1-step unroll currently supports nn.Linear head only")

        if alpha_inner_lr is None:
            alpha_inner_lr = float(optimizer.param_groups[0]["lr"])
        else:
            alpha_inner_lr = float(alpha_inner_lr)

        val_iter = iter(val_loader)
    else:
        val_iter = None

    for batch in loader:
        if len(batch) == 2:
            images, targets = batch
            indices = None
        else:
            images, targets, indices = batch

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # Head-only stage: backbone_trainable is False. Keep feats detached.
        with torch.set_grad_enabled(backbone_trainable):
            feats = backbone(images)
        feats = feats.detach()

        if alpha_on:
            if indices is None:
                raise ValueError("Dataloader must return indices when alpha is enabled (return_index=True)")
            indices = indices.to(device, non_blocking=True)

            # 1) Update alpha via 1-step unrolled head on a val batch
            try:
                val_batch = next(val_iter)  # type: ignore[arg-type]
            except StopIteration:
                val_iter = iter(val_loader)  # type: ignore[arg-type]
                val_batch = next(val_iter)  # type: ignore[arg-type]

            if len(val_batch) == 2:
                val_images, val_targets = val_batch
            else:
                val_images, val_targets, _ = val_batch
            val_images = val_images.to(device, non_blocking=True)
            val_targets = val_targets.to(device, non_blocking=True)

            with torch.no_grad():
                val_feats = backbone(val_images).detach()

            # Train loss per-sample (needs grad through alpha)
            alpha_optimizer.zero_grad(set_to_none=True)
            loss_vec = F.cross_entropy(_linear_forward(feats, head.weight, head.bias), targets, reduction="none")
            alpha_batch = alpha_weights.batch_alpha(indices=indices, targets=targets, num_classes=int(num_classes))
            # Weighted mean; classwise normalization keeps scale stable.
            train_loss_for_unroll = (alpha_batch * loss_vec).mean()

            l2 = 0.5 * lambda_l2 * (
                head.weight.pow(2).sum() + (head.bias.pow(2).sum() if head.bias is not None else 0.0)
            )
            train_loss_for_unroll = train_loss_for_unroll + l2

            # Simulate 1 SGD step on head params
            grads = torch.autograd.grad(
                train_loss_for_unroll,
                [head.weight, head.bias] if head.bias is not None else [head.weight],
                create_graph=True,
            )
            if head.bias is None:
                (gw,) = grads
                gb = None
            else:
                gw, gb = grads

            w_prime = head.weight - float(alpha_inner_lr) * gw
            b_prime = head.bias - float(alpha_inner_lr) * gb if gb is not None else None

            val_logits_prime = _linear_forward(val_feats, w_prime, b_prime)
            val_loss = F.cross_entropy(val_logits_prime, val_targets, reduction="mean")

            # Optional alpha regularizer to prevent collapse
            val_loss = val_loss + alpha_weights.regularizer(
                alpha_batch=alpha_batch,
                targets=targets,
                num_classes=int(num_classes),
            )

            val_loss.backward()
            alpha_optimizer.step()

        # 2) Update head (and any trainable backbone params, if configured)
        optimizer.zero_grad(set_to_none=True)
        logits = head(feats)  # head params require grad
        if alpha_on:
            # Recompute alpha after the alpha step; detach so head update doesn't backprop into alpha.
            alpha_batch2 = alpha_weights.batch_alpha(indices=indices, targets=targets, num_classes=int(num_classes)).detach()
            loss_vec2 = F.cross_entropy(logits, targets, reduction="none")
            loss_data = (alpha_batch2 * loss_vec2).mean()
        else:
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
