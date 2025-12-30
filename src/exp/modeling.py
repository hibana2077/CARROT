from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

try:
    import timm
except Exception as e:  # pragma: no cover
    timm = None
    _timm_import_error = e


def create_backbone_and_head(
    model_name: str,
    pretrained: bool,
    num_classes: int,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module, int]:
    if timm is None:
        raise RuntimeError(
            "timm is not available. Install it (pip install timm). Import error: " + repr(_timm_import_error)
        )

    backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
    backbone.to(device)
    backbone.eval()

    feat_dim = backbone.num_features
    head = nn.Linear(feat_dim, num_classes, bias=True).to(device)
    return backbone, head, feat_dim


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def freeze_batchnorm_stats(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def make_optimizer(
    optimizer_name: str,
    head: nn.Module,
    backbone: nn.Module,
    freeze_backbone: bool,
    lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    params: List[Dict[str, object]] = []
    params.append({"params": head.parameters(), "lr": lr, "weight_decay": weight_decay})
    if not freeze_backbone:
        params.append({"params": backbone.parameters(), "lr": backbone_lr, "weight_decay": weight_decay})

    if optimizer_name == "adamw":
        return torch.optim.AdamW(params)
    return torch.optim.SGD(params, momentum=0.9)
