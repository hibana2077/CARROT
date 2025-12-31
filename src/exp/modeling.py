from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

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


def _is_module_sequence(x: object) -> bool:
    return isinstance(x, (nn.ModuleList, list, tuple))


def discover_backbone_groups(backbone: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return an ordered list of trainable "groups" inside a timm backbone.

    Goal: support ViT/Swin/ResNet-like backbones without hardcoding specific model names.

    Heuristics (first match wins):
    - ViT-like: backbone.blocks (ModuleList)
    - Swin-like: backbone.layers or backbone.stages (ModuleList)
    - ResNet-like: layer1..layer4
    - Fallback: top-level children modules
    """
    if hasattr(backbone, "blocks") and _is_module_sequence(getattr(backbone, "blocks")):
        blocks = getattr(backbone, "blocks")
        return [(f"blocks.{i}", m) for i, m in enumerate(blocks)]

    if hasattr(backbone, "layers") and _is_module_sequence(getattr(backbone, "layers")):
        layers = getattr(backbone, "layers")
        return [(f"layers.{i}", m) for i, m in enumerate(layers)]

    if hasattr(backbone, "stages") and _is_module_sequence(getattr(backbone, "stages")):
        stages = getattr(backbone, "stages")
        return [(f"stages.{i}", m) for i, m in enumerate(stages)]

    stage_names = ["layer1", "layer2", "layer3", "layer4"]
    if all(hasattr(backbone, n) for n in stage_names):
        return [(n, getattr(backbone, n)) for n in stage_names]

    # Generic fallback: rely on named_children() order
    return [(name, module) for name, module in backbone.named_children()]


def set_trainable_backbone_groups(backbone: nn.Module, train_last_n: int) -> List[str]:
    """Freeze entire backbone, then unfreeze the last N discovered groups.

    Returns the list of group names that were unfrozen (may be empty).
    """
    set_requires_grad(backbone, False)
    n = int(train_last_n)
    if n <= 0:
        return []

    groups = discover_backbone_groups(backbone)
    if not groups:
        return []

    selected = groups[-n:]
    for _name, module in selected:
        for p in module.parameters():
            p.requires_grad = True
    return [name for name, _m in selected]


def freeze_batchnorm_stats(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def make_optimizer(
    optimizer_name: str,
    head: nn.Module,
    backbone: nn.Module,
    lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    params: List[Dict[str, object]] = []
    head_params: List[nn.Parameter] = [p for p in head.parameters() if p.requires_grad]
    params.append({"params": head_params, "lr": lr, "weight_decay": weight_decay})

    backbone_params: List[nn.Parameter] = [p for p in backbone.parameters() if p.requires_grad]
    if backbone_params:
        params.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})

    if optimizer_name == "adamw":
        return torch.optim.AdamW(params)
    return torch.optim.SGD(params, momentum=0.9)
