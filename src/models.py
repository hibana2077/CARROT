from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

import timm

try:
    # When running as a script: `python src/main.py`
    from zzz_head import ZZZHead
except ModuleNotFoundError:
    # When running as a module: `python -m src.main`
    from .zzz_head import ZZZHead


class FGModel(nn.Module):
    """Backbone + normalization + classifier head.

    Supports:
    - linear head (baseline)
    - ZZZ head (low-rank per-class covariance)

    Forward returns logits only to keep the training loop simple.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        head_type: str = "zzz",
        zzz_rank: int = 8,
        pretrained: bool = True,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        d = getattr(self.backbone, "num_features", None)
        if d is None:
            with torch.no_grad():
                x = torch.randn(1, 3, 224, 224)
                d = int(self.backbone(x).shape[-1])
        self.feat_dim = int(d)

        self.norm: Optional[nn.Module]
        if use_layernorm:
            self.norm = nn.LayerNorm(self.feat_dim)
        else:
            self.norm = None

        head_type = str(head_type).lower().strip()
        self.head_type = head_type

        if head_type == "linear":
            self.head = nn.Linear(self.feat_dim, int(num_classes), bias=False)
        elif head_type == "zzz":
            self.head = ZZZHead(num_classes=int(num_classes), feat_dim=self.feat_dim, rank=int(zzz_rank))
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        if self.norm is not None:
            z = self.norm(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_features(x)
        logits = self.head(z)
        return logits
