from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

import timm


class FGModel(nn.Module):
    """Backbone + normalization + classifier head.

    Returns logits and pooled features (z) for CARROT regularization.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        # timm: create model with no classifier -> calling model(x) returns pooled features
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

        self.head = nn.Linear(self.feat_dim, int(num_classes))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        if self.norm is not None:
            z = self.norm(z)
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.forward_features(x)
        logits = self.head(z)
        return logits, z
