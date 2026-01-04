from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

import timm


class TimmFGModel(nn.Module):
    """timm backbone + head wrapper.

    Returns (logits, z) where z is a per-sample feature vector used for evaluation metrics.
    """

    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        self.model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=int(num_classes))

        # Alias for code that wants the underlying timm module
        self.backbone = self.model

        # Best-effort for downstream metrics
        self.num_classes = int(num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.model, "forward_features") and hasattr(self.model, "forward_head"):
            feats = self.model.forward_features(x)
            logits = self.model.forward_head(feats, pre_logits=False)
            z = self._to_feature_vector(feats)
            return logits, z

        logits = self.model(x)
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
            z = self._to_feature_vector(feats)
        else:
            # Fallback: at least keep shapes consistent
            z = logits.detach()
        return logits, z

    @staticmethod
    def _to_feature_vector(feats: torch.Tensor | Tuple | list) -> torch.Tensor:
        if isinstance(feats, (tuple, list)) and len(feats) > 0:
            feats = feats[0]
        if not isinstance(feats, torch.Tensor):
            raise TypeError("Unsupported features type returned by timm model")

        if feats.dim() == 4:
            # CNN: [B, C, H, W] -> GAP
            return feats.mean(dim=(2, 3))
        if feats.dim() == 3:
            # ViT: [B, N, C] -> CLS token
            return feats[:, 0]
        if feats.dim() == 2:
            return feats
        return feats.view(feats.size(0), -1)
