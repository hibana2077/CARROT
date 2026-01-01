from __future__ import annotations

from typing import Any

import torch


def extract_timm_features(model: Any, images: torch.Tensor) -> torch.Tensor:
    """Extract a pooled feature vector from a timm model.

    Goal: return shape [B, d] suitable for CARROT prototypes.

    We try the most common timm APIs in order:
    - forward_features + forward_head(pre_logits=True)
    - forward_features returning already pooled vectors
    - fallback: flatten / global-average-pool
    """
    if hasattr(model, "forward_features"):
        feats = model.forward_features(images)
    else:
        raise ValueError("Model does not expose forward_features; cannot extract features for CARROT")

    # Preferred: let timm handle pooling & pre-logits projection
    if hasattr(model, "forward_head"):
        try:
            pooled = model.forward_head(feats, pre_logits=True)
            if pooled.ndim == 2:
                return pooled
        except TypeError:
            # Some models may not accept pre_logits kwarg
            pass

    # If it's already a vector
    if feats.ndim == 2:
        return feats

    # Common cases: [B, C, H, W] or [B, N, C]
    if feats.ndim == 4:
        # [B, C, H, W] -> GAP
        return feats.mean(dim=(2, 3))

    if feats.ndim == 3:
        # [B, N, C] -> mean token
        return feats.mean(dim=1)

    # Fallback: flatten everything but batch
    return feats.flatten(start_dim=1)
