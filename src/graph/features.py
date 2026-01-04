from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class NodeFeatures:
    x: torch.Tensor  # (B,N,D)
    pos: torch.Tensor  # (N,2)


def _unwrap_features(feats):
    if isinstance(feats, (tuple, list)) and len(feats) > 0:
        return feats[0]
    return feats


def _infer_hw_from_n(n: int) -> Tuple[int, int]:
    # best-effort square grid
    s = int(round(n ** 0.5))
    if s * s == n:
        return s, s
    # fallback: rectangular close to square
    for h in range(s, 0, -1):
        if n % h == 0:
            return h, n // h
    return 1, n


def timm_forward_to_nodes(backbone: nn.Module, images: torch.Tensor) -> NodeFeatures:
    """Run timm model forward_features and convert to node features.

    - CNN-like: feats (B,C,H,W) -> x (B, H*W, C)
    - ViT-like: feats (B,N,D) -> drop prefix tokens (CLS) when detectable

    Positions are a normalized (y,x) grid in [0,1].
    """

    feats = backbone.forward_features(images) if hasattr(backbone, "forward_features") else backbone(images)
    feats = _unwrap_features(feats)

    if not isinstance(feats, torch.Tensor):
        raise TypeError("Unsupported features type from timm backbone")

    if feats.dim() == 4:
        # (B,C,H,W)
        B, C, H, W = feats.shape
        x = feats.flatten(2).transpose(1, 2)  # (B,N,C)
        pos = _make_grid_pos(int(H), int(W), device=feats.device, dtype=feats.dtype)
        return NodeFeatures(x=x, pos=pos)

    if feats.dim() == 3:
        # (B,N,D)
        B, N, D = feats.shape

        num_prefix = int(getattr(backbone, "num_prefix_tokens", 0) or 0)
        if num_prefix == 0 and hasattr(backbone, "cls_token"):
            num_prefix = 1

        # Drop prefix tokens if present
        if num_prefix > 0 and N > num_prefix:
            feats = feats[:, num_prefix:, :]
            N = feats.size(1)

        # Determine token grid size
        H = W = None
        patch_embed = getattr(backbone, "patch_embed", None)
        grid = getattr(patch_embed, "grid_size", None) if patch_embed is not None else None
        if grid is not None and len(grid) == 2:
            H, W = int(grid[0]), int(grid[1])
            if H * W != N:
                H, W = _infer_hw_from_n(int(N))
        else:
            H, W = _infer_hw_from_n(int(N))

        x = feats  # (B,N,D)
        pos = _make_grid_pos(int(H), int(W), device=feats.device, dtype=feats.dtype)
        # In rare cases N != H*W, clip/extend positions by slicing
        if pos.size(0) != x.size(1):
            pos = pos[: x.size(1)]
        return NodeFeatures(x=x, pos=pos)

    if feats.dim() == 2:
        # (B,D) -> single node
        B, D = feats.shape
        x = feats[:, None, :]
        pos = torch.zeros((1, 2), device=feats.device, dtype=feats.dtype)
        return NodeFeatures(x=x, pos=pos)

    raise ValueError(f"Unsupported feature tensor shape: {tuple(feats.shape)}")


def _make_grid_pos(H: int, W: int, device=None, dtype=torch.float32) -> torch.Tensor:
    ys = torch.linspace(0.0, 1.0, steps=int(H), device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, steps=int(W), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)
    return pos
