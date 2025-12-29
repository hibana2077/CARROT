from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import timm


@dataclass
class PatchOutput:
    H: torch.Tensor   # (B, N, D) patch tokens
    P: torch.Tensor   # (N, 2) patch centers (normalized)
    grid_hw: Tuple[int, int]  # (gh, gw)
    patch_hw: Tuple[int, int] # (ph, pw)


class TimmViTPatchBackbone(nn.Module):
    """
    A thin wrapper that extracts patch tokens from timm ViT-like models robustly.
    - returns patch tokens H (no cls/dist tokens)
    - returns patch centers P aligned with the patch order
    """
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        img_size: Optional[int] = None,
        freeze: bool = True,
        out_norm: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # timm create_model: num_classes=0 removes classifier head for many models
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,          # some models accept it; safe to pass if you want fixed size
        )

        # Many timm ViT-like models expose these; we keep fallbacks
        self.out_norm = out_norm

        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

        if device is not None:
            self.model.to(device)

    def forward(self, x: torch.Tensor) -> PatchOutput:
        """
        x: (B, 3, H, W)
        """
        # 1) forward_features usually returns tokens (B, T, D) or a pooled vector depending on model.
        tokens = self._forward_tokens(x)  # (B, T, D)

        # 2) drop prefix tokens (cls/dist/others)
        patch_tokens = self._strip_prefix_tokens(tokens)  # (B, N, D)

        # 3) compute grid from input size & patch size
        gh, gw, ph, pw = self._infer_grid_and_patch(x, patch_tokens)

        # 4) optionally apply a normalization (some models already do this in forward_features)
        # We keep it optional to avoid double-normalizing.
        if self.out_norm and hasattr(self.model, "norm") and isinstance(self.model.norm, nn.Module):
            patch_tokens = self.model.norm(patch_tokens)

        # 5) patch centers positions (N, 2), normalized to [-1, 1]
        P = self._make_patch_centers(gh, gw, device=patch_tokens.device, dtype=patch_tokens.dtype)

        return PatchOutput(
            H=patch_tokens,
            P=P,
            grid_hw=(gh, gw),
            patch_hw=(ph, pw),
        )

    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Try best-effort methods to get token sequence from timm models.
        """
        # Most ViT-like models in timm implement forward_features(x) -> (B, T, D)
        if hasattr(self.model, "forward_features"):
            y = self.model.forward_features(x)
        else:
            # fallback to forward; may return pooled vector; we try to avoid this
            y = self.model(x)

        # Some models might return a tuple/list; take first tensor that looks like tokens
        if isinstance(y, (tuple, list)):
            for t in y:
                if torch.is_tensor(t) and t.dim() == 3:
                    return t
            raise RuntimeError("Cannot find token tensor (B, T, D) in model output tuple/list.")

        if torch.is_tensor(y) and y.dim() == 3:
            return y

        raise RuntimeError(
            "Model did not return tokens (B, T, D). "
            "Try a different timm model (ViT/DeiT/BEiT-style token models)."
        )

    def _strip_prefix_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Remove prefix tokens like CLS/DIST.
        timm ViT-like models often expose num_prefix_tokens.
        """
        n_prefix = None

        # timm VisionTransformer / DeiT commonly has num_prefix_tokens
        if hasattr(self.model, "num_prefix_tokens"):
            n_prefix = int(getattr(self.model, "num_prefix_tokens"))

        # fallback heuristics
        if n_prefix is None:
            # many ViT have cls_token attribute
            n_prefix = 1 if hasattr(self.model, "cls_token") else 0
            # DeiT dist token
            if hasattr(self.model, "dist_token"):
                n_prefix += 1

        if n_prefix == 0:
            return tokens

        if tokens.size(1) <= n_prefix:
            raise RuntimeError(f"Token length {tokens.size(1)} <= prefix tokens {n_prefix}.")

        return tokens[:, n_prefix:, :]

    def _infer_grid_and_patch(
        self, x: torch.Tensor, patch_tokens: torch.Tensor
    ) -> Tuple[int, int, int, int]:
        """
        Infer patch grid (gh, gw) and patch size (ph, pw).
        """
        B, _, H, W = x.shape
        N = patch_tokens.size(1)

        ph = pw = None
        if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "patch_size"):
            ps = self.model.patch_embed.patch_size
            # patch_size could be int or tuple
            if isinstance(ps, tuple):
                ph, pw = int(ps[0]), int(ps[1])
            else:
                ph = pw = int(ps)

        # Prefer explicit grid_size if available
        gh = gw = None
        if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "grid_size"):
            gs = self.model.patch_embed.grid_size
            if isinstance(gs, tuple):
                gh, gw = int(gs[0]), int(gs[1])

        # If grid_size missing or doesn't match N, derive from input size (more robust for dynamic resize)
        if ph is not None and pw is not None:
            gh2, gw2 = H // ph, W // pw
            if gh2 * gw2 == N:
                return gh2, gw2, ph, pw

        # last resort: assume square grid
        s = int(N ** 0.5)
        if s * s != N:
            raise RuntimeError(
                f"Cannot infer grid: N={N} not a perfect square, and input-derived grid failed. "
                f"Try ensuring input size divisible by patch size, or use a model exposing patch_embed.grid_size."
            )
        return s, s, (ph or -1), (pw or -1)

    def _make_patch_centers(
        self, gh: int, gw: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Create patch centers aligned with token order:
        Typically row-major order: (y=0,x=0..gw-1), (y=1, ...)
        Output normalized to [-1, 1] in both axes.
        """
        ys = (torch.arange(gh, device=device, dtype=dtype) + 0.5) / gh  # (gh,)
        xs = (torch.arange(gw, device=device, dtype=dtype) + 0.5) / gw  # (gw,)

        # meshgrid: y first then x to match row-major flatten
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (gh, gw)
        P = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (N, 2), in [0,1]

        # map [0,1] -> [-1,1]
        P = P * 2.0 - 1.0
        return P
