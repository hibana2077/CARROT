from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

import timm

try:
    # When running as a script: `python src/main.py`
    from graph.dsot import DSOTGraphBuilder
    from graph.features import timm_forward_to_nodes
    from graph.head import GraphClassifier
except ModuleNotFoundError:
    # When running as a module: `python -m src.main`
    from ..graph.dsot import DSOTGraphBuilder
    from ..graph.features import timm_forward_to_nodes
    from ..graph.head import GraphClassifier


@dataclass
class GraphModelConfig:
    dsot_k: int = 16
    dsot_eps: float = 0.10
    dsot_sinkhorn_iters: int = 20
    dsot_lambda_pos: float = 0.10
    dsot_self_loop_alpha: float = 0.20
    dsot_cost_normalize: bool = True

    gnn_hidden_dim: int = 256
    gnn_layers: int = 2
    gnn_dropout: float = 0.0


class TimmDSOTGraphModel(nn.Module):
    """Backbone-agnostic timm model + DSOT graph builder + PyG head.

    Returns (logits, z) where z is graph embedding.

    Supports:
      - CNN features (B,C,H,W)
      - ViT/Swin tokens (B,N,D) ; prefix tokens (CLS) are dropped when detectable
    """

    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool, cfg: GraphModelConfig) -> None:
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.num_classes = int(num_classes)

        self.dsot = DSOTGraphBuilder(
            k=cfg.dsot_k,
            eps=cfg.dsot_eps,
            sinkhorn_iters=cfg.dsot_sinkhorn_iters,
            lambda_pos=cfg.dsot_lambda_pos,
            self_loop_alpha=cfg.dsot_self_loop_alpha,
            cost_normalize=cfg.dsot_cost_normalize,
        )

        # Head init is deferred until we see feature dim
        self._head: GraphClassifier | None = None
        self._head_cfg = cfg

    def _ensure_head(self, in_dim: int, device: torch.device) -> None:
        if self._head is not None:
            # In case the module was created before .to(device) or under a different device context
            self._head.to(device)
            return
        self._head = GraphClassifier(
            in_dim=int(in_dim),
            num_classes=self.num_classes,
            hidden_dim=int(self._head_cfg.gnn_hidden_dim),
            num_layers=int(self._head_cfg.gnn_layers),
            dropout=float(self._head_cfg.gnn_dropout),
        )
        self._head.to(device)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        nodes = timm_forward_to_nodes(self.backbone, images)
        x = nodes.x
        pos = nodes.pos

        self._ensure_head(in_dim=int(x.size(-1)), device=x.device)
        assert self._head is not None

        edge_index, edge_weight, batch = self.dsot(x, pos)
        x_nodes = x.reshape(-1, x.size(-1))
        logits, g = self._head(x_nodes, edge_index, edge_weight, batch)
        return logits, g
