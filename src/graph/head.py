from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class GraphHeadConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0


class GraphClassifier(nn.Module):
    """Simple PyG head that consumes edge_weight.

    Uses GraphConv which supports edge weights.
    """

    def __init__(self, in_dim: int, num_classes: int, *, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        try:
            from torch_geometric.nn import GraphConv
        except Exception as e:
            raise RuntimeError(
                "torch-geometric is required for graph mode. Ensure torch-geometric is installed for your PyTorch/CUDA version."
            ) from e

        self.proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(max(1, self.num_layers)):
            self.convs.append(GraphConv(self.hidden_dim, self.hidden_dim))

        self.cls = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x_nodes: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        from torch_geometric.nn import global_mean_pool

        h = self.proj(x_nodes)
        for conv in self.convs:
            h = conv(h, edge_index, edge_weight)
            h = torch.relu(h)
            if self.dropout > 0:
                h = torch.dropout(h, p=self.dropout, train=self.training)

        g = global_mean_pool(h, batch)
        logits = self.cls(g)
        return logits, g
