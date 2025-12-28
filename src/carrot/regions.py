from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class RegionSet:
    features: Tensor  # H: (B, N, D)
    positions: Tensor # P: (N, 2)

    def to(self, device: torch.device):
        self.features = self.features.to(device)
        self.positions = self.positions.to(device)
        return self
