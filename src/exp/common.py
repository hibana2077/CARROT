from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc1: float
    val_loss: float
    val_acc1: float
    lr: float
    seconds: float
    alpha_mean: Optional[float] = None
    alpha_min: Optional[float] = None
    alpha_max: Optional[float] = None
    alpha_ess: Optional[float] = None


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.numel())


def avg_meter_update(meter: Dict[str, float], key: str, value_sum: float) -> None:
    meter[key] = meter.get(key, 0.0) + float(value_sum)


def avg_meter_compute(meter: Dict[str, float], total_count: int) -> Dict[str, float]:
    denom = max(1, int(total_count))
    return {k: v / denom for k, v in meter.items()}


def compute_alpha_stats(alpha: torch.Tensor) -> Dict[str, float]:
    a = alpha.detach().float().cpu()
    eps = 1e-12
    ess = (a.sum().item() ** 2) / (a.pow(2).sum().item() + eps)
    return {
        "alpha_mean": float(a.mean().item()),
        "alpha_min": float(a.min().item()),
        "alpha_max": float(a.max().item()),
        "alpha_ess": float(ess),
    }


def write_csv_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
