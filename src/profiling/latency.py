from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LatencyResult:
    ms_per_image: Optional[float]


@torch.no_grad()
def measure_latency_ms_per_image(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    warmup: int = 10,
    iters: int = 30,
) -> LatencyResult:
    """Measure inference latency (ms/image) on the provided batch."""

    if images.numel() == 0:
        return LatencyResult(ms_per_image=None)

    model.eval()
    images = images.to(device, non_blocking=True)

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # warmup
    for _ in range(int(warmup)):
        _ = model(images)
    _sync()

    t0 = time.perf_counter()
    for _ in range(int(iters)):
        _ = model(images)
    _sync()
    t1 = time.perf_counter()

    total_s = t1 - t0
    total_images = int(images.size(0)) * int(iters)
    if total_images <= 0:
        return LatencyResult(ms_per_image=None)

    ms_per_image = (total_s * 1000.0) / float(total_images)
    return LatencyResult(ms_per_image=float(ms_per_image))
