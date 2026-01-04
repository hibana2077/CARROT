from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class FlopsResult:
    flops: Optional[float]  # total FLOPs
    params: Optional[float]  # total parameters


def _parse_ptflops_number(s: str) -> Optional[float]:
    """Parse strings like '17.61 GMac' / '86.4 MMac' to a float (MACs)."""
    if not s:
        return None

    parts = s.strip().split()
    if not parts:
        return None

    try:
        val = float(parts[0])
    except ValueError:
        return None

    unit = parts[1] if len(parts) > 1 else ""
    unit = unit.lower()

    # ptflops typically reports MACs; we treat them as FLOPs-like proxy.
    if unit.startswith("g"):
        return val * 1e9
    if unit.startswith("m"):
        return val * 1e6
    if unit.startswith("k"):
        return val * 1e3
    return val


@torch.no_grad()
def compute_flops_params_ptflops(
    model: nn.Module,
    img_size: int,
    device: torch.device,
) -> FlopsResult:
    """Compute FLOPs/params using ptflops.

    Notes:
    - This is a best-effort proxy; some ops/models may be partially unsupported.
    - We run on CPU by default for stability with ptflops hooks.
    """

    try:
        from ptflops import get_model_complexity_info
    except Exception:
        return FlopsResult(flops=None, params=None)

    # ptflops is most reliable on CPU
    model_cpu = model.to("cpu").eval()

    try:
        macs_str, params_str = get_model_complexity_info(
            model_cpu,
            (3, int(img_size), int(img_size)),
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
    except Exception:
        return FlopsResult(flops=None, params=None)
    finally:
        # move back
        model.to(device)

    macs = _parse_ptflops_number(macs_str)
    params = _parse_ptflops_number(params_str)

    return FlopsResult(flops=macs, params=params)
