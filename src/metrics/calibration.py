from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def ece_from_logits(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> torch.Tensor:
    """Expected Calibration Error (ECE) for a batch.

    Returns a scalar tensor on the same device as logits.
    """

    probs = F.softmax(logits, dim=1)
    conf, preds = torch.max(probs, dim=1)
    correct = preds.eq(targets).float()

    # bins in (0,1]; include 0 in first bin by using >= on left edge
    bin_boundaries = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=logits.device)
    ece = torch.zeros((), device=logits.device)
    N = conf.numel()
    if N == 0:
        return ece

    for b in range(n_bins):
        lo = bin_boundaries[b]
        hi = bin_boundaries[b + 1]
        if b == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)

        cnt = mask.sum()
        if int(cnt.item()) == 0:
            continue
        avg_conf = conf[mask].mean()
        avg_acc = correct[mask].mean()
        ece = ece + (cnt.float() / float(N)) * torch.abs(avg_acc - avg_conf)

    return ece
