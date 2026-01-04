import argparse
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import timm

try:
    # When running as a script: `python src/main.py`
    from dataset.ufgvc import UFGVCDataset
except ModuleNotFoundError:
    # When running as a module: `python -m src.main`
    from .dataset.ufgvc import UFGVCDataset


class TimmFGModel(nn.Module):
    """timm backbone + head wrapper.

    Returns (logits, z) where z is a per-sample feature vector used for evaluation metrics.
    """

    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        self.model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=int(num_classes))

        # Best-effort for downstream metrics
        self.num_classes = int(num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.model, "forward_features") and hasattr(self.model, "forward_head"):
            feats = self.model.forward_features(x)
            logits = self.model.forward_head(feats, pre_logits=False)
            z = self._to_feature_vector(feats)
            return logits, z

        logits = self.model(x)
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
            z = self._to_feature_vector(feats)
        else:
            # Fallback: at least keep shapes consistent
            z = logits.detach()
        return logits, z

    @staticmethod
    def _to_feature_vector(feats: torch.Tensor | Tuple | list) -> torch.Tensor:
        if isinstance(feats, (tuple, list)) and len(feats) > 0:
            feats = feats[0]
        if not isinstance(feats, torch.Tensor):
            raise TypeError("Unsupported features type returned by timm model")

        if feats.dim() == 4:
            # CNN: [B, C, H, W] -> GAP
            return feats.mean(dim=(2, 3))
        if feats.dim() == 3:
            # ViT: [B, N, C] -> CLS token
            return feats[:, 0]
        if feats.dim() == 2:
            return feats
        return feats.view(feats.size(0), -1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class EpochStats:
    loss: float
    acc: float
    balanced_acc: Optional[float] = None
    top_k_conf_pairs_error: Optional[float] = None
    bhat: Optional[float] = None
    bhat_scale: Optional[float] = None
    nll: Optional[float] = None
    ece: Optional[float] = None
    intra_sim: Optional[float] = None
    concentration: Optional[float] = None


def _top1_correct(logits: torch.Tensor, targets: torch.Tensor) -> int:
    preds = torch.argmax(logits, dim=1)
    return int((preds == targets).sum().item())


def train_one_epoch(
    model: TimmFGModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> EpochStats:
    model.train()

    total_samples = 0
    total_loss = 0.0
    total_correct = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, _z = model(images)
        ce = criterion(logits, targets)

        loss = ce

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_correct += _top1_correct(logits, targets)

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    out = EpochStats(loss=avg_loss, acc=avg_acc)
    return out


@torch.no_grad()
def _ece_from_logits(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> torch.Tensor:
    """Expected Calibration Error (ECE) for a batch."""
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


def compute_concentration_metrics(features: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    """Compute intra-class similarity and estimated concentration (kappa)."""
    features = F.normalize(features, p=2, dim=1)
    unique_classes = torch.unique(targets)

    intra_sims = []
    concentrations = []
    for c in unique_classes:
        mask = targets == c
        feats_c = features[mask]
        if feats_c.size(0) < 2:
            continue

        R = feats_c.mean(dim=0)
        R_norm = R.norm().item()
        intra_sims.append(R_norm)

        D = feats_c.size(1)
        if R_norm >= 0.999:
            kappa = 100.0
        else:
            kappa = (R_norm * D - R_norm**3) / (1 - R_norm**2)
        concentrations.append(kappa)

    if not intra_sims:
        return 0.0, 0.0
    return float(np.mean(intra_sims)), float(np.mean(concentrations))


def compute_balanced_accuracy(targets: torch.Tensor, preds: torch.Tensor) -> float:
    unique_classes = torch.unique(targets)
    recalls = []
    for c in unique_classes:
        mask = targets == c
        if int(mask.sum().item()) == 0:
            continue
        recalls.append(float((preds[mask] == c).float().mean().item()))
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def compute_top_k_confusion_pairs_error(
    targets: torch.Tensor,
    preds: torch.Tensor,
    num_classes: int,
    k: int = 20,
) -> float:
    """Error mass on the top-k most frequent off-diagonal confusion pairs."""
    if targets.numel() == 0:
        return 0.0

    targets = targets.to(torch.long)
    preds = preds.to(torch.long)

    indices = targets * int(num_classes) + preds
    cm_flat = torch.bincount(indices, minlength=int(num_classes) ** 2)
    cm = cm_flat.view(int(num_classes), int(num_classes))
    cm.fill_diagonal_(0)

    flat = cm.flatten()
    if flat.numel() == 0:
        return 0.0
    k = min(int(k), int(flat.numel()))
    top_vals, _ = torch.topk(flat, k=k, largest=True)
    return float(top_vals.sum().item() / float(targets.numel()))


@torch.no_grad()
def evaluate(
    model: TimmFGModel,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    top_k_pairs: int = 20,
) -> EpochStats:
    model.eval()

    total_samples = 0
    total_loss = 0.0
    total_correct = 0
    total_nll = 0.0
    total_ece_weighted = 0.0

    all_features = []
    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits, z = model(images)
        loss = criterion(logits, targets)
        nll_sum = float(F.cross_entropy(logits, targets, reduction="sum").item())
        ece = float(_ece_from_logits(logits, targets).item())
        preds = torch.argmax(logits, dim=1)

        batch_size = targets.size(0)
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == targets).sum().item())
        total_nll += nll_sum
        total_ece_weighted += ece * batch_size

        all_features.append(z.cpu())
        all_targets.append(targets.cpu())
        all_preds.append(preds.cpu())

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    avg_nll = total_nll / max(1, total_samples)
    avg_ece = total_ece_weighted / max(1, total_samples)

    intra_sim, concentration = 0.0, 0.0
    balanced_acc = 0.0
    top_k_conf_pairs_error = 0.0

    if all_features:
        feats = torch.cat(all_features, dim=0)
        ys = torch.cat(all_targets, dim=0)
        ps = torch.cat(all_preds, dim=0)

        intra_sim, concentration = compute_concentration_metrics(feats, ys)
        balanced_acc = compute_balanced_accuracy(ys, ps)
        num_classes = int(getattr(model, "num_classes", int(ps.max().item()) + 1))
        top_k_conf_pairs_error = compute_top_k_confusion_pairs_error(ys, ps, num_classes, k=top_k_pairs)

    return EpochStats(
        loss=avg_loss,
        acc=avg_acc,
        balanced_acc=balanced_acc,
        top_k_conf_pairs_error=top_k_conf_pairs_error,
        nll=avg_nll,
        ece=avg_ece,
        intra_sim=intra_sim,
        concentration=concentration,
    )


def build_transforms(model: nn.Module, img_size: int):
    """Use timm transforms only."""
    from timm.data import create_transform, resolve_data_config

    data_config = resolve_data_config({}, model=model)
    data_config["input_size"] = (3, img_size, img_size)
    train_tf = create_transform(**data_config, is_training=True)
    eval_tf = create_transform(**data_config, is_training=False)
    return train_tf, eval_tf


def build_eval_dataset(
    dataset_name: str,
    data_root: str,
    transform,
    download: bool,
) -> Tuple[UFGVCDataset, str]:
    """Prefer 'test' split; fall back to 'val' if 'test' doesn't exist."""
    splits = UFGVCDataset.get_dataset_splits(dataset_name, root=data_root)
    splits = [str(s) for s in splits] if splits else []

    candidates = []
    if "test" in splits:
        candidates.append("test")
    if "val" in splits:
        candidates.append("val")
    if not candidates:
        candidates = ["test", "val"]

    last_err: Optional[Exception] = None
    for split in candidates:
        try:
            ds = UFGVCDataset(
                dataset_name=dataset_name,
                root=data_root,
                split=split,
                transform=transform,
                download=download,
            )
            return ds, split
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to build eval dataset for splits {candidates}: {last_err}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UFGVC training (timm backbone, no tqdm)")

    p.add_argument("--dataset", type=str, default="soybean")
    p.add_argument("--data_root", type=str, default="./data")

    p.add_argument("--model", type=str, default="vit_base_patch16_224")
    p.add_argument("--pretrained", action="store_true")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--no_download", action="store_true")
    p.add_argument("--top_k_conf_pairs", type=int, default=20)

    return p.parse_args()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    set_seed(int(args.seed))

    dummy_backbone = timm.create_model(args.model, pretrained=False, num_classes=0)
    train_tf, eval_tf = build_transforms(dummy_backbone, img_size=int(args.img_size))

    train_ds = UFGVCDataset(
        dataset_name=args.dataset,
        root=args.data_root,
        split="train",
        transform=train_tf,
        download=not args.no_download,
    )

    num_classes = len(train_ds.classes)
    model = TimmFGModel(
        backbone_name=args.model,
        num_classes=num_classes,
        pretrained=bool(args.pretrained),
    )
    model.to(device)

    # Rebuild transforms with actual model (timm can resolve better defaults)
    train_tf, eval_tf = build_transforms(model.backbone, img_size=int(args.img_size))
    train_ds.transform = train_tf

    eval_ds, _eval_split = build_eval_dataset(
        dataset_name=args.dataset,
        data_root=args.data_root,
        transform=eval_tf,
        download=not args.no_download,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
    )

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(args.lr),
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs))

    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
        )
        eval_stats = evaluate(
            model,
            eval_loader,
            device,
            criterion,
            top_k_pairs=int(args.top_k_conf_pairs),
        )
        scheduler.step()
        elapsed = time.time() - t0

        msg = (
            f"Epoch {epoch} - train_acc {train_stats.acc:.4f} - "
            f"test_acc {eval_stats.acc:.4f} - "
            f"test_balanced_acc {float(eval_stats.balanced_acc or 0.0):.4f} - "
            f"test_top_k_conf_pairs_error {float(eval_stats.top_k_conf_pairs_error or 0.0):.4f} - "
            f"test_nll {float(eval_stats.nll):.4f} - "
            f"test_ece {float(eval_stats.ece):.4f} - "
            f"test_intra_sim {float(eval_stats.intra_sim):.4f} - "
            f"test_concentration {float(eval_stats.concentration):.4f}"
        )

        msg += f" - {elapsed:.1f} seconds"
        print(msg, flush=True)

        lr = optimizer.param_groups[0]["lr"]
        print(f"lr {lr:.6f}", flush=True)


if __name__ == "__main__":
    main()
