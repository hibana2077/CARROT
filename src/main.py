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
    from models import FGModel  # type: ignore[import-not-found]
    from carrot import carrot_regularizer
except ModuleNotFoundError:
    # When running as a module: `python -m src.main`
    from .dataset.ufgvc import UFGVCDataset
    from .models import FGModel  # type: ignore[import-not-found]
    from .carrot import carrot_regularizer


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
    nll: Optional[float] = None
    ece: Optional[float] = None


def _top1_correct(logits: torch.Tensor, targets: torch.Tensor) -> int:
    preds = torch.argmax(logits, dim=1)
    return int((preds == targets).sum().item())


def train_one_epoch(
    model: FGModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    use_carrot: bool = False,
) -> EpochStats:
    model.train()

    total_samples = 0
    total_loss = 0.0
    total_correct = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, z = model(images)
        loss = criterion(logits, targets)

        if use_carrot:
            reg, _stats = carrot_regularizer(z, targets)
            loss = loss + reg

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_correct += _top1_correct(logits, targets)

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return EpochStats(loss=avg_loss, acc=avg_acc)


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


@torch.no_grad()
def evaluate(
    model: FGModel,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> EpochStats:
    model.eval()

    total_samples = 0
    total_loss = 0.0
    total_correct = 0
    total_nll = 0.0
    total_ece_weighted = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits, _ = model(images)
        loss = criterion(logits, targets)
        nll_sum = float(F.cross_entropy(logits, targets, reduction="sum").item())
        ece = float(_ece_from_logits(logits, targets).item())

        batch_size = targets.size(0)
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_correct += _top1_correct(logits, targets)
        total_nll += nll_sum
        total_ece_weighted += ece * batch_size

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    avg_nll = total_nll / max(1, total_samples)
    avg_ece = total_ece_weighted / max(1, total_samples)
    return EpochStats(loss=avg_loss, acc=avg_acc, nll=avg_nll, ece=avg_ece)


def build_transforms(model: nn.Module, img_size: int):
    """Prefer timm transforms; fall back to torchvision if timm API changes."""
    try:
        from timm.data import create_transform, resolve_data_config

        data_config = resolve_data_config({}, model=model)
        data_config["input_size"] = (3, img_size, img_size)

        train_tf = create_transform(**data_config, is_training=True)
        eval_tf = create_transform(**data_config, is_training=False)
        return train_tf, eval_tf
    except Exception:
        from torchvision import transforms

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        train_tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        eval_tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        return train_tf, eval_tf


def pick_eval_split(dataset_name: str, data_root: str) -> str:
    splits = UFGVCDataset.get_dataset_splits(dataset_name, root=data_root)
    splits = [str(s) for s in splits]
    if "test" in splits:
        return "test"
    if "val" in splits:
        return "val"
    return "test"


def build_eval_dataset(
    dataset_name: str,
    data_root: str,
    transform,
    download: bool,
) -> Tuple[UFGVCDataset, str]:
    """Prefer 'test' split; fall back to 'val' if 'test' doesn't exist."""
    # If we can discover splits, respect that; otherwise try common names.
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

    # CARROT
    p.add_argument("--use_carrot", action="store_true", help="Enable CARROT regularization")

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

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a dummy backbone model so timm can resolve default data config.
    dummy_backbone = timm.create_model(args.model, pretrained=False, num_classes=0)
    train_tf, eval_tf = build_transforms(dummy_backbone, img_size=args.img_size)

    train_ds = UFGVCDataset(
        dataset_name=args.dataset,
        root=args.data_root,
        split="train",
        transform=train_tf,
        download=not args.no_download,
    )

    num_classes = len(train_ds.classes)
    model = FGModel(
        backbone_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
    )
    model.to(device)

    # Rebuild transforms with the actual model when possible (safe if it stays same)
    train_tf, eval_tf = build_transforms(model.backbone, img_size=args.img_size)
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            use_carrot=args.use_carrot,
        )
        eval_stats = evaluate(model, eval_loader, device, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        # Required format (no tqdm; accuracy is epoch-average)
        print(
            f"Epoch {epoch} - train_acc {train_stats.acc:.4f} - "
            f"test_acc {eval_stats.acc:.4f} - "
            f"test_nll {float(eval_stats.nll):.4f} - "
            f"test_ece {float(eval_stats.ece):.4f} - "
            f"{elapsed:.1f} seconds",
            flush=True,
        )

        lr = optimizer.param_groups[0]["lr"]
        print(f"lr {lr:.6f}", flush=True)


if __name__ == "__main__":
    main()
