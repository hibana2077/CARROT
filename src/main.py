import argparse
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import timm

try:
    # When running as a script: `python src/main.py`
    from carrot import CARROT, grad_balanced_total_loss  # type: ignore[import-not-found]
except ModuleNotFoundError:
    # When running as a module: `python -m src.main`
    from .carrot import CARROT, grad_balanced_total_loss

try:
    # When running as a script: `python src/main.py`
    from dataset.ufgvc import UFGVCDataset
    from models import FGModel  # type: ignore[import-not-found]
except ModuleNotFoundError:
    # When running as a module: `python -m src.main`
    from .dataset.ufgvc import UFGVCDataset
    from .models import FGModel  # type: ignore[import-not-found]


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
    intra_sim: Optional[float] = None
    concentration: Optional[float] = None
    carrot_L: Optional[float] = None
    carrot_U: Optional[float] = None
    carrot_gamma_mean: Optional[float] = None
    carrot_gamma_max: Optional[float] = None
    carrot_frac_gamma_gt_1: Optional[float] = None
    carrot_r_mean: Optional[float] = None
    carrot_m_mean: Optional[float] = None
    carrot_alpha: Optional[float] = None
    carrot_reg: Optional[float] = None


def _top1_correct(logits: torch.Tensor, targets: torch.Tensor) -> int:
    preds = torch.argmax(logits, dim=1)
    return int((preds == targets).sum().item())


def train_one_epoch(
    model: FGModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    carrot: Optional[CARROT] = None,
) -> EpochStats:
    model.train()

    total_samples = 0
    total_loss = 0.0
    total_correct = 0

    carrot_batches = 0
    carrot_reg_sum = 0.0
    carrot_reg_batches = 0
    carrot_sum: Dict[str, float] = {
        "gamma_mean": 0.0,
        "gamma_max": 0.0,
        "frac_gamma_gt_1": 0.0,
        "r_mean": 0.0,
        "m_mean": 0.0,
        "alpha": 0.0,
    }

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, z = model(images)
        loss_base = criterion(logits, targets)

        if carrot is not None:
            # CARROT operator (imp.md): expand within-class clouds using in-batch geometry.
            z_plus, stats = carrot(z, targets)
            logits_plus = model.head(z_plus)

            # Logit consistency regularization (imp.md): KL( p(z) || p(z_plus) )
            T = 1.0
            log_p = F.log_softmax(logits / T, dim=1)
            q = F.softmax(logits_plus / T, dim=1)
            reg = F.kl_div(log_p, q, reduction="batchmean") * (T * T)

            loss, alpha = grad_balanced_total_loss(loss_base, reg, z)

            carrot_reg_sum += float(reg.detach().item())
            carrot_reg_batches += 1

            if stats.classes_in_batch >= 2 and stats.gamma_mean is not None:
                carrot_batches += 1
                carrot_sum["gamma_mean"] += float(stats.gamma_mean or 0.0)
                carrot_sum["gamma_max"] += float(stats.gamma_max or 0.0)
                carrot_sum["frac_gamma_gt_1"] += float(stats.frac_gamma_gt_1 or 0.0)
                carrot_sum["r_mean"] += float(stats.r_mean or 0.0)
                carrot_sum["m_mean"] += float(stats.m_mean or 0.0)
                carrot_sum["alpha"] += float(alpha.item())
        else:
            loss = loss_base

        batch_size = targets.size(0)

        loss.backward()
        optimizer.step()

        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_correct += _top1_correct(logits, targets)

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)

    out = EpochStats(loss=avg_loss, acc=avg_acc)
    if carrot is not None and carrot_batches > 0:
        denom = float(carrot_batches)
        out.carrot_gamma_mean = carrot_sum["gamma_mean"] / denom
        out.carrot_gamma_max = carrot_sum["gamma_max"] / denom
        out.carrot_frac_gamma_gt_1 = carrot_sum["frac_gamma_gt_1"] / denom
        out.carrot_r_mean = carrot_sum["r_mean"] / denom
        out.carrot_m_mean = carrot_sum["m_mean"] / denom
        out.carrot_alpha = carrot_sum["alpha"] / denom
    if carrot is not None and carrot_reg_batches > 0:
        out.carrot_reg = carrot_reg_sum / float(carrot_reg_batches)
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
    """
    Compute intra-class similarity and estimated concentration (kappa).
    features: (N, D)
    targets: (N,)
    """
    # Normalize features to unit sphere for cosine similarity and vMF estimation
    features = F.normalize(features, p=2, dim=1)
    
    unique_classes = torch.unique(targets)
    
    intra_sims = []
    concentrations = []
    
    for c in unique_classes:
        mask = (targets == c)
        feats_c = features[mask]
        if feats_c.size(0) < 2:
            continue
            
        # Mean resultant vector
        R = feats_c.mean(dim=0)
        R_norm = R.norm().item()
        
        # Intra-class similarity (mean cosine similarity to mean direction)
        intra_sims.append(R_norm)
        
        # Concentration estimation (approximate kappa for vMF)
        # kappa approx (R * D - R^3) / (1 - R^2)
        D = feats_c.size(1)
        if R_norm >= 0.999:
            kappa = 100.0 # Cap it to avoid overflow
        else:
            kappa = (R_norm * D - R_norm**3) / (1 - R_norm**2)
        concentrations.append(kappa)
        
    if not intra_sims:
        return 0.0, 0.0
        
    return float(np.mean(intra_sims)), float(np.mean(concentrations))


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

    all_features = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits, z = model(images)
        loss = criterion(logits, targets)
        nll_sum = float(F.cross_entropy(logits, targets, reduction="sum").item())
        ece = float(_ece_from_logits(logits, targets).item())

        batch_size = targets.size(0)
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_correct += _top1_correct(logits, targets)
        total_nll += nll_sum
        total_ece_weighted += ece * batch_size

        all_features.append(z.cpu())
        all_targets.append(targets.cpu())

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    avg_nll = total_nll / max(1, total_samples)
    avg_ece = total_ece_weighted / max(1, total_samples)

    # Compute concentration metrics
    if all_features:
        all_features = torch.cat(all_features, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        intra_sim, concentration = compute_concentration_metrics(all_features, all_targets)
    else:
        intra_sim, concentration = 0.0, 0.0

    return EpochStats(loss=avg_loss, acc=avg_acc, nll=avg_nll, ece=avg_ece, intra_sim=intra_sim, concentration=concentration)


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

    p.add_argument("--carrot", action="store_true", help="Enable CARROT regularizer")
    # Old CARROT args kept for CLI compatibility (no longer used by operator-based CARROT).
    p.add_argument("--carrot_q_hi", type=float, default=0.90)
    p.add_argument("--carrot_q_lo", type=float, default=0.10)
    p.add_argument("--carrot_k", type=int, default=4, help="Images per class (K) for PK batches")

    return p.parse_args()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    set_seed(args.seed)

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
    if args.carrot:
        # CARROT needs PK batches to estimate per-class stats reliably (imp.md §0).
        if args.batch_size % int(args.carrot_k) != 0:
            raise ValueError(
                f"--batch_size must be a multiple of --carrot_k for PK sampling. "
                f"Got batch_size={args.batch_size}, carrot_k={args.carrot_k}."
            )
        try:
            from pytorch_metric_learning import samplers  # type: ignore

            train_labels = train_ds.get_all_labels().tolist()
            sampler = samplers.MPerClassSampler(
                train_labels, m=int(args.carrot_k), batch_size=int(args.batch_size)
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                drop_last=True,
            )
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "CARROT requires pytorch-metric-learning for PK sampling. "
                "Please install it (see requirements.txt)."
            ) from e
    else:
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

    carrot = None
    if args.carrot:
        carrot = CARROT().to(device)
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
            carrot=carrot,
        )
        eval_stats = evaluate(model, eval_loader, device, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        # Required format (no tqdm; accuracy is epoch-average)
        msg = (
            f"Epoch {epoch} - train_acc {train_stats.acc:.4f} - "
            f"test_acc {eval_stats.acc:.4f} - "
            f"test_nll {float(eval_stats.nll):.4f} - "
            f"test_ece {float(eval_stats.ece):.4f} - "
            f"test_intra_sim {float(eval_stats.intra_sim):.4f} - "
            f"test_concentration {float(eval_stats.concentration):.4f}"
        )

        # imp.md §6 required CARROT logging metrics; append after existing metrics.
        if args.carrot:
            msg += (
                f" - train_carrot_gamma_mean {float(train_stats.carrot_gamma_mean or 0.0):.4f}"
                f" - train_carrot_gamma_max {float(train_stats.carrot_gamma_max or 0.0):.4f}"
                f" - train_carrot_frac_gamma_gt_1 {float(train_stats.carrot_frac_gamma_gt_1 or 0.0):.4f}"
                f" - train_carrot_r_mean {float(train_stats.carrot_r_mean or 0.0):.4f}"
                f" - train_carrot_m_mean {float(train_stats.carrot_m_mean or 0.0):.4f}"
                f" - train_carrot_alpha {float(train_stats.carrot_alpha or 0.0):.4f}"
                f" - train_carrot_reg {float(train_stats.carrot_reg or 0.0):.6f}"
            )

        msg += f" - {elapsed:.1f} seconds"
        print(msg, flush=True)

        lr = optimizer.param_groups[0]["lr"]
        print(f"lr {lr:.6f}", flush=True)


if __name__ == "__main__":
    main()
