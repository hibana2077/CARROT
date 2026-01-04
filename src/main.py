import argparse
import random
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import timm

try:
    # When running as a script: `python src/main.py`
    from dataset.ufgvc import UFGVCDataset
    from models.timm_fg import TimmFGModel
    from profiling.flops import compute_flops_params_ptflops
    from profiling.latency import measure_latency_ms_per_image
    from train_eval import evaluate, train_one_epoch
except ModuleNotFoundError:
    # When running as a module: `python -m src.main`
    from .dataset.ufgvc import UFGVCDataset
    from .models.timm_fg import TimmFGModel
    from .profiling.flops import compute_flops_params_ptflops
    from .profiling.latency import measure_latency_ms_per_image
    from .train_eval import evaluate, train_one_epoch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    # Inference profiling (best-effort): FLOPs (ptflops) + latency (ms/image)
    flops_res = compute_flops_params_ptflops(model.backbone, img_size=int(args.img_size), device=device)
    flops_g = (flops_res.flops / 1e9) if (flops_res.flops is not None) else None

    latency_ms_img = None
    try:
        sample_images, _sample_targets = next(iter(eval_loader))
        latency_res = measure_latency_ms_per_image(model, sample_images, device=device)
        latency_ms_img = latency_res.ms_per_image
    except Exception:
        latency_ms_img = None

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
            f"test_macro_f1 {float(getattr(eval_stats, 'macro_f1', 0.0) or 0.0):.4f} - "
            f"test_balanced_acc {float(eval_stats.balanced_acc or 0.0):.4f} - "
            f"test_top_k_conf_pairs_error {float(eval_stats.top_k_conf_pairs_error or 0.0):.4f} - "
            f"test_nll {float(eval_stats.nll):.4f} - "
            f"test_ece {float(eval_stats.ece):.4f} - "
            f"test_intra_sim {float(eval_stats.intra_sim):.4f} - "
            f"test_concentration {float(eval_stats.concentration):.4f} - "
            f"infer_flops_gmac {('NA' if flops_g is None else f'{flops_g:.3f}') } - "
            f"infer_latency_ms_per_img {('NA' if latency_ms_img is None else f'{latency_ms_img:.3f}') }"
        )

        msg += f" - {elapsed:.1f} seconds"
        print(msg, flush=True)

        lr = optimizer.param_groups[0]["lr"]
        print(f"lr {lr:.6f}", flush=True)


if __name__ == "__main__":
    main()
