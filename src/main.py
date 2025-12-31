"""CARROT (Representer Reweighting) experiment runner.

This implements the training flow described in docs/idea.md and docs/imp.md:
- timm backbone (optionally warm-up / then freeze)
- L2-regularized linear head
- per-epoch averaged metric reporting (train + val)
- optional representer attribution (Top-K support/inhibit training samples)

Constraints (per user request):
- DO NOT use tqdm, wandb, peft, pytorch-lightning, accelerate
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import torch

# Allow imports when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.append(str(PROJECT_ROOT))

from src.exp.common import EpochMetrics, resolve_device, set_seed, write_csv_row
from src.exp.data import make_dataloaders
from src.exp.common import compute_alpha_stats
from src.exp.modeling import (
    create_backbone_and_head,
    make_optimizer,
    set_requires_grad,
    set_trainable_backbone_groups,
)
from src.exp.alpha import AlphaConfig, AlphaWeights
from src.exp.representer import compute_representer_values, extract_features, representer_topk
from src.exp.train import evaluate, train_one_epoch

from src.dataset.ufgvc import UFGVCDataset


def _alpha_enabled(args: argparse.Namespace) -> bool:
    return str(getattr(args, "alpha_mode", "none")).strip().lower() != "none"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CARROT representer/reweighting experiments (no tqdm/wandb).")

    # Reproducibility
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")

    # IO
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./runs")
    p.add_argument("--run_name", type=str, default="debug")
    p.add_argument("--download", action="store_true")

    # Dataset
    p.add_argument("--dataset", type=str, default="soybean")
    # Use val for model selection to avoid test leakage; data loader will fallback if unavailable.
    p.add_argument(
        "--val_split",
        type=str,
        default="val",
        choices=["val", "test", "train_split", "train"],
        help=(
            "Validation split to use. 'val' is default; if missing, loader falls back to 'test'. "
            "Use 'train_split' to carve validation from the training split."
        ),
    )
    p.add_argument(
        "--train_val_ratio",
        type=float,
        default=0.1,
        help="Only used when --val_split=train_split. Fraction of train held out for validation.",
    )
    p.add_argument(
        "--train_val_seed",
        type=int,
        default=None,
        help="Only used when --val_split=train_split. Random seed for the train/val split (defaults to --seed).",
    )
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    # Model
    p.add_argument("--model", type=str, default="vit_base_patch16_224")
    p.add_argument("--pretrained", action="store_true")

    # Training
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--warmup_epochs", type=int, default=0, help="Stage-0 warmup: train backbone+head")
    p.add_argument("--lr", type=float, default=1e-3, help="Head LR (and backbone LR in warmup)")
    p.add_argument("--backbone_lr", type=float, default=1e-5, help="Backbone LR in warmup stage")
    p.add_argument(
        "--backbone_lr_mult",
        type=float,
        default=0.0,
        help="If > 0, overrides --backbone_lr with (lr * backbone_lr_mult).",
    )
    p.add_argument("--weight_decay", type=float, default=0.0, help="Optimizer weight decay (separate from explicit head L2)")
    p.add_argument("--lambda_l2", type=float, default=1e-2, help="Explicit L2 regularization strength for head")
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])

    # CARROT alpha reweighting (head-only + 1-step unroll)
    p.add_argument(
        "--alpha_mode",
        type=str,
        default="none",
        choices=["none", "learn"],
        help=(
            "Enable CARROT sample reweighting. 'learn' uses differentiable alpha (softplus(s)) and updates alpha "
            "via 1-step unrolled bilevel (val-driven). Requires frozen backbone in focus stage (train_last_n=0)."
        ),
    )
    p.add_argument("--alpha_lr", type=float, default=1e-1, help="Learning rate for alpha parameters (s).")
    p.add_argument(
        "--alpha_inner_lr",
        type=float,
        default=None,
        help="Inner (unroll) LR for the simulated head step; defaults to --lr.",
    )
    p.add_argument(
        "--alpha_entropy_reg",
        type=float,
        default=0.0,
        help="Entropy regularizer strength for alpha (helps prevent collapse).",
    )
    p.add_argument(
        "--alpha_s_l2_reg",
        type=float,
        default=0.0,
        help="L2 regularizer on alpha parameters s (only in learn mode).",
    )
    p.add_argument(
        "--alpha_classwise_batch_norm",
        action="store_true",
        help="If set, normalize alpha within each batch per class to keep per-class average alpha=1.",
    )

    # Focus training (post warm-up): unfreeze last N backbone groups
    p.add_argument(
        "--train_last_n",
        type=int,
        default=0,
        help=(
            "After warm-up, freeze backbone and optionally unfreeze the last N discovered backbone groups "
            "(e.g., last blocks/layers/stages). 0 means head-only after warm-up."
        ),
    )

    # Attribution
    p.add_argument("--do_attribution", action="store_true")
    p.add_argument("--attribution_limit", type=int, default=50)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--similarity", type=str, default="cosine", choices=["cosine", "dot"])
    p.add_argument("--attribution_class", type=str, default="pred", choices=["pred", "true"])

    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("Args:")
    print(json.dumps(vars(args), indent=2, sort_keys=True))

    set_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    # Alpha reweighting requires sample indices.
    return_index = _alpha_enabled(args)

    train_loader, val_loader, num_classes = make_dataloaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        augment=args.augment,
        val_split=args.val_split,
        download=args.download,
        return_index=return_index,
        train_val_ratio=float(args.train_val_ratio),
        train_val_seed=(args.seed if args.train_val_seed is None else int(args.train_val_seed)),
    )

    backbone, head, feat_dim = create_backbone_and_head(
        model_name=args.model,
        pretrained=args.pretrained,
        num_classes=num_classes,
        device=device,
    )
    print(f"Model: {args.model} feat_dim={feat_dim} classes={num_classes}")

    metrics_csv = out_dir / "metrics.csv"

    ckpt_last_path = out_dir / "ckpt_last.pt"
    ckpt_best_path = out_dir / "ckpt_best.pt"
    ckpt_best_warmup_path = out_dir / "ckpt_best_warmup.pt"

    best_val_acc1 = float("-inf")
    best_warmup_val_acc1 = float("-inf")

    effective_backbone_lr = float(args.backbone_lr)
    if float(args.backbone_lr_mult) > 0:
        effective_backbone_lr = float(args.lr) * float(args.backbone_lr_mult)

    optimizer = None
    prev_trainable_key = None

    alpha_weights = None
    alpha_optimizer = None
    train_labels_full = None
    if _alpha_enabled(args):
        if str(args.alpha_mode).strip().lower() != "learn":
            raise ValueError("Only --alpha_mode=learn is supported in this implementation")

        # Build a full train-split dataset to size alpha and compute classwise normalization/stats.
        # NOTE: indices returned by loaders must correspond to this train split (works with Subset). 
        train_ds_full = UFGVCDataset(
            dataset_name=args.dataset,
            root=args.data_root,
            split="train",
            transform=None,
            download=args.download,
            return_index=False,
        )
        train_labels_full = train_ds_full.get_all_labels()
        alpha_cfg = AlphaConfig(
            mode="learn",
            entropy_reg=float(args.alpha_entropy_reg),
            s_l2_reg=float(args.alpha_s_l2_reg),
            classwise_batch_norm=bool(args.alpha_classwise_batch_norm),
        )
        alpha_weights = AlphaWeights(n_train=int(len(train_ds_full)), device=device, config=alpha_cfg)
        alpha_optimizer = torch.optim.AdamW(alpha_weights.parameters(), lr=float(args.alpha_lr))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # If we did warm-up, start focus stage from the best warm-up checkpoint.
        if args.warmup_epochs > 0 and epoch == args.warmup_epochs + 1 and ckpt_best_warmup_path.exists():
            ckpt = torch.load(ckpt_best_warmup_path, map_location=device)
            backbone.load_state_dict(ckpt["backbone"])
            head.load_state_dict(ckpt["head"])

        in_warmup = epoch <= args.warmup_epochs
        if in_warmup:
            # Stage 0: train full backbone + head with clean CE
            set_requires_grad(backbone, True)
            trainable_key = ("warmup", "all")
        else:
            # Stage 1: freeze most of backbone, optionally unfreeze last N groups
            trainable_groups = set_trainable_backbone_groups(backbone, train_last_n=int(args.train_last_n))
            trainable_key = ("focus", tuple(trainable_groups))

        # Rebuild optimizer only when trainable parameter set changes
        if optimizer is None or trainable_key != prev_trainable_key:
            optimizer = make_optimizer(
                optimizer_name=args.optimizer,
                head=head,
                backbone=backbone,
                lr=args.lr,
                backbone_lr=effective_backbone_lr,
                weight_decay=args.weight_decay,
            )
            prev_trainable_key = trainable_key

        train_loss, train_acc1 = train_one_epoch(
            backbone=backbone,
            head=head,
            loader=train_loader,
            val_loader=(val_loader if _alpha_enabled(args) else None),
            alpha_weights=alpha_weights,
            alpha_optimizer=alpha_optimizer,
            alpha_inner_lr=(args.lr if args.alpha_inner_lr is None else float(args.alpha_inner_lr)),
            num_classes=int(num_classes),
            optimizer=optimizer,
            device=device,
            lambda_l2=args.lambda_l2,
            warmup=in_warmup,
        )
        val_loss, val_acc1 = evaluate(backbone=backbone, head=head, loader=val_loader, device=device)

        seconds = time.time() - t0
        lr_now = float(optimizer.param_groups[0]["lr"])

        alpha_stats = {}
        if alpha_weights is not None and train_labels_full is not None:
            full_alpha = alpha_weights.full_alpha_detached(train_labels=train_labels_full, num_classes=int(num_classes))
            alpha_stats = compute_alpha_stats(full_alpha)

        m = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc1=train_acc1,
            val_loss=val_loss,
            val_acc1=val_acc1,
            lr=lr_now,
            seconds=seconds,
            **alpha_stats,
        )

        # Required: report per-epoch averaged metrics (NOT single-batch)
        line = asdict(m)
        print("EpochSummary:")
        print(json.dumps(line, indent=2, sort_keys=True))
        write_csv_row(metrics_csv, line)

        ckpt = {
            "epoch": epoch,
            "args": vars(args),
            "backbone": backbone.state_dict(),
            "head": head.state_dict(),
            "metrics": line,
        }

        if alpha_weights is not None:
            ckpt["alpha"] = alpha_weights.state_dict()
        if alpha_optimizer is not None:
            ckpt["alpha_optimizer"] = alpha_optimizer.state_dict()

        # Always keep last checkpoint.
        torch.save(ckpt, ckpt_last_path)

        # Keep best checkpoints by validation accuracy.
        if float(val_acc1) > best_val_acc1:
            best_val_acc1 = float(val_acc1)
            torch.save(ckpt, ckpt_best_path)

        if in_warmup and float(val_acc1) > best_warmup_val_acc1:
            best_warmup_val_acc1 = float(val_acc1)
            torch.save(ckpt, ckpt_best_warmup_path)

    if args.do_attribution:
        print("Running representer attribution...")
        train_feats, train_labels = extract_features(backbone, train_loader, device)
        val_feats, val_labels = extract_features(backbone, val_loader, device)

        representer_r = compute_representer_values(head, train_feats, train_labels, lambda_l2=args.lambda_l2)

        limit = min(int(args.attribution_limit), int(val_feats.size(0)))
        results = []
        head_device = next(head.parameters()).device
        for i in range(limit):
            q = val_feats[i : i + 1]
            y_true = int(val_labels[i].item())
            with torch.no_grad():
                logits = head(q.to(head_device))
                y_pred = int(logits.argmax(dim=1).item())
            y_used = y_pred if args.attribution_class == "pred" else y_true

            support, inhibit = representer_topk(
                train_feats=train_feats,
                representer_r=representer_r,
                query_feats=q,
                class_index=y_used,
                top_k=args.top_k,
                similarity=args.similarity,
            )
            results.append(
                {
                    "val_index": i,
                    "true": y_true,
                    "pred": y_pred,
                    "class_used": y_used,
                    "support_train_indices": support.tolist(),
                    "inhibit_train_indices": inhibit.tolist(),
                }
            )

        torch.save(results, out_dir / "representer_attribution.pt")
        (out_dir / "representer_attribution.json").write_text(
            json.dumps(results[: min(10, len(results))], indent=2), encoding="utf-8"
        )
        print(f"Saved attribution to: {out_dir}")


if __name__ == "__main__":
    main()
