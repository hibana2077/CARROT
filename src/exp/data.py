from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms

from src.dataset.ufgvc import UFGVCDataset


def build_transforms(img_size: int, augment: bool) -> Tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if augment:
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

    val_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, val_tf


def make_dataloaders(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    img_size: int,
    augment: bool,
    val_split: str,
    download: bool,
    return_index: bool = False,
    train_val_ratio: float = 0.1,
    train_val_seed: int | None = None,
):
    train_tf, val_tf = build_transforms(img_size=img_size, augment=augment)

    val_split_norm = str(val_split).strip().lower()
    if val_split_norm in {"train_split", "train"}:
        if not (0.0 < float(train_val_ratio) < 1.0):
            raise ValueError(f"train_val_ratio must be in (0, 1), got {train_val_ratio}")

        full_train_aug = UFGVCDataset(
            dataset_name=dataset_name,
            root=data_root,
            split="train",
            transform=train_tf,
            download=download,
            return_index=return_index,
        )
        full_train_val = UFGVCDataset(
            dataset_name=dataset_name,
            root=data_root,
            split="train",
            transform=val_tf,
            download=download,
            return_index=return_index,
        )

        n_total = len(full_train_aug)
        n_val = int(n_total * float(train_val_ratio))
        n_val = max(1, min(n_total - 1, n_val))

        gen = torch.Generator()
        gen.manual_seed(int(0 if train_val_seed is None else train_val_seed))
        perm = torch.randperm(n_total, generator=gen).tolist()
        val_indices = perm[:n_val]
        train_indices = perm[n_val:]

        train_ds = Subset(full_train_aug, train_indices)
        val_ds = Subset(full_train_val, val_indices)
        val_split = f"train_split({float(train_val_ratio):.3f})"
        num_classes = len(full_train_aug.classes)
    else:
        train_ds = UFGVCDataset(
            dataset_name=dataset_name,
            root=data_root,
            split="train",
            transform=train_tf,
            download=download,
            return_index=return_index,
        )
        try:
            val_ds = UFGVCDataset(
                dataset_name=dataset_name,
                root=data_root,
                split=val_split,
                transform=val_tf,
                download=download,
                return_index=return_index,
            )
        except Exception:
            fallback = "val" if val_split != "val" else "test"
            val_ds = UFGVCDataset(
                dataset_name=dataset_name,
                root=data_root,
                split=fallback,
                transform=val_tf,
                download=download,
                return_index=return_index,
            )
            val_split = fallback

        num_classes = len(train_ds.classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Dataset={dataset_name} train={len(train_ds)} {val_split}={len(val_ds)} classes={num_classes}")
    return train_loader, val_loader, num_classes
