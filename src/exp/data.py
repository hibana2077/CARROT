from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader
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
):
    train_tf, val_tf = build_transforms(img_size=img_size, augment=augment)

    train_ds = UFGVCDataset(
        dataset_name=dataset_name,
        root=data_root,
        split="train",
        transform=train_tf,
        download=download,
    )
    try:
        val_ds = UFGVCDataset(
            dataset_name=dataset_name,
            root=data_root,
            split=val_split,
            transform=val_tf,
            download=download,
        )
    except Exception:
        fallback = "val" if val_split != "val" else "test"
        val_ds = UFGVCDataset(
            dataset_name=dataset_name,
            root=data_root,
            split=fallback,
            transform=val_tf,
            download=download,
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
