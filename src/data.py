"""CIFAR-10 dataset wrappers and SimCLR augmentation pipeline.

The key idea: each image is transformed *twice* independently so the model
learns to pull together two views of the same image while pushing apart views
of different images.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# CIFAR-10 channel-wise statistics (precomputed on the training split).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def simclr_augmentation() -> transforms.Compose:
    """Build the SimCLR stochastic augmentation pipeline for 32x32 images.

    Following the original paper (Chen et al., 2020) — adapted for CIFAR-10:
    * RandomResizedCrop to 32x32 with aggressive scale range.
    * Horizontal flip.
    * Color jitter (applied with p=0.8).
    * Random grayscale (p=0.2).
    * No Gaussian blur: the paper notes it helps on ImageNet (224x224) but
      32x32 images are already low-resolution enough that blur adds noise
      rather than useful invariance.
    * Standard CIFAR-10 normalization at the end.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


class SimCLRDataset(Dataset):
    """Wraps a torchvision dataset so each ``__getitem__`` returns two
    independent augmented views of the same image.

    The label is returned as well (unused during pretraining, but handy for
    debugging / visualization).
    """

    def __init__(self, base_dataset: datasets.CIFAR10) -> None:
        self.base_dataset = base_dataset
        self.transform = simclr_augmentation()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img, label = self.base_dataset[idx]  # PIL Image, int
        x_i = self.transform(img)
        x_j = self.transform(img)
        return x_i, x_j, label


def train_eval_transform() -> transforms.Compose:
    """Standard augmentation for supervised linear-probe training."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def test_transform() -> transforms.Compose:
    """Deterministic transform for evaluation (no augmentation)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_pretrain_loader(
    batch_size: int,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    """DataLoader that yields pairs of augmented views for pretraining."""
    base = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=None,
    )
    dataset = SimCLRDataset(base)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=True,
    )


def get_probe_loaders(
    batch_size: int,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Train and test DataLoaders for linear-probe evaluation."""
    train_ds = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_eval_transform(),
    )
    test_ds = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform(),
    )
    common: dict[str, Any] = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, test_loader
