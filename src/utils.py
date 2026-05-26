"""Shared utilities: reproducibility, metrics tracking, checkpointing."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Pin every source of randomness so runs are reproducible.

    cuDNN deterministic mode trades a small amount of speed for bitwise
    reproducibility across runs on the same hardware.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class AverageMeter:
    """Running mean tracker for a single scalar (loss, accuracy, …)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def save_checkpoint(
    state: dict[str, Any],
    path: str | Path,
) -> None:
    """Persist training state to disk, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Reload a checkpoint saved with :func:`save_checkpoint`."""
    return torch.load(path, map_location=device, weights_only=False)


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Sequence[int] = (1,),
) -> list[float]:
    """Compute top-k accuracy for the given logits and ground-truth labels.

    Returns a list of accuracies (as percentages) in the same order as *topk*.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results: list[float] = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0).item()
            results.append(correct_k * 100.0 / batch_size)
        return results
