"""Self-supervised SimCLR pretraining loop.

Optimizer : SGD + momentum (LARS is overkill at batch size ≤ 512).
LR schedule: linear warmup → cosine annealing to 0.
Mixed precision via ``torch.cuda.amp`` for faster training on Ampere+ GPUs.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.loss import NTXentLoss
from src.model import SimCLRModel
from src.utils import AverageMeter, save_checkpoint


def _lr_for_step(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
) -> float:
    """Compute the learning rate at a given *step* (not epoch).

    * Linear warmup from 0 → base_lr over the first ``warmup_steps``.
    * Cosine decay from base_lr → 0 over the remaining steps.
    """
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def pretrain(
    cfg: dict[str, Any],
    device: torch.device,
) -> None:
    """Run SimCLR pretraining end-to-end and save checkpoints."""
    from src.data import get_pretrain_loader

    pcfg = cfg["pretrain"]
    dlcfg = cfg["dataloader"]

    # ── Data ──
    loader: DataLoader = get_pretrain_loader(
        batch_size=pcfg["batch_size"],
        num_workers=dlcfg["num_workers"],
        pin_memory=dlcfg["pin_memory"],
        persistent_workers=dlcfg["persistent_workers"],
    )

    # ── Model ──
    model = SimCLRModel(
        projection_dim=pcfg["projection_dim"],
        hidden_dim=pcfg["hidden_dim"],
    ).to(device)

    # ── Loss ──
    criterion = NTXentLoss(temperature=pcfg["temperature"])

    # ── Optimizer ──
    # LR scaling rule from the paper: lr = base_lr * batch_size / 256
    base_lr: float = pcfg["lr_base"] * pcfg["batch_size"] / 256
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=pcfg["momentum"],
        weight_decay=pcfg["weight_decay"],
    )

    # ── Schedule bookkeeping ──
    steps_per_epoch = len(loader)
    total_steps = pcfg["epochs"] * steps_per_epoch
    warmup_steps = pcfg["warmup_epochs"] * steps_per_epoch
    global_step = 0

    # ── Mixed precision ──
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Optional TensorBoard ──
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()
    except ImportError:
        pass

    # ── Training loop ──
    for epoch in range(1, pcfg["epochs"] + 1):
        model.train()
        loss_meter = AverageMeter()
        t0 = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{pcfg['epochs']}", leave=False)
        for x_i, x_j, _ in pbar:
            x_i = x_i.to(device, non_blocking=True)
            x_j = x_j.to(device, non_blocking=True)

            lr = _lr_for_step(global_step, total_steps, warmup_steps, base_lr)
            _set_lr(optimizer, lr)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                _, z_i = model(x_i)
                _, z_j = model(x_j)
            loss = criterion(z_i.float(), z_j.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), x_i.size(0))
            global_step += 1
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", lr=f"{lr:.5f}")

        elapsed = time.time() - t0
        print(
            f"[Epoch {epoch:>3d}/{pcfg['epochs']}]  "
            f"loss={loss_meter.avg:.4f}  lr={lr:.5f}  "
            f"time={elapsed:.1f}s"
        )

        if writer is not None:
            writer.add_scalar("pretrain/loss", loss_meter.avg, epoch)
            writer.add_scalar("pretrain/lr", lr, epoch)

        # ── Checkpointing ──
        if epoch % pcfg["checkpoint_every"] == 0 or epoch == pcfg["epochs"]:
            tag = "final" if epoch == pcfg["epochs"] else f"ep{epoch}"
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_meter.avg,
                },
                path=f"./checkpoints/simclr_{tag}.pt",
            )
            print(f"  → checkpoint saved: checkpoints/simclr_{tag}.pt")

    if writer is not None:
        writer.close()
