"""Linear probing evaluation for a pretrained SimCLR backbone.

Protocol: freeze the backbone, discard the projection head, train a single
linear layer (512 → 10) on CIFAR-10 labels.  This measures how linearly
separable the learned representations are — the standard SSL evaluation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import SimCLRModel
from src.utils import AverageMeter, accuracy, load_checkpoint


class LinearClassifier(nn.Module):
    """Single linear layer on top of frozen features."""

    def __init__(self, in_dim: int = 512, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@torch.no_grad()
def _extract_features(
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen backbone over the full dataset once and return
    (features, labels) tensors on CPU.

    Extracting features up-front avoids redundant forward passes through the
    backbone every epoch and keeps the probe training loop fast.
    """
    backbone.eval()
    all_feats: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for images, labels in tqdm(loader, desc="Extracting features", leave=False):
        images = images.to(device, non_blocking=True)
        feats, _ = backbone(images)
        all_feats.append(feats.cpu())
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


def linear_probe(
    cfg: dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """Train and evaluate a linear probe on pretrained SimCLR features."""
    from src.data import get_probe_loaders

    pcfg = cfg["probe"]
    dlcfg = cfg["dataloader"]

    # ── Load pretrained backbone ──
    model = SimCLRModel(
        projection_dim=cfg["pretrain"]["projection_dim"],
        hidden_dim=cfg["pretrain"]["hidden_dim"],
    ).to(device)

    ckpt = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")

    # Freeze backbone — only the linear head will be trained
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # ── Data ──
    train_loader, test_loader = get_probe_loaders(
        batch_size=pcfg["batch_size"],
        num_workers=dlcfg["num_workers"],
        pin_memory=dlcfg["pin_memory"],
        persistent_workers=dlcfg["persistent_workers"],
    )

    # ── Extract features once (avoids repeated backbone forward passes) ──
    print("Extracting train features…")
    train_feats, train_labels = _extract_features(model, train_loader, device)
    print("Extracting test features…")
    test_feats, test_labels = _extract_features(model, test_loader, device)

    train_feat_ds = torch.utils.data.TensorDataset(train_feats, train_labels)
    test_feat_ds = torch.utils.data.TensorDataset(test_feats, test_labels)
    train_feat_loader = DataLoader(train_feat_ds, batch_size=pcfg["batch_size"], shuffle=True)
    test_feat_loader = DataLoader(test_feat_ds, batch_size=pcfg["batch_size"], shuffle=False)

    # ── Linear classifier ──
    classifier = LinearClassifier(in_dim=512, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=pcfg["lr"],
        momentum=pcfg["momentum"],
        weight_decay=pcfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pcfg["epochs"])

    best_acc = 0.0

    for epoch in range(1, pcfg["epochs"] + 1):
        # ── Train ──
        classifier.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for feats, labels in train_feat_loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = classifier(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            acc1 = accuracy(logits, labels, topk=(1,))[0]
            loss_meter.update(loss.item(), feats.size(0))
            acc_meter.update(acc1, feats.size(0))

        scheduler.step()

        # ── Evaluate ──
        classifier.eval()
        test_acc_meter = AverageMeter()

        with torch.no_grad():
            for feats, labels in test_feat_loader:
                feats = feats.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = classifier(feats)
                acc1 = accuracy(logits, labels, topk=(1,))[0]
                test_acc_meter.update(acc1, feats.size(0))

        test_acc = test_acc_meter.avg
        if test_acc > best_acc:
            best_acc = test_acc

        print(
            f"[Epoch {epoch:>3d}/{pcfg['epochs']}]  "
            f"train_loss={loss_meter.avg:.4f}  train_acc={acc_meter.avg:.2f}%  "
            f"test_acc={test_acc:.2f}%  best={best_acc:.2f}%"
        )

    print(f"\nLinear probe best test accuracy: {best_acc:.2f}%")
