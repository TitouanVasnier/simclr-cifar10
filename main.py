"""Entry point for SimCLR pretraining and linear-probe evaluation on CIFAR-10.

Usage
-----
    python main.py pretrain --config config.yaml
    python main.py probe    --config config.yaml --checkpoint ./checkpoints/simclr_final.pt
"""

from __future__ import annotations

import argparse

import torch
import yaml

from src.utils import set_seed


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="SimCLR on CIFAR-10")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── pretrain ──
    p_pretrain = sub.add_parser("pretrain", help="Self-supervised pretraining")
    p_pretrain.add_argument("--config", type=str, default="config.yaml")

    # ── probe ──
    p_probe = sub.add_parser("probe", help="Linear-probe evaluation")
    p_probe.add_argument("--config", type=str, default="config.yaml")
    p_probe.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a pretrained checkpoint (.pt)",
    )

    args = parser.parse_args()
    cfg = _load_config(args.config)

    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.command == "pretrain":
        from src.pretrain import pretrain
        pretrain(cfg, device)
    elif args.command == "probe":
        from src.linear_probe import linear_probe
        linear_probe(cfg, args.checkpoint, device)


if __name__ == "__main__":
    main()
