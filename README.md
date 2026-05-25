# SimCLR on CIFAR-10 (PyTorch)

A from-scratch implementation of **SimCLR** (Chen et al., 2020 — *A Simple Framework for Contrastive Learning of Visual Representations*) trained on CIFAR-10, with linear-probe evaluation.

No pre-built SSL libraries — pure PyTorch + torchvision for the dataset and base ResNet building blocks.

## Quick start

```bash
# NixOS
nix-shell
pip install -r requirements.txt

# Other distros
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Pretrain

```bash
python main.py pretrain --config config.yaml
```

Checkpoints are saved to `./checkpoints/` every 50 epochs and at the end.

### Linear probe

```bash
python main.py probe --config config.yaml --checkpoint ./checkpoints/simclr_final.pt
```

Trains a single linear layer on frozen backbone features and reports train/test accuracy each epoch.

## Project structure

```
simclr-cifar10/
├── main.py                Entry point (pretrain / probe subcommands)
├── config.yaml            All hyperparameters in one place
├── requirements.txt
├── shell.nix              Nix dev shell (Python 3.12 + venv)
└── src/
    ├── data.py            CIFAR-10 loader + SimCLR dual-view augmentations
    ├── model.py           CIFAR-adapted ResNet-18 backbone + MLP projection head
    ├── loss.py            NT-Xent loss (fully vectorised, no Python loops)
    ├── pretrain.py        Self-supervised pretraining loop (AMP, cosine schedule)
    ├── linear_probe.py    Linear probing evaluation
    └── utils.py           Seeding, checkpointing, top-k accuracy
```

## Technical choices

### CIFAR-adapted ResNet-18
Standard ImageNet ResNet-18 uses a 7×7 stride-2 conv + max-pool as the stem, which aggressively down-samples the input. On 32×32 CIFAR images this destroys spatial information before the residual blocks even begin. The stem is replaced with a 3×3 stride-1 conv and the max-pool is removed, following the convention from prior CIFAR SSL works.

### No Gaussian blur
The SimCLR paper reports that Gaussian blur improves ImageNet performance, but notes diminishing returns on smaller images. At 32×32, blur is more likely to destroy useful texture than to teach blur-invariance, so it is omitted.

### Temperature τ = 0.5
The original paper uses τ = 0.5 for CIFAR-scale experiments. Lower temperatures sharpen the softmax distribution, making the loss more sensitive to hard negatives — but too low and training becomes unstable. 0.5 is the sweet spot for batch size 256–512 on CIFAR-10.

### Projection head: 512 → 512 → 128
The paper shows that downstream performance improves when the contrastive loss is applied in a lower-dimensional projected space rather than directly on backbone features. A 2-layer MLP with batch-norm bridges the 512-d backbone output to 128-d projections. Representations *before* the projection head are used for downstream tasks.

### LR scaling: lr = 0.5 × batch_size / 256
Linear LR scaling keeps the effective update magnitude consistent across batch sizes. Combined with 10-epoch linear warmup and cosine annealing, this schedule is robust without needing LARS.

### SGD over LARS
LARS is designed for very large batch sizes (≥ 1024). At batch size 256–512, plain SGD with momentum matches LARS and is simpler. Weight decay 5e-4 provides sufficient regularisation.

## References

* Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations*. ICML 2020. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
