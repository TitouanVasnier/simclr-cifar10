"""SimCLR model: CIFAR-adapted ResNet-18 backbone + MLP projection head.

The two CIFAR-specific modifications to a standard ImageNet ResNet-18 are:
1. Replace the 7x7 stride-2 conv1 with a 3x3 stride-1 conv (32x32 images
   are already small - a 7x7 kernel would collapse spatial information).
2. Remove the initial max-pool for the same reason.

The projection head maps the 512-d backbone features into a 128-d space where
the contrastive loss is computed.  Following the paper, representations *before*
the projection head (i.e. the 512-d features) are the ones used downstream.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, ResNet


class ResNet18CIFAR(ResNet):
    """ResNet-18 re-wired for 32x32 inputs (CIFAR-10/100).

    Inherits the full ResNet block machinery from torchvision but overrides
    the stem (conv1 + pool) so spatial resolution is preserved through the
    early layers.
    """

    def __init__(self) -> None:
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=1)
        # 3x3 stride-1 instead of 7x7 stride-2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove the max-pool entirely (identity keeps the forward path valid)
        self.maxpool = nn.Identity()
        # We don't need the classifier - features come from avgpool
        del self.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return 512-d features after global average pooling."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # identity

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)  # (B, 512)


class ProjectionHead(nn.Module):
    """2-layer MLP that maps backbone features into the contrastive space.

    Architecture: Linear(in, hidden) → BN → ReLU → Linear(hidden, out).
    Output is L2-normalised so cosine similarity reduces to a dot product.
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 512,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=1)


class SimCLRModel(nn.Module):
    """Full SimCLR model: backbone encoder + projection head.

    ``forward`` returns both representations so the caller can choose:
    * *features* (512-d) - for downstream tasks (linear probe).
    * *projections* (128-d, L2-normed) - for the contrastive loss.
    """

    def __init__(self, projection_dim: int = 128, hidden_dim: int = 512) -> None:
        super().__init__()
        self.backbone = ResNet18CIFAR()
        self.projection_head = ProjectionHead(
            in_dim=512,
            hidden_dim=hidden_dim,
            out_dim=projection_dim,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        projections = self.projection_head(features)
        return features, projections
