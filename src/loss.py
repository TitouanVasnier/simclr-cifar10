"""NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss for SimCLR.

Given a mini-batch of N images, SimCLR produces 2N augmented views.
For each anchor view *i*, there is exactly one positive (its paired view) and
2N-2 negatives (every other view in the batch).  The loss pushes the anchor
closer to its positive while repelling it from all negatives, scaled by a
temperature parameter τ.

Reference: Chen et al., "A Simple Framework for Contrastive Learning of Visual
Representations" (ICML 2020), Section 3 / Eq. 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NTXentLoss(nn.Module):
    """Vectorised NT-Xent loss — no Python loop over the batch."""

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute the NT-Xent loss for a batch of paired projections.

        Parameters
        ----------
        z_i : (N, D) — L2-normalised projections of the first views.
        z_j : (N, D) — L2-normalised projections of the second views.

        Returns
        -------
        Scalar loss averaged over all 2N anchors.
        """
        N = z_i.size(0)

        # ── 1. Stack both views into a single (2N, D) tensor ──
        # Layout: [z_i_0, z_i_1, …, z_i_{N-1}, z_j_0, z_j_1, …, z_j_{N-1}]
        z = torch.cat([z_i, z_j], dim=0)  # (2N, D)

        # ── 2. Pairwise cosine similarity matrix ──
        # Because z is already L2-normalised, cosine sim = dot product.
        # sim[a, b] = z[a] · z[b] / τ
        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)

        # ── 3. Mask out self-similarity on the diagonal ──
        # An anchor should never be compared to itself (would dominate the
        # softmax with sim = 1/τ).  Setting diagonal to -inf ensures exp → 0.
        mask_self = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask_self, float("-inf"))

        # ── 4. Build the positive-pair target indices ──
        # For anchor i   (first view of image i, index i     in [0..N-1]):
        #   its positive is j (second view of same image, index i + N).
        # For anchor i+N (second view of image i, index i + N in [N..2N-1]):
        #   its positive is i (first view, index i).
        #
        # So: targets = [N, N+1, …, 2N-1,  0, 1, …, N-1]
        targets = torch.cat([
            torch.arange(N, 2 * N, device=z.device),  # positives for first views
            torch.arange(0, N, device=z.device),       # positives for second views
        ])  # (2N,)

        # ── 5. Cross-entropy loss ──
        # Each row of `sim` is a logit vector over 2N candidates (self excluded
        # via -inf).  The correct class for row k is targets[k].
        # cross_entropy applies softmax internally, so this is equivalent to:
        #   -log( exp(sim[k, targets[k]]) / Σ_{m≠k} exp(sim[k, m]) )
        loss = nn.functional.cross_entropy(sim, targets)

        return loss
