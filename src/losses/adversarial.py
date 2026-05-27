"""
adversarial.py — Adversarial losses for Phase B PatchGAN training.

Three components:

  HingeLoss            — discriminator update (Eq. 59) + generator update (Eq. 60)
  FeatureMatchingLoss  — L1 on intermediate discriminator features (Eq. 61)

No sigmoid is applied to discriminator logits — these losses are designed for
the raw per-patch scores output by PatchGANDiscriminator.

Discriminator warmup: the training loop in phase_b.py should call
HingeLoss.generator_loss() and FeatureMatchingLoss only after
cfg.phase_b.discriminator_warmup_epochs (CLAUDE.md constraint #6).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HingeLoss(nn.Module):
    """Hinge loss for PatchGAN discriminator and generator.

    Discriminator objective (Eq. 59):
        L_D = E[max(0, 1 − D(x_real))] + E[max(0, 1 + D(x_fake))]

    Generator adversarial objective (Eq. 60):
        L_adv_G = −E[D(x̂_HR)]

    The two objectives are separated because they are optimised in different
    gradient-tape scopes: discriminator_loss() during the D update step,
    generator_loss() during the G update step.
    """

    def discriminator_loss(
        self,
        logits_real: Tensor,
        logits_fake: Tensor,
    ) -> Tensor:
        """Hinge loss for the discriminator update step.

        Args:
            logits_real: (B, 1, P) — D(x_real), raw logits, no sigmoid
            logits_fake: (B, 1, P) — D(x̂_HR).detach(), raw logits
                         Must be detached from the generator graph.

        Returns:
            Scalar discriminator loss.
        """
        loss_real = F.relu(1.0 - logits_real).mean()
        loss_fake = F.relu(1.0 + logits_fake).mean()
        return loss_real + loss_fake

    def generator_loss(self, logits_fake: Tensor) -> Tensor:
        """Adversarial term for the generator update step.

        Args:
            logits_fake: (B, 1, P) — D(x̂_HR), raw logits.
                         Must NOT be detached — gradients flow to the generator.

        Returns:
            Scalar adversarial generator loss (negative mean of fake logits).
        """
        return -logits_fake.mean()


class FeatureMatchingLoss(nn.Module):
    """L1 distance between intermediate discriminator features (Eq. 61).

        L_FM = Σ_{l=1}^{3} (1/N_l) × ‖D_l(x̂_HR) − D_l(x_real)‖₁

    where D_l are the intermediate activations at layer l and N_l is the
    number of elements (normalises across layers with different resolutions).

    Stabilises GAN training by encouraging the generator to produce features
    that match real signal statistics at multiple discriminator scales —
    less prone to mode collapse than the adversarial loss alone.

    Args:
        n_layers: Number of intermediate feature layers to match (default 3,
                  matching PatchGANDiscriminator's three strided layers).
    """

    def __init__(self, n_layers: int = 3) -> None:
        super().__init__()
        self.n_layers = n_layers

    def forward(
        self,
        features_fake: list[Tensor],
        features_real: list[Tensor],
    ) -> Tensor:
        """
        Args:
            features_fake: [D1, D2, D3] from discriminator on generated signal.
                           Gradients must flow to the generator — do NOT detach.
            features_real: [D1, D2, D3] from discriminator on real signal.
                           Should be detached — real signal is not being optimised.

        Returns:
            Scalar feature matching loss (sum of per-layer normalised L1).
        """
        assert len(features_fake) == len(features_real), (
            f"Feature list length mismatch: {len(features_fake)} vs {len(features_real)}"
        )
        n = min(self.n_layers, len(features_fake))
        loss = torch.zeros(1, device=features_fake[0].device, dtype=features_fake[0].dtype)
        for f_fake, f_real in zip(features_fake[:n], features_real[:n]):
            # F.l1_loss(reduction='mean') normalises by N_l automatically
            loss = loss + F.l1_loss(f_fake, f_real.detach())
        return loss.squeeze(0)
