"""
film.py — Feature-wise Linear Modulation (FiLM) conditioning module.

FiLM injects the global physiological representation z_cls into every
decoder stage, allowing patient-level context (pathology class, morphology,
age/sex) to modulate local reconstruction at each spatial resolution.

Spec (Eq. 43-44):
    γ, β = split(Linear(z_cls)) ∈ R^C × R^C
    u_FiLM = γ ⊙ LN(u) + β

where LN normalises over the channel dimension and γ, β are per-channel scalars
broadcast over the temporal dimension T.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class FiLM(nn.Module):
    """Feature-wise Linear Modulation.

    A single linear layer maps z_cls to twice the number of channels, then
    splits into gamma (scale) and beta (shift) applied after LayerNorm.

    Reusable across all decoder stages — instantiate once per stage with the
    correct ``channels`` matching the stage's feature map width.

    Args:
        d_model:  Dimension of the conditioning vector z_cls.
        channels: Number of channels C in the feature map to modulate.
    """

    def __init__(self, d_model: int, channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 2 * channels)
        # LayerNorm over channel dim; operates after transpose → back
        self.norm = nn.LayerNorm(channels)

    def forward(self, u: Tensor, z_cls: Tensor) -> Tensor:
        """
        Args:
            u:     (B, C, T) — feature map to modulate
            z_cls: (B, d_model) — global conditioning vector from Transformer CLS output

        Returns:
            (B, C, T) — γ ⊙ LN(u) + β
        """
        params = self.linear(z_cls)                              # (B, 2*C)
        gamma, beta = params.chunk(2, dim=-1)                    # each (B, C)

        # LN along channel dim: (B, C, T) → transpose → (B, T, C) → norm → transpose back
        u_norm = self.norm(u.transpose(1, 2)).transpose(1, 2)    # (B, C, T)

        # Broadcast gamma/beta over temporal dim
        return gamma.unsqueeze(-1) * u_norm + beta.unsqueeze(-1)  # (B, C, T)
