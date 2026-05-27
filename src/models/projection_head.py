"""
projection_head.py — SimCLR projection head for Phase A self-supervised pre-training.

Maps z_cls → low-dimensional space for NT-Xent contrastive learning.
This module is DISCARDED after Phase A; only encoder weights (encoder.pt) are kept.

IMPORTANT — CLAUDE.md constraint #3:
    L2 normalisation belongs INSIDE the NT-Xent loss function, NOT here.
    ProjectionHead outputs raw (unnormalised) projection vectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head for NT-Xent contrastive loss (SimCLR).

    Architecture (Chen et al. 2020):
        z_cls (B, d_model) → Linear(d_model → hidden) → GELU → Linear(hidden → proj_dim)
                           → (B, proj_dim)   ← raw, unnormalised

    L2 normalisation is intentionally absent — it is applied inside the NT-Xent
    loss (CLAUDE.md constraint #3), keeping the model layer itself generic.

    Args:
        cfg: Full OmegaConf config (uses cfg.model.d_model, cfg.phase_a.simclr).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        d_model = cfg.model.d_model
        simclr  = cfg.phase_a.simclr

        self.mlp = nn.Sequential(
            nn.Linear(d_model, simclr.projection_hidden_dim),
            nn.GELU(),
            nn.Linear(simclr.projection_hidden_dim, simclr.projection_output_dim),
        )

    def forward(self, z_cls: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_cls: (B, d_model) — CLS token from TransformerEncoder output.

        Returns:
            projection: (B, proj_dim) — raw projection vector (no L2 norm applied).
        """
        return self.mlp(z_cls)
