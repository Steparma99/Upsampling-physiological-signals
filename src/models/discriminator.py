"""
discriminator.py — PatchGAN discriminator for ECG super-resolution (Phase B).

Architecture (Eq. 51-54):
  D1:   Conv1d(1→64,   k=15, stride=4, pad=6)            + LeakyReLU(0.2)
  D2:   Conv1d(64→128, k=15, stride=4, pad=6) + IN       + LeakyReLU(0.2)
  D3:   Conv1d(128→256,k=15, stride=4, pad=6) + IN       + LeakyReLU(0.2)
  Dout: Conv1d(256→1,  k=7,            pad=3)

Spatial progression (N_HR=5000):
  5000 → 1250 → 312 → 78  (per-patch output)

No sigmoid — discriminator is trained with hinge loss.
Intermediate features D1, D2, D3 are returned for feature matching loss.

Activation order per layer: Conv → InstanceNorm (layers 2-3) → LeakyReLU.
No normalization on the first layer (standard PatchGAN convention).
"""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator for ECG super-resolution.

    Classifies every local patch of the input HR signal as real or generated.
    Designed for hinge loss — no sigmoid at the output.

    Exposes intermediate features [D1, D2, D3] for the feature matching
    loss term in the Phase B generator objective.

    Args:
        cfg: Full OmegaConf config (top-level).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        dcfg = cfg.model.discriminator
        in_ch = [1] + list(dcfg.channels)          # [1, 64, 128, 256]
        out_ch = list(dcfg.channels)                # [64, 128, 256]
        k: int = dcfg.kernel_size                   # 15
        s: int = dcfg.stride                        # 4
        p: int = dcfg.padding                       # 6
        slope: float = dcfg.leaky_slope             # 0.2

        # Three strided layers — InstanceNorm on layers 2-3, none on layer 1
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch[i], out_ch[i], kernel_size=k, stride=s, padding=p)
            for i in range(len(out_ch))
        ])
        self.norms = nn.ModuleList([
            nn.Identity() if i == 0 else nn.InstanceNorm1d(out_ch[i])
            for i in range(len(out_ch))
        ])
        self.act = nn.LeakyReLU(slope, inplace=True)

        # Output head: per-patch logit map, no sigmoid
        hk: int = dcfg.head_kernel_size             # 7
        self.head = nn.Conv1d(out_ch[-1], 1, kernel_size=hk, padding=hk // 2)

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """
        Args:
            x: (B, N_HR) or (B, 1, N_HR) — full HR ECG signal (real or generated).
               The generator output should be detached when training the discriminator.

        Returns:
            logits:   (B, 1, 78) — per-patch real/fake scores, no sigmoid applied.
            features: [D1, D2, D3] — intermediate activations for feature matching.
                      D1: (B, 64,  1250)
                      D2: (B, 128,  312)
                      D3: (B, 256,   78)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (B, 1, N_HR)

        features: list[Tensor] = []
        for conv, norm in zip(self.convs, self.norms):
            x = self.act(norm(conv(x)))
            features.append(x)

        logits = self.head(x)           # (B, 1, 78)
        return logits, features
