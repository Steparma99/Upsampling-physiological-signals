"""
decoder.py — U-Net decoder for ECG super-resolution (Phase B).

Architecture (Table 4 of the design spec):

  Input:  z_patches (B, P=20, d_model=256)  — Transformer patch outputs
          z_cls     (B, d_model=256)         — global CLS for FiLM
          skips     [L × (B, d_model, N_LR)] — CNN encoder intermediate outputs

  Stage 1:  (B, 256,  20) ×5 SubPixel → (B, 256, 100)  + FiLM
  Stage 2:  (B, 256, 100) ×5 SubPixel → (B, 128, 500)  + FiLM
  Stage 3:  (B, 128, 500) ×2 SubPixel → (B, 128,1000) + skip e[3] + FiLM
  Stage 4:  (B, 128,1000) ×5 SubPixel → (B,  64,5000)  (no FiLM)
  Head:     Conv1D(64→1, k=7)          → (B,   1,5000) → squeeze → (B, 5000)

SubPixel reshape: (C×R, T) → (C, T×R) via pixel shuffle — CLAUDE.md constraint #7.
Never use transposed convolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from .film import FiLM


# =============================================================================
# SubPixel Conv 1D — pixel shuffle upsampling
# =============================================================================

class SubPixelConv1D(nn.Module):
    """1D pixel shuffle upsampling (Shi et al. 2016, adapted to 1D).

    A Conv1d produces ``out_channels * upscale_factor`` channels, then the
    output is rearranged from (B, C*R, T) to (B, C, T*R) by interleaving
    channels along the temporal axis:

        output[b, c, t*R + r] = conv_out[b, c*R + r, t]

    This avoids checkerboard artifacts that transposed convolution produces.

    Args:
        in_channels:    Input channel count C_in.
        out_channels:   Output channel count C after upsampling.
        upscale_factor: Temporal upsampling factor R.
        kernel_size:    Conv kernel size (default 3, keeps length-preserving).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_factor: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        # Conv produces C_out * R channels; pixel shuffle distributes them in time
        self.conv = nn.Conv1d(
            in_channels,
            out_channels * upscale_factor,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C_in, T)

        Returns:
            (B, C_out, T*R)
        """
        x = self.conv(x)                       # (B, C_out*R, T)
        B, CR, T = x.shape
        C = CR // self.upscale_factor
        R = self.upscale_factor
        # Pixel shuffle: (C*R, T) → (C, T*R) — CLAUDE.md constraint #7
        x = x.view(B, C, R, T)                 # (B, C, R, T)
        x = x.permute(0, 1, 3, 2)             # (B, C, T, R)
        return x.reshape(B, C, T * R)          # (B, C, T*R)


# =============================================================================
# Decoder Block — one U-Net stage
# =============================================================================

class DecoderBlock(nn.Module):
    """One U-Net decoder stage.

    Operations in fixed order:
      1. SubPixelConv1D — temporal upsampling
      2. Skip connection  — concat encoder feature map + project back to out_channels
      3. FiLM conditioning — modulate with z_cls (optional, disabled at stage 4)

    The skip projection conv is only instantiated when ``skip_channels`` is provided.
    FiLM is only instantiated when ``use_film=True``.

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count after upsampling.
        upscale_factor: Temporal upsampling factor R.
        d_model:      Dimension of z_cls for FiLM.
        skip_channels: Channel count of the encoder skip tensor (None = no skip).
        use_film:     Whether to apply FiLM at the end of this stage.
        kernel_size:  Conv kernel size (default 3).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_factor: int,
        d_model: int,
        skip_channels: int | None = None,
        use_film: bool = True,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.subpixel = SubPixelConv1D(in_channels, out_channels, upscale_factor, kernel_size)

        # Projection after skip concat: (out_channels + skip_channels) → out_channels
        self.skip_proj: nn.Conv1d | None = None
        if skip_channels is not None:
            self.skip_proj = nn.Conv1d(
                out_channels + skip_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )

        self.film: FiLM | None = FiLM(d_model, out_channels) if use_film else None

    def forward(
        self,
        x: Tensor,
        z_cls: Tensor,
        skip: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x:     (B, C_in, T_in)
            z_cls: (B, d_model)
            skip:  (B, skip_channels, T_out) — encoder feature map at target resolution

        Returns:
            (B, C_out, T_out)
        """
        x = self.subpixel(x)                           # (B, C_out, T_out)

        if skip is not None and self.skip_proj is not None:
            x = torch.cat([x, skip], dim=1)            # (B, C_out + skip_channels, T_out)
            x = self.skip_proj(x)                      # (B, C_out, T_out)

        if self.film is not None:
            x = self.film(x, z_cls)                    # (B, C_out, T_out)

        return x


# =============================================================================
# U-Net Decoder
# =============================================================================

class UNetDecoder(nn.Module):
    """Four-stage U-Net decoder for ECG super-resolution.

    Reads all architecture parameters from config (no hardcoded values).
    Channel widths are derived from d_model and base_channels:

        stage_channels = [d_model, d_model, d_model//2, d_model//2, base_channels]
        # stage i:  in = stage_channels[i],  out = stage_channels[i+1]

    Skip connection: stage 3 concatenates encoder output at cfg.decoder.skip_layer_index
    (spec: e^(4) → index 3, shape (B, d_model, N_LR)).

    FiLM is applied to the first cfg.decoder.n_film_stages stages (spec: stages 1-3).

    Args:
        cfg: Full OmegaConf config (top-level).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        mcfg = cfg.model
        d: int = mcfg.d_model                          # 256
        base: int = mcfg.decoder.base_channels         # 64
        factors: list[int] = list(mcfg.decoder.stage_factors)   # [5, 5, 2, 5]
        n_film: int = mcfg.decoder.n_film_stages       # 3
        self.skip_idx: int = mcfg.decoder.skip_layer_index      # 3

        # Channel widths at each stage boundary (input then output per stage)
        ch = [d, d, d // 2, d // 2, base]             # [256, 256, 128, 128, 64]

        assert len(factors) == len(ch) - 1, (
            f"stage_factors length ({len(factors)}) must equal number of stages ({len(ch)-1})"
        )

        # Stage 3 gets the encoder skip from skip_layer_index (d channels, N_LR resolution)
        # Stages 1, 2, 4 have no skip connections in the base spec.
        skip_channels_per_stage = [None, None, d, None]

        self.stages = nn.ModuleList([
            DecoderBlock(
                in_channels=ch[i],
                out_channels=ch[i + 1],
                upscale_factor=factors[i],
                d_model=d,
                skip_channels=skip_channels_per_stage[i],
                use_film=(i < n_film),
            )
            for i in range(len(factors))
        ])

        # Final projection: single output channel at HR resolution
        # k=7 provides a wider receptive field for the final waveform smoothing
        self.head = nn.Conv1d(base, 1, kernel_size=7, padding=3)

    def forward(
        self,
        z_patches: Tensor,       # (B, P, d_model)
        z_cls: Tensor,           # (B, d_model)
        skips: list[Tensor],     # [L × (B, d_model, N_LR)]
    ) -> Tensor:
        """
        Args:
            z_patches: (B, P=20,   d_model=256) — Transformer patch outputs
            z_cls:     (B,         d_model=256) — global CLS for FiLM conditioning
            skips:     list of L tensors, each (B, d_model=256, N_LR=1000)

        Returns:
            x_HR: (B, N_HR=5000) — reconstructed high-rate ECG signal
        """
        # Transpose from sequence convention to conv convention
        u = z_patches.transpose(1, 2)                  # (B, d_model, P) = (B, 256, 20)

        # Select skip tensor for stage 3 — safe even if encoder has fewer blocks
        skip3 = skips[min(self.skip_idx, len(skips) - 1)] if skips else None
        stage_skips = [None, None, skip3, None]

        for stage, skip in zip(self.stages, stage_skips):
            u = stage(u, z_cls, skip=skip)

        # (B, d_model, P) → (B, 256, 20) → ... → (B, 64, 5000) → (B, 1, 5000)
        u = self.head(u)                               # (B, 1, N_HR)
        return u.squeeze(1)                            # (B, N_HR=5000)
