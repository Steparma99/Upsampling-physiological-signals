"""
reconstruction.py — Reconstruction losses for Phase B generator training.

Three components (Eq. 55-57):
  L1    — pixel-wise mean absolute error                     (dominant term)
  FFT   — log-magnitude spectral L1                         (global frequency fidelity)
  STFT  — multi-scale short-time Fourier magnitude + log    (fine QRS + global P/T)

All three are returned separately so the training loop can log them individually.
The caller (phase_b.py) applies the loss weights from cfg.phase_b.loss_weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

# Numerical stability constant for log-magnitude (Eq. 56, spec ε = 1e-7)
_LOG_EPS: float = 1e-7


class ReconstructionLoss(nn.Module):
    """L1 pixel-wise + FFT log-magnitude L1 + multi-scale STFT loss.

    Args:
        cfg: Full OmegaConf config. Reads:
             cfg.phase_b.stft_window_lengths — list of STFT window sizes in HR samples.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.window_lengths: list[int] = list(cfg.phase_b.stft_window_lengths)

        # Pre-allocate Hann windows as buffers — one per STFT window size.
        # Using register_buffer ensures they move to the correct device with .to(device).
        for n in self.window_lengths:
            self.register_buffer(f"hann_{n}", torch.hann_window(n))

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            pred:   (B, N_HR) — generator output (z-score normalised)
            target: (B, N_HR) — ground-truth HR signal (same normalisation)

        Returns:
            (l1_loss, fft_loss, stft_loss) — all scalars, un-weighted.
        """
        l1_loss = self._l1(pred, target)
        fft_loss = self._fft(pred, target)
        stft_loss = self._multiscale_stft(pred, target)
        return l1_loss, fft_loss, stft_loss

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def _l1(self, pred: Tensor, target: Tensor) -> Tensor:
        """Pixel-wise L1: (1/N_HR) × ‖pred − target‖₁  (Eq. 55)."""
        return F.l1_loss(pred, target)

    def _fft(self, pred: Tensor, target: Tensor) -> Tensor:
        """Log-magnitude spectral L1 over the full signal (Eq. 56).

        Operates on the one-sided FFT magnitude. The log transform forces the
        network to synthesise high-frequency content above the LR Nyquist (50 Hz).
        """
        mag_pred = torch.fft.rfft(pred, dim=-1).abs()
        mag_tgt = torch.fft.rfft(target, dim=-1).abs()
        log_pred = torch.log(mag_pred + _LOG_EPS)
        log_tgt = torch.log(mag_tgt + _LOG_EPS)
        return F.l1_loss(log_pred, log_tgt)

    def _multiscale_stft(self, pred: Tensor, target: Tensor) -> Tensor:
        """Sum of per-scale (magnitude + log-magnitude) Frobenius losses (Eq. 57).

        For each window length n, computes:
            ‖|Sn(pred)| − |Sn(target)|‖_F  +  ‖log|Sn(pred)| − log|Sn(target)|‖_F

        where ‖·‖_F is the normalised Frobenius norm (= RMSE over freq×time bins),
        which keeps each scale's contribution comparable regardless of STFT resolution.
        """
        total = torch.zeros(1, device=pred.device, dtype=pred.dtype)
        for n in self.window_lengths:
            window: Tensor = getattr(self, f"hann_{n}")
            hop = n // 4

            mag_p = torch.stft(pred, n_fft=n, hop_length=hop, win_length=n,
                               window=window, center=False,
                               return_complex=True).abs()         # (B, F, T)
            mag_t = torch.stft(target, n_fft=n, hop_length=hop, win_length=n,
                               window=window, center=False,
                               return_complex=True).abs()         # (B, F, T)

            # Normalised Frobenius (RMSE): scale-invariant across window sizes
            mag_loss = (mag_p - mag_t).pow(2).mean().sqrt()

            log_p = torch.log(mag_p + _LOG_EPS)
            log_t = torch.log(mag_t + _LOG_EPS)
            log_loss = (log_p - log_t).pow(2).mean().sqrt()

            total = total + mag_loss + log_loss

        return total.squeeze(0)
