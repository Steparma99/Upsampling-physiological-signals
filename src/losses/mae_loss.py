import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MAELoss(nn.Module):
    """MAE reconstruction loss for Phase A pre-training.

    Computed only on masked patches. Per-patch z-score normalisation uses TARGET
    patch statistics, never prediction statistics (CLAUDE.md constraint #4).

    Args:
        fft_loss_weight: λ for FFT component (config: phase_a.mae.fft_loss_weight)
    """

    def __init__(self, fft_loss_weight: float = 0.1) -> None:
        super().__init__()
        self.fft_loss_weight = fft_loss_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            pred:   (B, P, patch_len) — decoder output for all P patch positions
            target: (B, P, patch_len) — original signal patches (ground truth)
            mask:   (B, P) bool — True at masked (reconstructed) positions

        Returns:
            (total_loss, rec_loss, fft_loss) — all scalars
        """
        # Gather only masked patches: (M_total, patch_len)
        pred_m = pred[mask]
        target_m = target[mask]

        # Per-patch statistics from TARGET only — constraint #4
        mu = target_m.mean(dim=-1, keepdim=True)           # (M_total, 1)
        sigma = target_m.std(dim=-1, keepdim=True) + 1e-6  # (M_total, 1)

        # Z-score normalise both with target stats: equivalent to ||pred-target||^2 / sigma^2
        pred_norm = (pred_m - mu) / sigma
        target_norm = (target_m - mu) / sigma

        rec_loss = F.mse_loss(pred_norm, target_norm)

        # Spectral component: L1 on FFT magnitude of raw (un-normalised) patches
        fft_pred = torch.fft.rfft(pred_m, dim=-1).abs()
        fft_target = torch.fft.rfft(target_m, dim=-1).abs()
        fft_loss = F.l1_loss(fft_pred, fft_target)

        total = rec_loss + self.fft_loss_weight * fft_loss
        return total, rec_loss, fft_loss
