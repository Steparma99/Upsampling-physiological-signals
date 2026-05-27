"""
morphological.py — Morphological loss for Phase B generator training.

Focuses MSE on clinically relevant PQRST landmark windows (Eq. 58):

    L_morph = (1/|A|) × Σ_{(t,w)∈A} ‖x̂_HR[t−w:t+w] − x_HR[t−w:t+w]‖²₂

where A is the set of PQRST landmarks (position t, half-window w) for the batch.
The loss is skipped entirely when annotations are None (PTB-XL with no annotations,
or batches where the annotation tensor is unavailable).

Annotation format
-----------------
Tensor of shape (B, K, 2):
    [:, :, 0] — landmark positions in HR sample indices (int)
    [:, :, 1] — per-landmark half-window sizes in HR samples (int)
    Position < 0 marks a padded / missing landmark — excluded from the loss.

When the dataset provides only positions (no per-landmark window sizes), pass a
(B, K) tensor; the module uses cfg.phase_b.morphological_half_window for all landmarks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor


class MorphologicalLoss(nn.Module):
    """MSE on PQRST landmark windows.

    Returns a scalar loss when valid annotations are present, or a zero tensor
    (no gradient) when annotations is None or contains no valid landmarks.

    Args:
        cfg: Full OmegaConf config. Reads:
             cfg.phase_b.morphological_half_window — fallback half-window size (HR samples)
             used when annotations are provided as (B, K) positions without window sizes.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.default_half_window: int = int(cfg.phase_b.morphological_half_window)

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        annotations: Tensor | None,
    ) -> Tensor:
        """
        Args:
            pred:        (B, N_HR) — generator output
            target:      (B, N_HR) — ground-truth HR signal
            annotations: (B, K, 2) with [position, half_window] per landmark,
                         OR (B, K)  with positions only (uses default_half_window),
                         OR None    (loss is skipped — returns 0.0 gradient-free).

        Returns:
            Scalar MSE loss over all valid landmark windows.
            Returns a zero tensor attached to pred's graph when no valid annotations.
        """
        if annotations is None:
            return pred.sum() * 0.0   # zero with gradient — safe to add to total loss

        # Normalise shape: (B, K) → (B, K, 2) with default half-window
        if annotations.dim() == 2:
            hw = torch.full(
                (*annotations.shape, 1),
                self.default_half_window,
                dtype=annotations.dtype,
                device=annotations.device,
            )
            annotations = torch.cat([annotations.unsqueeze(-1), hw], dim=-1)
            # annotations: (B, K, 2)

        positions  = annotations[:, :, 0]   # (B, K) — int positions in HR samples
        half_wins  = annotations[:, :, 1]   # (B, K) — half-window sizes
        valid_mask = positions >= 0          # (B, K) — True for real landmarks

        if not valid_mask.any():
            return pred.sum() * 0.0

        N_HR = pred.shape[-1]
        total_mse = torch.zeros(1, device=pred.device, dtype=pred.dtype)
        n_valid = 0

        # Iterate over landmarks — K is small (≤ ~60 per window), loop is acceptable
        B, K = positions.shape
        for b in range(B):
            for k in range(K):
                if not valid_mask[b, k]:
                    continue
                t = int(positions[b, k].item())
                w = int(half_wins[b, k].item())
                lo = max(0, t - w)
                hi = min(N_HR, t + w)
                if hi <= lo:
                    continue
                window_pred = pred[b, lo:hi]
                window_tgt = target[b, lo:hi]
                # Per-window MSE normalised by window length
                total_mse = total_mse + (window_pred - window_tgt).pow(2).mean()
                n_valid += 1

        if n_valid == 0:
            return pred.sum() * 0.0

        return (total_mse / n_valid).squeeze(0)
