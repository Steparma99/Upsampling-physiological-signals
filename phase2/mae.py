"""
Masked Autoencoder (MAE) components for Phase 2 pre-training.

Components:
  create_mask          — rhythm-aware boolean mask over patches
  get_r_peak_patches   — convert PQRST annotations to patch indices
  apply_mask_to_tokens — replace masked patch tokens with a learnable token
  MAEDecoder           — lightweight Transformer decoder for patch reconstruction
  mae_loss             — MSE + spectral (FFT) loss on masked patches only
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Masking utilities
# ---------------------------------------------------------------------------

def create_mask(
    num_patches: int,
    mask_ratio: float,
    r_peak_patches: list[int] | None = None,
    weight_qrs: float = 2.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Create a boolean mask over ``num_patches`` patches (True = masked).

    When ``r_peak_patches`` is provided the sampling is rhythm-aware:
    patches containing an R-peak are assigned weight ``weight_qrs`` so
    the model is more likely to be asked to reconstruct QRS complexes
    (the most diagnostically informative region).

    Args:
        num_patches:    Number of patches (e.g. 20).
        mask_ratio:     Fraction of patches to mask (e.g. 0.40).
        r_peak_patches: List of patch indices that contain an R-peak.
        weight_qrs:     Relative sampling weight for QRS-bearing patches.
        rng:            numpy Generator; a fresh default_rng is created if None.

    Returns:
        mask: bool ndarray shape (num_patches,), True = masked.

    Note: operates on a single sample.  Call per-sample inside the DataLoader
    to produce different masks across the batch.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_mask = max(1, int(mask_ratio * num_patches))

    weights = np.ones(num_patches, dtype=np.float64)
    if r_peak_patches:
        for idx in r_peak_patches:
            if 0 <= idx < num_patches:
                weights[idx] = weight_qrs

    weights /= weights.sum()   # normalise to probability distribution

    masked_indices = rng.choice(num_patches, size=n_mask, replace=False, p=weights)
    mask = np.zeros(num_patches, dtype=bool)
    mask[masked_indices] = True
    return mask


def get_r_peak_patches(
    a: np.ndarray,
    num_patches: int,
    patch_len: int,
    r_peak_type: int = 0,
    upsample_factor: int = 5,
) -> list[int]:
    """
    Convert PQRST annotation array to a list of LR patch indices that
    contain at least one R-peak.

    Args:
        a:               Annotation array shape (K, 2).
                         Column 0 = peak type (0 = R-peak by Phase 1 convention).
                         Column 1 = peak position in HR samples (500 Hz space).
        num_patches:     Total number of LR patches (e.g. 20).
        patch_len:       LR samples per patch (e.g. 50).
        r_peak_type:     Integer code for R-peaks in ``a[:, 0]``.
        upsample_factor: HR/LR ratio = 5  (500 Hz / 100 Hz).

    Returns:
        List of unique patch indices in [0, num_patches − 1].
    """
    if a is None or len(a) == 0:
        return []

    r_mask = a[:, 0].astype(int) == r_peak_type
    r_positions_hr = a[r_mask, 1].astype(int)

    patch_indices = []
    for pos_hr in r_positions_hr:
        pos_lr = pos_hr // upsample_factor
        patch_idx = pos_lr // patch_len
        if 0 <= patch_idx < num_patches:
            patch_indices.append(patch_idx)

    return list(set(patch_indices))


# ---------------------------------------------------------------------------
# Mask application
# ---------------------------------------------------------------------------

def apply_mask_to_tokens(
    z_seq: torch.Tensor,
    mask: torch.Tensor,
    mask_token: nn.Parameter,
) -> torch.Tensor:
    """
    Replace masked patch tokens with a learnable mask token.

    Args:
        z_seq:      (B, num_patches+1, d_model)
                    Position 0 is the physiological conditioning token — it is
                    NEVER masked; only positions [1:] (the CNN patch tokens) are
                    candidates for masking.
        mask:       BoolTensor (B, num_patches)  — True = masked.
        mask_token: Learnable Parameter (d_model,) shared across all positions.

    Returns:
        z_masked: (B, num_patches+1, d_model) with masked positions replaced.
    """
    z_masked = z_seq.clone()
    # z_seq[:, 1:, :] has shape (B, num_patches, d_model)
    # mask has shape (B, num_patches) → selects (N_masked_total, d_model) rows
    z_masked[:, 1:, :][mask] = mask_token
    return z_masked


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class MAEDecoder(nn.Module):
    """
    Lightweight Transformer decoder for MAE patch reconstruction.

    Used ONLY during pre-training.  Discarded before Phase 3
    (only the encoder is exported via export_encoder_for_phase3).

    Architecture:
      1. Input projection: Linear(d_model, decoder_d_model)
      2. Transformer stack:  N_dec × TransformerEncoderLayer
         (same architecture as the encoder — bidirectional, Pre-LN)
      3. Output head: Linear(decoder_d_model, patch_len)
         applied ONLY to patch positions [1:], not the physio token.

    Input:  z_masked  (B, num_patches+1, d_model)
    Output: x_rec     (B, num_patches, patch_len)
            — reconstruction of ALL patches (loss is computed only on
              masked ones, but reconstructing all keeps the implementation
              simple and avoids conditional indexing in the decoder).
    """

    def __init__(
        self,
        d_model: int,
        decoder_d_model: int,
        num_layers: int,
        num_heads: int,
        patch_len: int,
    ) -> None:
        super().__init__()
        self.proj_in = nn.Linear(d_model, decoder_d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=decoder_d_model,
            nhead=num_heads,
            dim_feedforward=decoder_d_model * 4,   # standard 4× expansion
            dropout=0.0,                            # no dropout in decoder
            activation="gelu",
            batch_first=True,
            norm_first=True,                        # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(decoder_d_model, patch_len)

    def forward(self, z_masked: torch.Tensor) -> torch.Tensor:
        # z_masked: (B, num_patches+1, d_model)
        x = self.proj_in(z_masked)           # (B, num_patches+1, decoder_d_model)
        x = self.transformer(x)              # (B, num_patches+1, decoder_d_model)
        x_rec = self.head(x[:, 1:, :])      # (B, num_patches, patch_len)
        return x_rec


# ---------------------------------------------------------------------------
# MAE loss
# ---------------------------------------------------------------------------

def mae_loss(
    x_rec: torch.Tensor,
    x_lr: torch.Tensor,
    mask: torch.Tensor,
    lambda_fft: float = 0.1,
) -> torch.Tensor:
    """
    Compute the MAE reconstruction loss on masked patches only.

    Loss = MSE_masked + λ_fft · FFT_masked

    The per-patch z-score normalisation (using the *original* patch stats,
    not the reconstructed ones) prevents the model from trivially minimising
    loss by predicting a flat line.

    Args:
        x_rec:      (B, num_patches, patch_len) — decoder output.
        x_lr:       (B, L)                      — original LR signal.
        mask:       BoolTensor (B, num_patches)  — True = masked patch.
        lambda_fft: Weight for the spectral component of the loss.

    Returns:
        loss: Scalar tensor.
    """
    patch_len = x_rec.shape[-1]

    # 1. Patchify x_lr using unfold (non-overlapping)
    x_patches = x_lr.unfold(-1, patch_len, patch_len)  # (B, num_patches, patch_len)

    # 2. Per-patch z-score normalisation using x_patches statistics
    #    IMPORTANT: normalise x_rec with x_patches stats (not x_rec stats)
    #    to prevent trivial zero-loss solutions.
    mu_patch  = x_patches.mean(dim=-1, keepdim=True)           # (B, num_patches, 1)
    std_patch = x_patches.std(dim=-1, keepdim=True) + 1e-6     # (B, num_patches, 1)
    x_norm     = (x_patches - mu_patch) / std_patch            # (B, num_patches, patch_len)
    x_rec_norm = (x_rec     - mu_patch) / std_patch            # same scaling applied to rec

    # 3. MSE loss on masked patches
    #    diff_sq[mask] selects (N_masked_total, patch_len) elements
    diff_sq  = (x_rec_norm - x_norm) ** 2                      # (B, num_patches, patch_len)
    loss_rec = diff_sq[mask].mean()

    # 4. Spectral (FFT magnitude) loss on masked patches
    fft_rec  = torch.fft.rfft(x_rec_norm, dim=-1).abs()        # (B, num_patches, patch_len//2+1)
    fft_orig = torch.fft.rfft(x_norm,     dim=-1).abs()
    fft_diff = (fft_rec - fft_orig) ** 2
    loss_fft = fft_diff[mask].mean()

    return loss_rec + lambda_fft * loss_fft
