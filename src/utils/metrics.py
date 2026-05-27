"""
metrics.py — Quantitative evaluation metrics for ECG super-resolution.

Five functions:

  compute_snr()       — Signal-to-Noise Ratio in dB
  compute_prd()       — Percent Root-mean-square Difference (%)
  compute_rmse()      — Root Mean Square Error (normalised units)
  compute_ssim()      — 1-D Structural Similarity Index (0–1)
  evaluate_batch()    — aggregates all metrics for a batch, returns a dict

All per-sample metrics are averaged across the batch before returning.
All functions accept (B, N) tensors; N is the signal length.
evaluate_batch() operates under torch.no_grad() internally.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Individual metrics
# =============================================================================

def compute_snr(pred: Tensor, target: Tensor) -> Tensor:
    """Signal-to-Noise Ratio in dB, averaged over the batch.

    SNR_i = 10 × log₁₀( ‖target_i‖² / ‖target_i − pred_i‖² )

    A perfect reconstruction gives SNR → +∞; the noise floor is clamped to 1e-10
    to avoid log(0).  Higher is better.

    Args:
        pred:   (B, N) — model output (any normalisation).
        target: (B, N) — ground-truth signal.

    Returns:
        Scalar mean SNR in dB over the batch.
    """
    signal_power = target.pow(2).sum(dim=-1)                           # (B,)
    noise_power  = (target - pred).pow(2).sum(dim=-1).clamp(min=1e-10) # (B,)
    snr = 10.0 * torch.log10(signal_power / noise_power)
    return snr.mean()


def compute_prd(pred: Tensor, target: Tensor) -> Tensor:
    """Percent Root-mean-square Difference (%), averaged over the batch.

    PRD_i = 100 × ‖target_i − pred_i‖₂ / ‖target_i‖₂

    Standard metric in ECG compression / reconstruction literature.
    A PRD of 0 % is a perfect reconstruction; lower is better.
    The denominator is clamped to 1e-8 to handle near-silent segments.

    Args:
        pred:   (B, N) — model output.
        target: (B, N) — ground-truth signal.

    Returns:
        Scalar mean PRD (%) over the batch.
    """
    error_norm  = (target - pred).pow(2).sum(dim=-1).sqrt()            # (B,)
    signal_norm = target.pow(2).sum(dim=-1).sqrt().clamp(min=1e-8)     # (B,)
    return (100.0 * error_norm / signal_norm).mean()


def compute_rmse(pred: Tensor, target: Tensor) -> Tensor:
    """Root Mean Square Error, averaged over the batch.

    RMSE_i = √( (1/N) × Σ_t (pred_{i,t} − target_{i,t})² )

    Reported in the same units as the input signals (normalised z-score units
    when called on normalised predictions; see evaluate_batch() for the option
    to report in original amplitude units).

    Args:
        pred:   (B, N) — model output.
        target: (B, N) — ground-truth signal.

    Returns:
        Scalar mean RMSE over the batch.
    """
    return (pred - target).pow(2).mean(dim=-1).sqrt().mean()


def compute_ssim(
    pred: Tensor,
    target: Tensor,
    window_size: int = 63,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tensor:
    """1-D Structural Similarity Index (SSIM), averaged over the batch.

    SSIM is computed with a sliding box window (avg_pool1d) over the temporal
    axis, following the standard formula (Wang et al. 2004) adapted to 1-D:

        SSIM(x,y) = (2μ_x μ_y + C1)(2σ_xy + C2)
                    ────────────────────────────────────────
                    (μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)

    Stability constants:
        C1 = (k1 × L)²,   C2 = (k2 × L)²
        L  = per-sample dynamic range of target (max − min)

    Using per-sample L rather than a fixed value gives a scale-invariant
    metric that is robust to the wide amplitude variation in ECG recordings.

    Args:
        pred:        (B, N) — model output.
        target:      (B, N) — ground-truth signal.
        window_size: Sliding-window length in samples (default 63 ≈ 126 ms at 500 Hz).
                     Must be odd so that symmetric padding preserves signal length.
        k1:          Stability constant (default 0.01).
        k2:          Stability constant (default 0.03).

    Returns:
        Scalar mean SSIM (range nominally −1 to 1; perfect = 1) over the batch.
    """
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")

    B    = pred.shape[0]
    pad  = window_size // 2

    # Per-sample dynamic range → per-sample stability constants (B, 1, 1)
    L  = (target.amax(dim=-1) - target.amin(dim=-1)).clamp(min=1e-8)
    C1 = (k1 * L).pow(2).view(B, 1, 1)
    C2 = (k2 * L).pow(2).view(B, 1, 1)

    # (B, N) → (B, 1, N) for avg_pool1d
    p = pred.unsqueeze(1)
    t = target.unsqueeze(1)

    # Local means
    mu_p = F.avg_pool1d(p, kernel_size=window_size, stride=1, padding=pad)
    mu_t = F.avg_pool1d(t, kernel_size=window_size, stride=1, padding=pad)

    mu_p2 = mu_p.pow(2)
    mu_t2 = mu_t.pow(2)
    mu_pt = mu_p * mu_t

    # Local variances and cross-covariance: Var[X] = E[X²] − E[X]²
    sigma_p2 = F.avg_pool1d(p.pow(2), kernel_size=window_size, stride=1, padding=pad) - mu_p2
    sigma_t2 = F.avg_pool1d(t.pow(2), kernel_size=window_size, stride=1, padding=pad) - mu_t2
    sigma_pt = F.avg_pool1d(p * t,    kernel_size=window_size, stride=1, padding=pad) - mu_pt

    # SSIM map: (B, 1, N)
    num      = (2.0 * mu_pt + C1) * (2.0 * sigma_pt + C2)
    denom    = (mu_p2 + mu_t2 + C1) * (sigma_p2 + sigma_t2 + C2).clamp(min=1e-12)
    ssim_map = num / denom

    return ssim_map.mean()


# =============================================================================
# Batch aggregator
# =============================================================================

@torch.no_grad()
def evaluate_batch(
    pred: Tensor,
    target: Tensor,
    mu_w: Tensor | None = None,
    sigma_w: Tensor | None = None,
    ssim_window_size: int = 63,
) -> dict[str, float]:
    """Compute all evaluation metrics for a batch of ECG predictions.

    Metrics are computed on the normalised signals (pred, target) and optionally
    also on the original-amplitude signals when mu_w / sigma_w are provided.

    Args:
        pred:             (B, N_HR) — model output, z-score normalised.
        target:           (B, N_HR) — ground-truth HR signal, same normalisation.
        mu_w:             (B,) — per-sample normalisation mean (optional).
        sigma_w:          (B,) — per-sample normalisation std  (optional).
        ssim_window_size: Box-window length for SSIM (default 63).

    Returns:
        Dict with keys:
          snr_db  — mean SNR in dB (higher better)
          prd     — mean PRD in %  (lower better)
          rmse    — mean RMSE in normalised units (lower better)
          ssim    — mean SSIM  (higher better, max 1)
          rmse_mv — mean RMSE in original amplitude units (only if mu_w/sigma_w given)
    """
    result: dict[str, float] = {
        "snr_db": compute_snr(pred, target).item(),
        "prd":    compute_prd(pred, target).item(),
        "rmse":   compute_rmse(pred, target).item(),
        "ssim":   compute_ssim(pred, target, window_size=ssim_window_size).item(),
    }

    if mu_w is not None and sigma_w is not None:
        # Denormalise: x_orig = x_norm * sigma_w + mu_w
        mu    = mu_w.to(pred.device).unsqueeze(-1)      # (B, 1)
        sigma = sigma_w.to(pred.device).unsqueeze(-1)   # (B, 1)
        pred_raw   = pred   * sigma + mu
        target_raw = target * sigma + mu
        result["rmse_mv"] = compute_rmse(pred_raw, target_raw).item()

    return result
