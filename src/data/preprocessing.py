"""
preprocessing.py — ECG signal preprocessing for the super-resolution pipeline.

Five responsibilities:
1. CONSTANTS  — physical properties of the problem (not hyperparameters).
2. DOWNSAMPLING CHAIN  — Kaiser FIR anti-alias filter + decimation.
                         Only for non-native datasets (MIT-BIH, CPSC2018).
                         PTB-XL already ships dual-rate pairs: never decimate it.
3. Z-SCORE NORMALIZATION  — parameters computed on x_LR only (Eq. 3 PDF).
                             Same params applied to x_HR to preserve scale (Eq. 4).
4. R-PEAK DETECTION  — NeuroKit2 Hamilton 2002 as primary detector,
                        wfdb GQRS as first fallback, manual Pan-Tompkins last.
5. R-DISTANCE MAP  — d_map[m] = min_k|m - r_k| / mean_RR, shape (N_LR,) (Eq. 5 PDF).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from scipy.signal import filtfilt, firwin, kaiserord, resample_poly

try:
    import neurokit2 as nk
except ImportError:
    nk = None

from wfdb.processing import xqrs_detect

logger = logging.getLogger(__name__)


# =============================================================================
# 1. CONSTANTS
# =============================================================================
# Physical properties of the problem — not hyperparameters.
# Hyperparameters (d_model, n_layers, lr, …) live in configs/default.yaml.

FS_HR: int = 500          # Target sample rate (Hz)
FS_LR: int = 100          # Input sample rate (Hz)
UPSAMPLE_RATIO: int = 5   # R = FS_HR / FS_LR

WINDOW_SEC: float = 10.0  # Window duration in seconds

L_HR: int = 5000          # Samples per window at 500 Hz  (10 s × 500)
L_LR: int = 1000          # Samples per window at 100 Hz  (10 s × 100)


# =============================================================================
# 2. DOWNSAMPLING CHAIN
# =============================================================================
# D = Q ∘ N ∘ S ∘ F  (degradation chain, Phase 1 dataset preparation).
# Only the F (filter) and S (subsample) steps live here.
# This is applied to MIT-BIH and CPSC2018 — NEVER to PTB-XL, which already
# provides both 500 Hz and 100 Hz recordings.
#
# Kaiser FIR rationale over Butterworth IIR:
#   - Linear phase → zero morphological distortion (QRS intervals unchanged)
#   - Exact stopband spec via kaiserord() without order iteration
#   - FIR is unconditionally stable; high-order IIR can have numerical issues

def design_antialias_filter(
    cutoff_hz: float = 50.0,
    fs_hz: float = FS_HR,
    attenuation_db: float = 80.0,
    transition_width_hz: float = 10.0,
) -> NDArray[np.float64]:
    """Design a Kaiser-windowed FIR lowpass anti-aliasing filter.

    Cutoff at 50 Hz = Nyquist of the 100 Hz LR signal.
    kaiserord() automatically computes the optimal order and beta parameter.

    Args:
        cutoff_hz: Passband edge in Hz.
        fs_hz: Sampling rate of the input signal in Hz.
        attenuation_db: Minimum stopband attenuation in dB.
        transition_width_hz: Transition band width in Hz.

    Returns:
        1-D array of FIR filter coefficients (float64).
    """
    nyq = fs_hz / 2.0
    numtaps, beta = kaiserord(attenuation_db, transition_width_hz / nyq)
    if numtaps % 2 == 0:
        numtaps += 1  # odd length for symmetric (linear-phase) FIR
    taps = firwin(numtaps, cutoff_hz / nyq, window=("kaiser", beta))
    return taps


def downsample_signal(
    signal_hr: NDArray[np.floating],
    decimation_factor: int = UPSAMPLE_RATIO,
    fs_hr: float = FS_HR,
    cutoff_hz: float = 50.0,
    attenuation_db: float = 80.0,
) -> NDArray[np.float32]:
    """Apply anti-aliasing FIR filter then decimate.

    Constraint: do NOT call this on PTB-XL records — they already have
    native 100 Hz recordings. This is strictly for MIT-BIH and CPSC2018
    where 100 Hz must be simulated from 500 Hz.

    The filter is applied zero-phase (filtfilt) to avoid morphological distortion.

    Args:
        signal_hr: Input signal at fs_hr Hz, shape (L_HR,).
        decimation_factor: Integer downsampling factor (default 5 → 500→100 Hz).
        fs_hr: Sampling rate of the input signal.
        cutoff_hz: Anti-aliasing cutoff frequency.
        attenuation_db: Stopband attenuation in dB.

    Returns:
        Decimated signal at fs_hr / decimation_factor Hz, shape (L_LR,).
    """
    taps = design_antialias_filter(cutoff_hz, fs_hr, attenuation_db)
    decimated = resample_poly(
        signal_hr.astype(np.float64),
        up=1,
        down=decimation_factor,
        window=taps,
        padtype="line",
    )
    return decimated.astype(np.float32)


# =============================================================================
# 3. Z-SCORE NORMALIZATION
# =============================================================================
# Parameters mu_w and sigma_w are always computed on x_LR.
# The same parameters are applied to x_HR to preserve the amplitude scale
# between the two paired signals.
# In inference only x_LR is available — this is why the stats come from LR only.

def zscore_normalize(
    signal_lr: NDArray[np.floating],
) -> tuple[NDArray[np.float32], float, float]:
    """Compute z-score parameters from x_LR and return the normalized signal.

    sigma_w uses ddof=1 (N-1 denominator) as in Eq. 3 of the PDF.
    sigma_w is clamped to a minimum of 1e-8 to avoid division by zero on
    flat windows (which will be rejected by artifact rejection anyway).

    Args:
        signal_lr: Raw LR window, shape (L_LR,).

    Returns:
        Tuple (x_LR_hat, mu_w, sigma_w) where x_LR_hat has dtype float32.
    """
    mu_w = float(np.mean(signal_lr))
    sigma_w = float(np.std(signal_lr, ddof=1))
    sigma_w = max(sigma_w, 1e-8)
    x_lr_hat = ((signal_lr - mu_w) / sigma_w).astype(np.float32)
    return x_lr_hat, mu_w, sigma_w


def apply_normalization(
    signal: NDArray[np.floating],
    mu_w: float,
    sigma_w: float,
) -> NDArray[np.float32]:
    """Apply pre-computed z-score parameters to a signal.

    Used on x_HR: parameters were computed on the paired x_LR window.
    Do NOT recompute stats from signal itself — that would break the
    amplitude relationship between x_LR and x_HR.

    Args:
        signal: Raw HR window, shape (L_HR,), or any array.
        mu_w: Mean computed on the paired x_LR window.
        sigma_w: Std computed on the paired x_LR window.

    Returns:
        Normalized signal as float32.
    """
    return ((signal - mu_w) / sigma_w).astype(np.float32)


def denormalize(
    signal_normalized: NDArray[np.floating] | torch.Tensor,
    mu_w: float | torch.Tensor,
    sigma_w: float | torch.Tensor,
) -> NDArray[np.float32] | torch.Tensor:
    """Invert z-score normalization: x = x_hat * sigma_w + mu_w.

    Accepts both NumPy arrays and PyTorch tensors so it can be used both
    in evaluation scripts (NumPy) and inside the training loop (GPU tensors).

    Args:
        signal_normalized: Model output or any z-score normalized signal.
        mu_w: Mean used during normalization.
        sigma_w: Std used during normalization.

    Returns:
        Denormalized signal in the original amplitude scale.
    """
    return signal_normalized * sigma_w + mu_w


# =============================================================================
# 4. R-PEAK DETECTION
# =============================================================================
# Three-tier detection chain in order of robustness:
#   1. NeuroKit2 Hamilton 2002 — most robust at low fs and on arrhythmias.
#      Handles low-amplitude QRS, noise, and irregular rhythms better than
#      Pan-Tompkins. Preferred method.
#   2. wfdb GQRS — battle-tested detector from PhysioNet, good fallback.
#   3. Manual Pan-Tompkins — self-contained implementation, no external deps,
#      used only if both library methods fail.

def detect_r_peaks(
    signal: NDArray[np.floating],
    fs: float = FS_LR,
) -> NDArray[np.intp]:
    """Detect R-peaks in an ECG signal.

    Tries wfdb XQRS, then NeuroKit2 Hamilton 2002, then manual Pan-Tompkins.
    All three return 0-based sample indices in the same domain as `signal`.

    Args:
        signal: 1-D ECG signal, any amplitude (z-score not required).
        fs: Sampling rate in Hz.

    Returns:
        Array of integer indices of detected R-peaks.
    """
    sig = np.asarray(signal, dtype=np.float64)

    # --- Primary: wfdb XQRS ---
    try:
        peaks = xqrs_detect(sig=sig, fs=float(fs), verbose=False)
        if len(peaks) > 0:
            return np.array(peaks, dtype=np.intp)
        raise ValueError("zero peaks from XQRS")
    except Exception as exc:
        logger.warning("nk.ecg_peaks failed (%s) — trying wfdb GQRS.", exc)

    if nk is None:
        return _detect_r_peaks_pantompkins(sig, fs)

    try:
        _, info = nk.ecg_peaks(sig, sampling_rate=int(fs), method="hamilton2002")
        peaks = np.array(info["ECG_R_Peaks"], dtype=np.intp)
        if len(peaks) > 0:
            return peaks
    except Exception as exc:
        logger.warning("nk.ecg_peaks failed (%s) - using manual Pan-Tompkins.", exc)
        return _detect_r_peaks_pantompkins(sig, fs)

    return _detect_r_peaks_pantompkins(sig, fs)

    # --- Fallback 1: wfdb GQRS ---
    try:
        peaks = gqrs_detect(sig, fs=float(fs))
        if len(peaks) > 0:
            return np.array(peaks, dtype=np.intp)
        raise ValueError("zero peaks from GQRS")
    except Exception as exc:
        logger.warning("wfdb gqrs_detect failed (%s) — using manual Pan-Tompkins.", exc)

    # --- Fallback 2: manual Pan-Tompkins ---
    return _detect_r_peaks_pantompkins(sig, fs)


def _refine_r_peak_positions(
    sig: NDArray[np.float64],
    peaks: NDArray[np.intp],
    fs: float,
) -> NDArray[np.intp]:
    """Refine candidate detections onto the local raw-signal maximum."""
    if len(peaks) == 0:
        return peaks.astype(np.intp, copy=False)

    radius = max(int(0.075 * fs), 1)
    refined: list[int] = []
    for p in np.asarray(peaks, dtype=np.intp):
        lo = max(0, int(p) - radius)
        hi = min(len(sig), int(p) + radius + 1)
        refined.append(lo + int(np.argmax(sig[lo:hi])))

    return np.array(sorted(set(refined)), dtype=np.intp)


def detect_r_peaks(
    signal: NDArray[np.floating],
    fs: float = FS_LR,
) -> NDArray[np.intp]:
    """Detect R-peaks and refine them onto the visual R apex."""
    from wfdb.processing import gqrs_detect as _gqrs_detect

    sig = np.asarray(signal, dtype=np.float64)

    try:
        peaks = xqrs_detect(sig=sig, fs=float(fs), verbose=False)
        if len(peaks) > 0:
            return _refine_r_peak_positions(sig, np.array(peaks, dtype=np.intp), fs)
        raise ValueError("zero peaks from XQRS")
    except Exception as exc:
        logger.warning("wfdb xqrs_detect failed (%s) - trying NeuroKit Hamilton.", exc)

    if nk is not None:
        try:
            _, info = nk.ecg_peaks(sig, sampling_rate=int(fs), method="hamilton2002")
            peaks = np.array(info["ECG_R_Peaks"], dtype=np.intp)
            if len(peaks) > 0:
                return _refine_r_peak_positions(sig, peaks, fs)
            raise ValueError("zero peaks from NeuroKit Hamilton")
        except Exception as exc:
            logger.warning("nk.ecg_peaks failed (%s) - trying wfdb GQRS.", exc)

    try:
        peaks = _gqrs_detect(sig, fs=float(fs))
        if len(peaks) > 0:
            return _refine_r_peak_positions(sig, np.array(peaks, dtype=np.intp), fs)
        raise ValueError("zero peaks from GQRS")
    except Exception as exc:
        logger.warning("wfdb gqrs_detect failed (%s) - using manual Pan-Tompkins.", exc)

    return _detect_r_peaks_pantompkins(sig, fs)


def _detect_r_peaks_pantompkins(
    sig: NDArray[np.float64],
    fs: float,
) -> NDArray[np.intp]:
    """Pan-Tompkins (1985) simplified — final fallback, no external dependencies.

    Four-stage pipeline:
      1. Bandpass 5–15 Hz  (FIR, zero-phase)
      2. First derivative
      3. Squaring
      4. Moving-window integration + adaptive threshold (EMA, 200 ms refractory)
    Refinement step corrects the detection delay by searching the original
    signal maximum in a ±75 ms window around each candidate.
    """
    nyq = fs / 2.0
    low = 5.0 / nyq
    high = min(15.0 / nyq, 0.99)
    numtaps = int(0.15 * fs) | 1  # ensure odd
    bp = firwin(numtaps, [low, high], pass_zero=False)
    filtered = filtfilt(bp, 1.0, sig)

    squared = np.diff(filtered) ** 2

    win_len = max(int(0.15 * fs), 1)
    integrated = np.convolve(squared, np.ones(win_len) / win_len, mode="same")

    init_win = min(int(2.0 * fs), len(integrated))
    threshold = 0.6 * np.max(integrated[:init_win])
    refractory = int(0.20 * fs)

    peaks: list[int] = []
    last = -refractory
    for i in range(1, len(integrated) - 1):
        if (integrated[i] > threshold
                and i - last > refractory
                and integrated[i] >= integrated[i - 1]
                and integrated[i] >= integrated[i + 1]):
            peaks.append(i)
            last = i
            threshold = 0.125 * integrated[i] + 0.875 * threshold

    radius = int(0.075 * fs)
    refined: list[int] = []
    for p in peaks:
        lo = max(0, p - radius)
        hi = min(len(sig), p + radius + 1)
        refined.append(lo + int(np.argmax(sig[lo:hi])))

    return np.array(refined, dtype=np.intp)


# =============================================================================
# 5. R-DISTANCE MAP
# =============================================================================
# For each sample m in the LR window, compute the distance to the nearest
# R-peak normalized by the mean R-R interval (Eq. 5 PDF):
#
#   d_map[m] = min_k |m - r_k| / mean_RR   ∈ [0, ~0.5]
#
# The map is 0 at each R-peak and grows linearly toward the midpoint between
# two consecutive beats. It gives the network an explicit phase signal:
# different parts of the cardiac cycle require different synthesis strategies.

def compute_r_distance_map(
    r_peaks: NDArray[np.intp],
    length: int = L_LR,
) -> NDArray[np.float32]:
    """Compute the normalized R-distance map (Eq. 5 PDF).

    Args:
        r_peaks: R-peak indices in the LR sample domain.
        length: Length of the LR window (default L_LR = 1000).

    Returns:
        d_map of shape (length,), values in [0, ~0.5].
        All-zeros if no R-peaks are found (safe fallback for noisy windows).
    """
    if len(r_peaks) == 0:
        logger.warning("No R-peaks found — d_map will be all zeros.")
        return np.zeros(length, dtype=np.float32)

    # Mean R-R interval in samples.
    # With a single peak we cannot compute R-R; default to the full window
    # length (~1 s at 100 Hz ≈ 60 bpm), which is physiologically reasonable.
    if len(r_peaks) >= 2:
        mean_rr = float(np.mean(np.diff(r_peaks).astype(np.float64)))
    else:
        mean_rr = float(length)

    # Distance from each sample to the nearest R-peak via broadcasting.
    # Shape: (length, n_peaks) → min over n_peaks axis.
    peak_mask = np.ones(length, dtype=bool)
    valid_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < length)]
    peak_mask[valid_peaks] = False
    min_dist = distance_transform_edt(peak_mask)

    return (min_dist / mean_rr).astype(np.float32)
