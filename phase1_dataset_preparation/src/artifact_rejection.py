"""
Artifact rejection for ECG windows.

Four quality checks (Section 4 of the project spec):
  1. Saturation     — clipping / motion artifacts
  2. Variance       — flat signal or signal explosion
  3. SQI            — spectral quality index (Welch periodogram)
  4. RMSSD          — excessive RR interval variability (Pan-Tompkins)

Each check returns True when the window PASSES (is acceptable).
`is_valid_window` runs all four and returns (is_valid, rejection_reason).
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Tuple

from .pan_tompkins import detect_r_peaks


# ---------------------------------------------------------------------------
# Check 1 — Saturation
# ---------------------------------------------------------------------------

def check_saturation(
    win_hr_norm: np.ndarray,
    threshold: float = 5.0,
    max_fraction: float = 0.01,
) -> bool:
    """
    Reject window if more than max_fraction of samples exceed the z-score threshold.

    Real ADC clipping leaves many consecutive samples stuck at the rail → many
    samples will exceed the threshold.  Normal ECG R-peaks produce only a handful
    of samples above 5 σ and must not be rejected, so a single-sample max check
    is inappropriate here.

    Criteria: mean(|x̂_HR| > τ_sat) > max_fraction  →  reject

    Parameters
    ----------
    win_hr_norm : np.ndarray
        Z-score normalized HR window.
    threshold : float
        τ_sat (default 5).
    max_fraction : float
        Maximum allowed fraction of samples above threshold (default 1 %).

    Returns
    -------
    bool : True if window passes (no saturation detected).
    """
    fraction_clipped = float(np.mean(np.abs(win_hr_norm) > threshold))
    return fraction_clipped <= max_fraction


# ---------------------------------------------------------------------------
# Check 2 — Variance
# ---------------------------------------------------------------------------

def check_variance(
    sigma_w: float,
    min_std: float = 0.05,
    max_std: float = 5.0,
) -> bool:
    """
    Reject window if the HR standard deviation is outside physiological range.

    Criteria: σ_w < σ_min  (flat signal)  →  reject
              σ_w > σ_max  (explosive)    →  reject

    Parameters
    ----------
    sigma_w : float
        Per-window standard deviation of x_HR (in original mV units).
    min_std : float
        Minimum acceptable σ (mV). Default 0.05 mV.
    max_std : float
        Maximum acceptable σ (mV). Default 5.0 mV.

    Returns
    -------
    bool : True if window passes.
    """
    return bool(min_std <= sigma_w <= max_std)


# ---------------------------------------------------------------------------
# Check 3 — Signal Quality Index (SQI)
# ---------------------------------------------------------------------------

def compute_sqi(
    win_hr: np.ndarray,
    fs_hr: float,
    ecg_band: Tuple[float, float] = (0.5, 45.0),
) -> float:
    """
    Compute spectral quality index via Welch's periodogram.

    SQI = ∫_{0.5}^{45} S_xx(f) df  /  ∫_0^{f_N} S_xx(f) df

    High SQI → most power is in the physiological ECG band.
    Low SQI  → noise, muscle artifact, or non-ECG signal dominates.

    Parameters
    ----------
    win_hr : np.ndarray
        HR window (raw, not normalized).
    fs_hr : float
        Sampling frequency of the HR window (Hz).
    ecg_band : tuple
        (f_low, f_high) of the ECG physiological band (Hz).

    Returns
    -------
    sqi : float in [0, 1].
    """
    nperseg = min(256, len(win_hr) // 4)
    freqs, psd = sp_signal.welch(win_hr, fs=fs_hr, nperseg=nperseg)

    total_power = float(np.trapezoid(psd, freqs))
    if total_power < 1e-30:
        return 0.0

    ecg_mask = (freqs >= ecg_band[0]) & (freqs <= ecg_band[1])
    ecg_power = float(np.trapezoid(psd[ecg_mask], freqs[ecg_mask]))

    return ecg_power / total_power


def check_sqi(
    win_hr: np.ndarray,
    fs_hr: float,
    threshold: float = 0.80,
    ecg_band: Tuple[float, float] = (0.5, 45.0),
) -> bool:
    """
    Reject window if SQI < threshold.

    Parameters
    ----------
    win_hr : np.ndarray
        Raw (un-normalized) HR window.
    fs_hr : float
        HR sampling frequency (Hz).
    threshold : float
        τ_SQI (default 0.80).
    ecg_band : tuple
        ECG physiological frequency band (Hz).

    Returns
    -------
    bool : True if window passes.
    """
    return compute_sqi(win_hr, fs_hr, ecg_band) >= threshold


# ---------------------------------------------------------------------------
# Check 4 — RMSSD via Pan-Tompkins
# ---------------------------------------------------------------------------

def compute_rmssd(r_peaks: np.ndarray, fs_hr: float) -> float:
    """
    Compute RMSSD of successive RR interval differences (ms).

    RMSSD = sqrt( mean( (RR[k] − RR[k−1])^2 ) )  for k=1..K-1

    Parameters
    ----------
    r_peaks : np.ndarray
        R-peak indices (in samples).
    fs_hr : float
        Sampling frequency (Hz) to convert samples → seconds.

    Returns
    -------
    rmssd : float
        RMSSD in milliseconds. Returns inf if fewer than 3 peaks detected.
    """
    if len(r_peaks) < 3:
        return float("inf")

    rr_intervals_ms = np.diff(r_peaks) / fs_hr * 1000.0
    successive_diffs = np.diff(rr_intervals_ms)
    return float(np.sqrt(np.mean(successive_diffs ** 2)))


def check_rmssd(
    win_hr: np.ndarray,
    fs_hr: float,
    threshold_ms: float = 300.0,
) -> bool:
    """
    Reject window if RMSSD of successive RR differences exceeds threshold.

    Detects R-peaks via Pan-Tompkins; rejects if RMSSD > τ_RR.

    Parameters
    ----------
    win_hr : np.ndarray
        Raw (un-normalized) HR window.
    fs_hr : float
        HR sampling frequency (Hz).
    threshold_ms : float
        τ_RR in milliseconds (default 300 ms).

    Returns
    -------
    bool : True if window passes.
    """
    r_peaks = detect_r_peaks(win_hr, fs_hr)
    rmssd = compute_rmssd(r_peaks, fs_hr)
    return rmssd <= threshold_ms


# ---------------------------------------------------------------------------
# Combined quality gate
# ---------------------------------------------------------------------------

def is_valid_window(
    win_hr_norm: np.ndarray,
    win_hr_raw: np.ndarray,
    sigma_w: float,
    fs_hr: float,
    saturation_threshold: float = 5.0,
    saturation_max_fraction: float = 0.01,
    min_std: float = 0.05,
    max_std: float = 5.0,
    sqi_threshold: float = 0.80,
    ecg_band: Tuple[float, float] = (0.5, 45.0),
    rmssd_threshold_ms: float = 300.0,
) -> Tuple[bool, str]:
    """
    Run all four artifact rejection checks on a single window.

    Checks are ordered by computational cost (cheapest first) to fail fast.

    Parameters
    ----------
    win_hr_norm : np.ndarray
        Z-score normalized HR window.
    win_hr_raw : np.ndarray
        Raw (un-normalized) HR window in original mV units.
    sigma_w : float
        Per-window σ computed during z-score normalization (mV).
    fs_hr : float
        HR sampling frequency (Hz).
    saturation_threshold : float
        τ_sat for check 1.
    saturation_max_fraction : float
        Max fraction of samples above τ_sat before rejecting (default 1 %).
    min_std, max_std : float
        σ bounds (mV) for check 2.
    sqi_threshold : float
        τ_SQI for check 3.
    ecg_band : tuple
        ECG band for SQI computation.
    rmssd_threshold_ms : float
        τ_RR (ms) for check 4.

    Returns
    -------
    (is_valid, rejection_reason) : (bool, str)
        is_valid = True if all checks pass.
        rejection_reason = "" if valid, else one of:
            "saturation", "variance", "sqi", "rmssd"
    """
    # Check 1 — cheapest (pure array operation)
    if not check_saturation(win_hr_norm, saturation_threshold, saturation_max_fraction):
        return False, "saturation"

    # Check 2 — O(1)
    if not check_variance(sigma_w, min_std, max_std):
        return False, "variance"

    # Check 3 — Welch PSD (moderate cost)
    if not check_sqi(win_hr_raw, fs_hr, sqi_threshold, ecg_band):
        return False, "sqi"

    # Check 4 — Pan-Tompkins (most expensive)
    if not check_rmssd(win_hr_raw, fs_hr, rmssd_threshold_ms):
        return False, "rmssd"

    return True, ""
