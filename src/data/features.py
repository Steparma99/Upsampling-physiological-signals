"""
features.py — Feature vector extraction for ECG super-resolution conditioning.

Two conditioning vectors are produced here (Eq. 7–8, architectural PDF):

  f_signal ∈ R^12
    Computed entirely from x_LR — available at both training and inference.
    [mu_w, sigma_w, HR, RMSSD, delta_RR, Skew, Kurt, QRS_area, slope, ZCR, QTc, T_pol]

  f_meta ∈ R^7
    Patient demographics and diagnostic labels (PTB-XL and other annotated sets).
    [age/100, sex, NORM, MI, STTC, CD, HYP]

Both vectors are projected by independent MLPs and concatenated before injection
into the Transformer encoder (see CLAUDE.md — Conditioning Vectors).
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch
from scipy.stats import kurtosis as _scipy_kurtosis
from scipy.stats import skew as _scipy_skew

# np.trapezoid (NumPy >= 2.0) replaced np.trapz; support both.
try:
    from numpy import trapezoid as _np_trapz
except ImportError:
    from numpy import trapz as _np_trapz  # type: ignore[no-redef]

try:
    import neurokit2 as nk
except ImportError:
    nk = None

from .preprocessing import FS_LR

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================
# Fixed order — any change breaks existing checkpoint weight files.

F_SIGNAL_DIM: int = 12
F_META_DIM: int = 7

# Fixed order used for the multi-hot encoding of f_meta[2:7]
DIAGNOSTIC_CLASSES: tuple[str, ...] = ("NORM", "MI", "STTC", "CD", "HYP")


# =============================================================================
# Private helper — PQRST delineation
# =============================================================================

def _clean_ecg_signal(
    signal: NDArray[np.floating],
    fs: float,
) -> NDArray[np.float64]:
    """Clean ECG once with NeuroKit2 when available, else return raw signal."""
    sig = np.asarray(signal, dtype=np.float64)
    if nk is None:
        return sig
    try:
        return np.asarray(nk.ecg_clean(sig, sampling_rate=int(fs)), dtype=np.float64)
    except Exception as exc:
        logger.warning("nk.ecg_clean failed (%s) - using raw ECG for morphology features.", exc)
        return sig


def _delineate_ecg(
    cleaned_signal: NDArray[np.floating],
    r_peaks: NDArray[np.intp],
    fs: float,
) -> dict | None:
    """Delineate P/Q/R/S/T landmarks with nk.ecg_clean + nk.ecg_delineate (DWT).

    Called once inside build_f_signal; the resulting waves dict is shared
    between compute_qtc and compute_t_polarity to avoid redundant computation.

    Args:
        cleaned_signal: Cleaned or raw LR window.
        r_peaks: R-peak indices in the LR sample domain.
        fs: Sampling rate in Hz.

    Returns:
        waves dict with keys such as "ECG_T_Peaks", "ECG_T_Offsets",
        "ECG_R_Onsets" (sample-index arrays, may contain NaN for failed beats).
        Returns None if delineation is not possible or raises an exception.
    """
    if nk is None or len(r_peaks) < 2:
        return None
    try:
        _, waves = nk.ecg_delineate(
            np.asarray(cleaned_signal, dtype=np.float64),
            r_peaks,
            sampling_rate=int(fs),
            method="dwt",
        )
        return waves
    except Exception as exc:
        logger.warning("nk.ecg_delineate failed (%s) — QTc/T_polarity will use fallbacks.", exc)
        return None


# =============================================================================
# Individual feature functions
# =============================================================================

def compute_hr(
    r_peaks: NDArray[np.intp],
    fs: float = FS_LR,
) -> float:
    """Heart rate in BPM computed from mean R-R interval.

    Returns 0.0 when fewer than 2 R-peaks are available (sentinel value;
    the downstream normalisation layer learns to ignore it).

    Args:
        r_peaks: R-peak indices in the LR sample domain, shape (N,).
        fs: Sampling rate in Hz.
    """
    if len(r_peaks) < 2:
        return 0.0
    rr_sec = np.diff(r_peaks) / fs
    return float(60.0 / np.mean(rr_sec))


def compute_rmssd(
    r_peaks: NDArray[np.intp],
    fs: float = FS_LR,
) -> float:
    """RMSSD in milliseconds — time-domain HRV metric.

    Uses nk.hrv_time for the clinically validated implementation.
    Falls back to the manual formula if NeuroKit2 raises an exception.
    Requires ≥ 3 R-peaks (2 RR intervals → 1 successive difference).

    Returns 0.0 when not computable.
    """
    if len(r_peaks) < 3:
        return 0.0
    if nk is None:
        rr_ms = np.diff(r_peaks) / fs * 1000.0
        return float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
    try:
        hrv = nk.hrv_time(r_peaks, sampling_rate=int(fs), show=False)
        val = float(hrv["HRV_RMSSD"].iloc[0])
        return val if np.isfinite(val) else 0.0
    except Exception:
        rr_ms = np.diff(r_peaks) / fs * 1000.0
        return float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))


def compute_delta_rr(
    r_peaks: NDArray[np.intp],
    fs: float = FS_LR,
) -> float:
    """Relative R-R variation: (max_RR − min_RR) / mean_RR.

    High values indicate arrhythmia or pronounced respiratory modulation.
    Returns 0.0 when fewer than 2 R-peaks are available.
    """
    if len(r_peaks) < 2:
        return 0.0
    rr = np.diff(r_peaks) / fs
    mean_rr = float(np.mean(rr))
    if mean_rr < 1e-8:
        return 0.0
    return float((np.max(rr) - np.min(rr)) / mean_rr)


def compute_mean_qrs_area(
    signal: NDArray[np.floating],
    r_peaks: NDArray[np.intp],
    fs: float = FS_LR,
) -> float:
    """Mean area under the QRS complex, averaged across all beats.

    Area per beat = trapezoidal integral of |signal| over a ±50 ms window
    centred on the R-peak. Captures QRS amplitude and energy.

    Returns 0.0 if no R-peaks.
    """
    if len(r_peaks) == 0:
        return 0.0
    half_qrs = int(0.05 * fs)
    areas: list[float] = []
    for rp in r_peaks:
        lo = max(0, rp - half_qrs)
        hi = min(len(signal), rp + half_qrs + 1)
        areas.append(float(_np_trapz(np.abs(signal[lo:hi]), dx=1.0 / fs)))
    return float(np.mean(areas))


def compute_spectral_slope(
    signal: NDArray[np.floating],
    fs: float = FS_LR,
    f_low: float = 0.5,
    f_high: float = 40.0,
) -> float:
    """Spectral slope via log-log linear regression on Welch PSD (0.5–40 Hz).

    Normal ECG signals have a negative slope (more power at low frequencies).
    Pathologies like AF significantly alter the spectral shape.

    Returns 0.0 if the frequency band is empty or the PSD has non-positive values.
    """
    nperseg = min(256, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    mask = (freqs >= f_low) & (freqs <= f_high)
    freqs_b, psd_b = freqs[mask], psd[mask]
    if len(freqs_b) < 2 or np.any(psd_b <= 0):
        return 0.0
    coeffs = np.polyfit(np.log10(freqs_b), np.log10(psd_b), deg=1)
    return float(coeffs[0])


def compute_zcr(signal: NDArray[np.floating]) -> float:
    """Zero crossing rate — sign changes per sample.

    High ZCR correlates with high-frequency content (noise, AF flutter waves).
    Returns 0.0 when the signal has fewer than 2 samples.
    """
    n = len(signal)
    if n < 2:
        return 0.0
    crossings = int(np.sum(np.abs(np.diff(np.sign(signal))) > 0))
    return float(crossings / (n - 1))


def compute_qtc(
    signal: NDArray[np.floating],
    r_peaks: NDArray[np.intp],
    fs: float = FS_LR,
    waves: dict | None = None,
) -> float:
    """QTc via Bazett formula: QTc = QT / sqrt(mean_RR).

    Returns float('nan') when the interval cannot be estimated:
      - fewer than 2 R-peaks (no RR interval computable)
      - mean RR < 200 ms (>300 bpm, physiologically impossible)

    QT estimation priority:
      1. T-offset from delineation (actual T-wave endpoint)
         Uses ECG_T_Offsets − ECG_R_Onsets; falls back to R-peaks as QRS start
         proxy when R-onsets are not available. Only beats where both landmarks
         are finite and plausible (0.15 s < QT < 0.70 s) contribute.
      2. Empirical approximation: QT ≈ 0.40 × mean_RR.

    Args:
        signal: Non-normalized LR window (unused directly — kept for API symmetry).
        r_peaks: R-peak indices, shape (N,).
        fs: Sampling rate in Hz.
        waves: Output dict from _delineate_ecg, or None.

    Returns:
        QTc in seconds, or float('nan') if not estimable.
    """
    if len(r_peaks) < 2:
        return float("nan")

    rr_sec = np.diff(r_peaks) / fs
    mean_rr = float(np.mean(rr_sec))
    if mean_rr < 0.2:
        return float("nan")

    # --- Primary: delineation-based QT ---
    if waves is not None:
        try:
            t_offsets = np.asarray(waves.get("ECG_T_Offsets", []), dtype=float)
            r_onsets = np.asarray(waves.get("ECG_R_Onsets", []), dtype=float)
            n = min(len(t_offsets), len(r_peaks))
            t_off = t_offsets[:n]
            qrs_start = (
                r_onsets[:n] if len(r_onsets) >= n else r_peaks[:n].astype(float)
            )
            valid = np.isfinite(t_off) & np.isfinite(qrs_start) & (t_off > qrs_start)
            if np.sum(valid) > 0:
                qt_sec = float(np.mean(t_off[valid] - qrs_start[valid])) / fs
                if 0.15 < qt_sec < 0.70:
                    return float(qt_sec / np.sqrt(mean_rr))
        except Exception:
            pass

    # --- Fallback: empirical QT ≈ 0.40 × RR ---
    qt_sec = 0.40 * mean_rr
    return float(qt_sec / np.sqrt(mean_rr))


def compute_t_polarity(
    signal: NDArray[np.floating],
    r_peaks: NDArray[np.intp],
    fs: float = FS_LR,
    waves: dict | None = None,
) -> float:
    """T-wave polarity: +1.0 (upright) or −1.0 (inverted).

    Inverted T-waves can indicate ischaemia, hypertrophy, or bundle-branch block.

    Detection priority:
      1. T-peak amplitudes from delineation (precise T-wave location)
      2. Mean amplitude over fixed window R+200 ms…R+400 ms (anatomical approximation)

    Returns +1.0 if no R-peaks are available (upright is the most common case).
    """
    if len(r_peaks) == 0:
        return 1.0

    # --- Primary: delineation-based T-peaks ---
    if waves is not None:
        try:
            t_peaks = np.asarray(waves.get("ECG_T_Peaks", []), dtype=float)
            valid_idx = np.where(np.isfinite(t_peaks))[0]
            if len(valid_idx) > 0:
                t_amps = [
                    float(signal[int(t_peaks[i])])
                    for i in valid_idx
                    if 0 <= int(t_peaks[i]) < len(signal)
                ]
                if t_amps:
                    return 1.0 if float(np.mean(t_amps)) >= 0.0 else -1.0
        except Exception:
            pass

    # --- Fallback: fixed window R+200ms..R+400ms ---
    t_start = int(0.20 * fs)
    t_end = int(0.40 * fs)
    t_amps = []
    for rp in r_peaks:
        if rp + t_end < len(signal):
            t_amps.append(float(np.mean(signal[rp + t_start : rp + t_end])))
    if not t_amps:
        return 1.0
    return 1.0 if float(np.mean(t_amps)) >= 0.0 else -1.0


# =============================================================================
# Assembly functions
# =============================================================================

def build_f_signal(
    signal_lr: NDArray[np.floating],
    mu_w: float,
    sigma_w: float,
    r_peaks: NDArray[np.intp],
    fs: float = FS_LR,
) -> NDArray[np.float32]:
    """Assemble the f_signal conditioning vector ∈ R^12 (Eq. 7, Table 1 PDF).

    Fixed feature order (must not change — altering it breaks checkpoint
    compatibility because the projection MLP weights are tied to this layout):

      idx  category    feature         notes
      ---  ----------  --------------  -------------------------------------------
       0   global      mu_w            mean of LR window (pre-normalisation)
       1   global      sigma_w         std  of LR window (pre-normalisation)
       2   rhythm      HR              beats per minute
       3   rhythm      RMSSD           HRV time-domain, milliseconds
       4   rhythm      delta_RR        relative R-R variation
       5   morphology  Skewness        amplitude distribution (full 1000-sample window)
       6   morphology  Kurtosis        excess kurtosis, Fisher definition (normal=0)
       7   morphology  QRS_area        mean |QRS| trapezoidal integral across beats
       8   frequency   spec_slope      log-log slope in 0.5–40 Hz band
       9   morphology  ZCR             zero crossing rate
      10   morphology  QTc             Bazett-corrected QT interval, seconds
      11   morphology  T_polarity      T-wave polarity ∈ {−1, +1}

    NaN values (e.g. QTc when < 2 R-peaks) are converted to 0.0. The network
    learns to treat 0.0 as a "not available" sentinel via normalisation statistics.

    _delineate_ecg is called once and its result shared between compute_qtc and
    compute_t_polarity to avoid running the DWT delineation twice per window.

    Args:
        signal_lr: Non-normalized LR window, shape (L_LR,) = (1000,).
        mu_w: Mean of signal_lr (computed by zscore_normalize in preprocessing.py).
        sigma_w: Std  of signal_lr.
        r_peaks: R-peak indices in the LR sample domain.
        fs: Sampling rate in Hz.

    Returns:
        float32 array of shape (F_SIGNAL_DIM,) = (12,).
    """
    cleaned_signal = _clean_ecg_signal(signal_lr, fs)
    waves = _delineate_ecg(cleaned_signal, r_peaks, fs)

    f = np.empty(F_SIGNAL_DIM, dtype=np.float32)
    f[0]  = mu_w
    f[1]  = sigma_w
    f[2]  = compute_hr(r_peaks, fs)
    f[3]  = compute_rmssd(r_peaks, fs)
    f[4]  = compute_delta_rr(r_peaks, fs)
    f[5]  = float(_scipy_skew(signal_lr, bias=True))
    f[6]  = float(_scipy_kurtosis(signal_lr, fisher=True, bias=True))
    f[7]  = compute_mean_qrs_area(signal_lr, r_peaks, fs)
    f[8]  = compute_spectral_slope(signal_lr, fs)
    f[9]  = compute_zcr(signal_lr)
    f[10] = compute_qtc(cleaned_signal, r_peaks, fs, waves=waves)
    f[11] = compute_t_polarity(cleaned_signal, r_peaks, fs, waves=waves)

    # Sentinel conversion: NaN → 0.0, ±inf → large finite (safety clamp)
    np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return f


def build_f_meta(
    age: float | None,
    sex: float | None,
    diagnostic_labels: dict[str, bool] | None,
    p_drop: float = 0.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float32]:
    """Assemble the f_meta conditioning vector ∈ R^7 (Eq. 8 PDF).

    Feature layout:
      [0] age / 100   normalised patient age  (0 if missing)
      [1] sex         0=male, 1=female, 0.5 if unknown
      [2] NORM        ]
      [3] MI          ]  multi-hot diagnostic labels
      [4] STTC        ]  (a record can carry multiple simultaneous classes)
      [5] CD          ]  0 if the entire diagnostic_labels dict is None
      [6] HYP         ]

    Missing-value policy (from architectural PDF):
      - age missing   → 0
      - sex missing   → 0.5  (equidistant neutral value)
      - labels None   → all-zero subvector [0,0,0,0,0]

    Metadata dropout (training only):
      With probability p_drop the *entire* vector is zeroed jointly. This forces
      the network to function without metadata, preventing it from collapsing
      to metadata-only predictions. The dropout is all-or-nothing — individual
      features are never zeroed independently (CLAUDE.md constraint #5).
      Leave p_drop=0.0 at inference (default).

    Args:
        age: Patient age in years, or None.
        sex: 0.0 (male) / 1.0 (female), or None.
        diagnostic_labels: Dict keyed by DIAGNOSTIC_CLASSES strings, values bool.
                           None if no diagnostic information is available.
        p_drop: Probability of zeroing the full vector (use config value in training).
        rng: Optional seeded numpy Generator for reproducible dropout in unit tests.

    Returns:
        float32 array of shape (F_META_DIM,) = (7,).
    """
    f = np.zeros(F_META_DIM, dtype=np.float32)
    f[0] = float(age / 100.0) if age is not None else 0.0
    f[1] = float(sex) if sex is not None else 0.5

    if diagnostic_labels is not None:
        for i, cls_name in enumerate(DIAGNOSTIC_CLASSES):
            f[2 + i] = 1.0 if diagnostic_labels.get(cls_name, False) else 0.0

    # Joint metadata dropout — never partial (CLAUDE.md constraint #5)
    if p_drop > 0.0:
        draw = rng.random() if rng is not None else np.random.random()
        if draw < p_drop:
            f[:] = 0.0

    return f
