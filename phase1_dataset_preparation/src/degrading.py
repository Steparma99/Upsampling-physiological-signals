"""
Degrading pipeline: D = Q ∘ N ∘ S ∘ F

Converts x_HR (fs_hr Hz) → x_LR (fs_lr Hz) through four sequential stages:
  F  — Kaiser FIR anti-aliasing filter (linear phase, zero-phase via filtfilt)
  S  — Decimation by integer factor R = fs_hr / fs_lr
  N  — Composite noise (thermal + powerline + baseline wander)
  Q  — 12-bit ADC quantization

Reference: project specification, Section 2.
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Stage F — Anti-Aliasing Filter
# ---------------------------------------------------------------------------

def design_antialiasing_filter(
    fs_hr: float,
    fs_lr: float,
    stopband_attenuation_db: float = 80.0,
    transition_width_normalized: float = 0.02,
) -> np.ndarray:
    """
    Design a Kaiser-windowed FIR anti-aliasing filter (Type I, linear phase).

    The filter attenuates all energy above f_N^LR = fs_lr / 2 = 50 Hz.

    Parameters
    ----------
    fs_hr : float
        High-resolution sampling frequency (Hz).
    fs_lr : float
        Low-resolution sampling frequency (Hz).
    stopband_attenuation_db : float
        Minimum stopband attenuation A_s (dB). Default 80 dB.
    transition_width_normalized : float
        Normalized transition width Δf / (fs_hr/2). Default 0.02.

    Returns
    -------
    h : np.ndarray
        FIR filter coefficients (length M+1, M even).
    """
    nyquist_lr = fs_lr / 2.0                    # 50 Hz
    fc_normalized = nyquist_lr / (fs_hr / 2.0)  # = 0.1 (normalized to Nyquist)

    # Kaiser β from stopband attenuation (Harris 1978 / Kaiser formula)
    A_s = stopband_attenuation_db
    if A_s >= 50.0:
        beta = 0.1102 * (A_s - 8.7)
    elif A_s >= 21.0:
        beta = 0.5842 * (A_s - 21.0) ** 0.4 + 0.07886 * (A_s - 21.0)
    else:
        beta = 0.0

    # Filter order estimate: M ≥ (A_s − 8) / (2.285 × Δω)
    # Δω = transition_width_normalized × π  (in radians per sample)
    delta_omega = transition_width_normalized * np.pi
    M = int(np.ceil((A_s - 8.0) / (2.285 * delta_omega)))
    if M % 2 != 0:  # enforce even order → Type I FIR (symmetric, odd length)
        M += 1

    h = sp_signal.firwin(M + 1, fc_normalized, window=("kaiser", beta))
    return h


def apply_antialiasing_filter(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Apply the FIR anti-aliasing filter with zero-phase (via filtfilt).

    filtfilt is appropriate for offline dataset construction; it introduces
    no delay and avoids edge artifacts compared to causal lfilter + alignment.

    Parameters
    ----------
    x : np.ndarray
        Input signal (1-D, length L).
    h : np.ndarray
        FIR filter coefficients from `design_antialiasing_filter`.

    Returns
    -------
    x_filtered : np.ndarray
        Filtered signal, same length as x.
    """
    # filtfilt requires padlen <= len(x) - 1
    padlen = min(3 * (len(h) - 1), len(x) - 1)
    return sp_signal.filtfilt(h, 1.0, x, padlen=padlen)


# ---------------------------------------------------------------------------
# Stage S — Decimation
# ---------------------------------------------------------------------------

def decimate_signal(x: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample by integer factor (simple stride after anti-aliasing).

    Parameters
    ----------
    x : np.ndarray
        Pre-filtered signal.
    factor : int
        Decimation factor R.

    Returns
    -------
    x_down : np.ndarray
        Decimated signal of length floor(len(x) / factor).
    """
    return x[::factor].copy()


# ---------------------------------------------------------------------------
# Stage N — Composite Noise
# ---------------------------------------------------------------------------

def add_composite_noise(
    x: np.ndarray,
    fs_lr: float,
    snr_db_range: Tuple[float, float] = (25.0, 45.0),
    powerline_freq: float = 50.0,
    powerline_amp_range: Tuple[float, float] = (0.0, 0.05),
    bw_freq_range: Tuple[float, float] = (0.05, 0.8),
    bw_amp_range: Tuple[float, float] = (0.0, 0.1),
    n_bw_components: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Add composite noise to a decimated LR signal.

    Three noise components (Section 2.3):
      1. Thermal noise  — white Gaussian, SNR_dB ~ U(snr_db_range)
      2. Powerline      — sinusoid at powerline_freq Hz, random phase
      3. Baseline wander— sum of n_bw_components low-frequency sinusoids

    Parameters
    ----------
    x : np.ndarray
        Decimated signal at fs_lr.
    fs_lr : float
        Low-resolution sampling frequency (Hz).
    snr_db_range : tuple
        (min, max) SNR in dB for thermal noise.
    powerline_freq : float
        Powerline frequency (Hz). 50 Hz (EU) or 60 Hz (US).
    powerline_amp_range : tuple
        (min, max) powerline amplitude as fraction of σ_x.
    bw_freq_range : tuple
        (min, max) baseline wander component frequency (Hz).
    bw_amp_range : tuple
        (min, max) baseline wander amplitude as fraction of σ_x.
    n_bw_components : int
        Number of baseline wander sinusoids.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    x_noisy : np.ndarray
        Signal with composite noise added.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(x)
    t = np.arange(N) / fs_lr
    sigma_x = float(np.std(x))

    if sigma_x < 1e-10:
        # Flat signal — skip noise to avoid division by zero
        return x.copy()

    # 1. Thermal noise (white Gaussian)
    snr_db = rng.uniform(*snr_db_range)
    sigma_th = sigma_x * 10.0 ** (-snr_db / 20.0)
    eta_th = rng.normal(0.0, sigma_th, N)

    # 2. Powerline interference
    A_pl = rng.uniform(*powerline_amp_range) * sigma_x
    phi_pl = rng.uniform(0.0, 2.0 * np.pi)
    eta_pl = A_pl * np.sin(2.0 * np.pi * powerline_freq * t + phi_pl)

    # 3. Baseline wander
    eta_bw = np.zeros(N, dtype=np.float64)
    for _ in range(n_bw_components):
        f_k = rng.uniform(*bw_freq_range)
        A_k = rng.uniform(*bw_amp_range) * sigma_x
        phi_k = rng.uniform(0.0, 2.0 * np.pi)
        eta_bw += A_k * np.sin(2.0 * np.pi * f_k * t + phi_k)

    return (x + eta_th + eta_pl + eta_bw).astype(x.dtype)


# ---------------------------------------------------------------------------
# Stage Q — Quantization
# ---------------------------------------------------------------------------

def quantize_signal(
    x: np.ndarray,
    bits: int = 12,
    v_min: float = -5.0,
    v_max: float = 5.0,
) -> np.ndarray:
    """
    Simulate ADC quantization at given bit depth (Section 2.4).

    Quantization step: Δ = (V_max − V_min) / (2^b − 1)
    Quantized value:   x_Q[m] = Δ · floor(x[m] / Δ + 0.5)

    For b=12, ±5 mV: Δ ≈ 2.44 μV.
    The quantization error e_Q ∈ [−Δ/2, Δ/2] is uniform with power Δ²/12.

    Parameters
    ----------
    x : np.ndarray
        Input signal (mV).
    bits : int
        ADC bit depth.
    v_min, v_max : float
        ADC input voltage range (mV).

    Returns
    -------
    x_q : np.ndarray
        Quantized signal (same shape as x).
    """
    delta = (v_max - v_min) / (2 ** bits - 1)
    return (delta * np.floor(x / delta + 0.5)).astype(x.dtype)


# ---------------------------------------------------------------------------
# Full degrading chain
# ---------------------------------------------------------------------------

def degrade_signal(
    x_hr: np.ndarray,
    h: np.ndarray,
    upsample_factor: int = 5,
    fs_lr: float = 100.0,
    noise_kwargs: Optional[dict] = None,
    quantize: bool = True,
    quant_kwargs: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Full degrading chain: D = Q ∘ N ∘ S ∘ F.

    Parameters
    ----------
    x_hr : np.ndarray
        Input HR signal at fs_hr = upsample_factor × fs_lr.
    h : np.ndarray
        Pre-designed anti-aliasing FIR filter coefficients.
    upsample_factor : int
        R = fs_hr / fs_lr.
    fs_lr : float
        Low-resolution target frequency (Hz).
    noise_kwargs : dict, optional
        Extra kwargs forwarded to `add_composite_noise`.
    quantize : bool
        Whether to apply Stage Q.
    quant_kwargs : dict, optional
        Extra kwargs forwarded to `quantize_signal`.
    rng : np.random.Generator, optional
        Random generator for noise reproducibility.

    Returns
    -------
    x_lr : np.ndarray
        Degraded LR signal.
    """
    if rng is None:
        rng = np.random.default_rng()
    if noise_kwargs is None:
        noise_kwargs = {}
    if quant_kwargs is None:
        quant_kwargs = {}

    # Stage F: anti-aliasing filter (zero-phase FIR)
    x_filtered = apply_antialiasing_filter(x_hr, h)

    # Stage S: decimation by R
    x_decimated = decimate_signal(x_filtered, upsample_factor)

    # Stage N: composite noise
    x_noisy = add_composite_noise(x_decimated, fs_lr, rng=rng, **noise_kwargs)

    # Stage Q: quantization (optional)
    if quantize:
        x_lr = quantize_signal(x_noisy, **quant_kwargs)
    else:
        x_lr = x_noisy

    return x_lr
