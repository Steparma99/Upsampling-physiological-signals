"""
augmentations.py — ECG-specific augmentations for SimCLR contrastive pre-training.

Six primitive augmentations (Table 3 of the architectural PDF):

  add_gaussian_noise   — additive white noise, SNR drawn from U(snr_min, snr_max) dB
  amplitude_scaling    — global amplitude factor drawn from U(scale_min, scale_max)
  baseline_wander      — slow sinusoidal drift at 0.05–0.8 Hz
  powerline_noise      — 50 Hz additive sinusoid (configurable frequency)
  segment_masking      — zero a random contiguous segment ≤ max_frac of the window
  time_warping         — smooth local time deformation via cubic spline; re-detects
                         R-peaks and recomputes d_map (CLAUDE.md constraint #10)

ECGAugmentation class:
  Orchestrates the six primitives to produce two independent augmented views
  (view_a, view_b) from a single input signal, as required by the SimCLR/NT-Xent
  training objective (Phase A). Each view gets its own d_map, recomputed when
  time warping is applied.

All parameters are expected from configs/default.yaml → augmentations section.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.ndimage import map_coordinates

try:
    import neurokit2 as nk
except ImportError:
    nk = None

from .preprocessing import FS_LR, compute_r_distance_map, detect_r_peaks

logger = logging.getLogger(__name__)


def _lerp(start: float, end: float, t: float) -> float:
    """Linear interpolation: start at t=0, end at t=1."""
    return start + (end - start) * t


# =============================================================================
# Primitive augmentations
# =============================================================================

def add_gaussian_noise(
    signal: NDArray[np.floating],
    snr_db_min: float = 25.0,
    snr_db_max: float = 45.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float32]:
    """Add white Gaussian noise at a random SNR drawn from U(snr_db_min, snr_db_max).

    Noise standard deviation is derived from signal power so that the requested
    SNR is met regardless of the signal's absolute amplitude:

        sigma_noise = std(signal) / 10^(SNR_dB / 20)

    Args:
        signal: 1-D ECG signal, any amplitude.
        snr_db_min: Lower bound of SNR range in dB (PDF: 25 dB).
        snr_db_max: Upper bound of SNR range in dB (PDF: 45 dB).
        rng: Optional seeded numpy Generator for reproducibility.

    Returns:
        Noisy signal as float32.
    """
    if rng is None:
        rng = np.random.default_rng()
    sig = np.asarray(signal, dtype=np.float64)
    snr_db = rng.uniform(snr_db_min, snr_db_max)
    sigma_noise = float(np.std(sig)) / (10.0 ** (snr_db / 20.0))
    noise = rng.normal(0.0, sigma_noise, size=sig.shape)
    return (sig + noise).astype(np.float32)


def amplitude_scaling(
    signal: NDArray[np.floating],
    scale_min: float = 0.7,
    scale_max: float = 1.3,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float32]:
    """Scale signal amplitude by a factor drawn from U(scale_min, scale_max).

    Args:
        signal: 1-D ECG signal.
        scale_min: Lower bound of scaling factor (PDF: 0.7).
        scale_max: Upper bound of scaling factor (PDF: 1.3).
        rng: Optional seeded numpy Generator.

    Returns:
        Scaled signal as float32.
    """
    if rng is None:
        rng = np.random.default_rng()
    alpha = rng.uniform(scale_min, scale_max)
    return (np.asarray(signal, dtype=np.float64) * alpha).astype(np.float32)


def baseline_wander(
    signal: NDArray[np.floating],
    fs: float = FS_LR,
    freq_min: float = 0.05,
    freq_max: float = 0.8,
    amplitude_frac: float = 0.10,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float32]:
    """Add sinusoidal baseline wander at a random frequency in [freq_min, freq_max] Hz.

    Amplitude is amplitude_frac × std(signal), matching realistic ECG drift.
    Random phase is drawn from U(0, 2π) so the drift direction varies between views.

    Args:
        signal: 1-D ECG signal.
        fs: Sampling rate in Hz.
        freq_min: Minimum wander frequency (PDF: 0.05 Hz).
        freq_max: Maximum wander frequency (PDF: 0.8 Hz).
        amplitude_frac: Wander amplitude relative to signal std (default 10%).
        rng: Optional seeded numpy Generator.

    Returns:
        Signal with added wander, float32.
    """
    if rng is None:
        rng = np.random.default_rng()
    sig = np.asarray(signal, dtype=np.float64)
    N = len(sig)
    t = np.arange(N) / fs
    freq = rng.uniform(freq_min, freq_max)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    amplitude = amplitude_frac * float(np.std(sig))
    wander = amplitude * np.sin(2.0 * np.pi * freq * t + phase)
    return (sig + wander).astype(np.float32)


def powerline_noise(
    signal: NDArray[np.floating],
    fs: float = FS_LR,
    freq: float = 50.0,
    amplitude_frac: float = 0.05,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float32]:
    """Add powerline interference as a sinusoid at `freq` Hz.

    Amplitude is amplitude_frac × std(signal). Random phase is drawn to avoid
    systematic phase alignment between the two SimCLR views.

    Args:
        signal: 1-D ECG signal.
        fs: Sampling rate in Hz.
        freq: Interference frequency in Hz (PDF: 50 Hz).
        amplitude_frac: Interference amplitude relative to signal std (PDF: ±5%).
        rng: Optional seeded numpy Generator.

    Returns:
        Corrupted signal as float32.
    """
    if rng is None:
        rng = np.random.default_rng()
    sig = np.asarray(signal, dtype=np.float64)

    if nk is not None:
        try:
            distorted = nk.signal_distort(
                sig,
                sampling_rate=int(fs),
                noise_amplitude=0,
                powerline_amplitude=float(amplitude_frac),
                powerline_frequency=float(freq),
                random_state=rng,
                silent=True,
            )
            return np.asarray(distorted, dtype=np.float32)
        except Exception as exc:
            logger.warning("nk.signal_distort failed (%s) - falling back to sinusoidal powerline noise.", exc)

    N = len(sig)
    t = np.arange(N) / fs
    phase = rng.uniform(0.0, 2.0 * np.pi)
    amplitude = amplitude_frac * float(np.std(sig))
    interference = amplitude * np.sin(2.0 * np.pi * freq * t + phase)
    return (sig + interference).astype(np.float32)


def segment_masking(
    signal: NDArray[np.floating],
    max_mask_frac: float = 0.10,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float32]:
    """Zero a random contiguous segment of length ≤ max_mask_frac × N.

    The segment start is drawn uniformly so the masked region can fall anywhere
    in the window. This forces the encoder to use context outside the masked
    region, promoting global ECG representations.

    Args:
        signal: 1-D ECG signal, shape (N,).
        max_mask_frac: Maximum masked fraction of the window (PDF: ≤ 10%).
        rng: Optional seeded numpy Generator.

    Returns:
        Signal with zeroed segment, float32.
    """
    if rng is None:
        rng = np.random.default_rng()
    sig = np.asarray(signal, dtype=np.float32).copy()
    N = len(sig)
    max_len = max(1, int(max_mask_frac * N))
    mask_len = rng.integers(1, max_len + 1)
    start = rng.integers(0, N - mask_len + 1)
    sig[start : start + mask_len] = 0.0
    return sig


def time_warping(
    signal: NDArray[np.floating],
    r_peaks: NDArray[np.intp],
    fs: float = FS_LR,
    warp_frac: float = 0.05,
    n_knots: int = 4,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.intp]]:
    """Locally deform the time axis via a smooth random cubic spline.

    A displacement field d(t) is constructed by fitting a clamped CubicSpline
    through n_knots randomly displaced interior control points (boundary
    derivatives are clamped to zero for signal-boundary stability). The warped
    signal is obtained by resampling the original at positions t + d(t) using
    cubic interpolation (scipy.ndimage.map_coordinates, order=3).

    R-peaks are re-detected on the warped signal to satisfy CLAUDE.md
    constraint #10: "after time warping augmentation, d_map must be recomputed."
    The caller (ECGAugmentation._augment_single) uses the returned r_peaks to
    recompute d_map.

    Args:
        signal: 1-D ECG signal, shape (N,).
        r_peaks: R-peak indices before warping.
        fs: Sampling rate in Hz.
        warp_frac: Maximum displacement as fraction of window length (PDF: ±5%).
        n_knots: Number of random interior control points for the displacement spline.
        rng: Optional seeded numpy Generator.

    Returns:
        Tuple (warped_signal, new_r_peaks) where both are in the warped domain.
    """
    if rng is None:
        rng = np.random.default_rng()
    sig = np.asarray(signal, dtype=np.float64)
    N = len(sig)
    max_disp = warp_frac * N

    # Smooth displacement field: clamped spline through random interior knots.
    # Endpoint values are fixed at 0 so the warp does not shift the signal boundaries.
    knot_x = np.linspace(0, N - 1, n_knots + 2)
    knot_y = np.zeros(n_knots + 2)
    knot_y[1:-1] = rng.uniform(-max_disp, max_disp, size=n_knots)
    cs = CubicSpline(knot_x, knot_y, bc_type="clamped")

    t_orig = np.arange(N, dtype=np.float64)
    # Source coordinates: where in the original signal each output sample comes from
    source_coords = np.clip(t_orig + cs(t_orig), 0.0, N - 1.0)

    warped = map_coordinates(sig, [source_coords], order=3, mode="nearest").astype(np.float32)

    # Re-detect R-peaks on warped signal (CLAUDE.md constraint #10)
    new_r_peaks = detect_r_peaks(warped, fs)
    return warped, new_r_peaks


# =============================================================================
# Orchestrator — two-view generator for SimCLR
# =============================================================================

class ECGAugmentation:
    """Generates two independently augmented views of an ECG window for SimCLR.

    Each call produces (view_a, view_b, d_map_a, d_map_b, r_peaks_a, r_peaks_b)
    where the two views are produced by applying independent random draws of the
    six primitives to the same input signal.

    Curriculum augmentation
    -----------------------
    Pass ``ramp_epochs > 0`` to enable progressive augmentation. Compute the
    strength scalar each epoch with ``ECGAugmentation.compute_epoch_strength()``
    and pass it to ``__call__``.  strength=1.0 (default) gives full-intensity
    augmentation from epoch 0 (curriculum disabled).

    Example::

        aug = ECGAugmentation(cfg.augmentations, ramp_epochs=cfg.augmentations.curriculum_ramp_epochs)
        for epoch in range(cfg.phase_a.epochs):
            s = ECGAugmentation.compute_epoch_strength(epoch, aug.ramp_epochs)
            view_a, view_b, d_a, d_b, rp_a, rp_b = aug(signal, r_peaks, strength=s)

    Parameters are loaded from the ``augmentations`` section of configs/default.yaml.
    """

    def __init__(
        self,
        params: dict[str, Any],
        fs: float = FS_LR,
        seed: int | None = None,
        ramp_epochs: int = 0,
    ) -> None:
        """
        Args:
            params: Augmentation config dict (from OmegaConf or plain dict).
            fs: Sampling rate of the input LR signal.
            seed: Optional integer seed for the internal numpy Generator.
            ramp_epochs: Number of epochs over which to ramp augmentation strength
                from 0 to 1 (cosine schedule). 0 disables curriculum (strength=1 always).
        """
        self.params = params
        self.fs = fs
        self.rng = np.random.default_rng(seed)
        self.ramp_epochs = ramp_epochs

    # ------------------------------------------------------------------
    # Curriculum schedule
    # ------------------------------------------------------------------

    @staticmethod
    def compute_epoch_strength(epoch: int, ramp_epochs: int) -> float:
        """Cosine curriculum schedule: 0 at epoch 0, 1.0 at epoch ≥ ramp_epochs.

        strength(t) = 0.5 × (1 − cos(π × min(t, ramp) / ramp))

        The cosine shape gives zero derivative at both ends — no hard corner at
        t=0 (unlike a linear ramp) and a smooth plateau once full strength is
        reached. Returns 1.0 immediately when ramp_epochs ≤ 0.
        """
        if ramp_epochs <= 0:
            return 1.0
        t = min(epoch, ramp_epochs) / ramp_epochs
        return 0.5 * (1.0 - math.cos(math.pi * t))

    # ------------------------------------------------------------------
    # Public call
    # ------------------------------------------------------------------

    def __call__(
        self,
        signal: NDArray[np.floating],
        r_peaks: NDArray[np.intp],
        strength: float = 1.0,
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.intp],
        NDArray[np.intp],
    ]:
        """Produce two independently augmented views of the same input window.

        The two views share the same original signal but receive independent random
        augmentation draws — this is the fundamental SimCLR requirement.

        Args:
            signal: LR ECG window, shape (L_LR,) = (1000,).
            r_peaks: R-peak indices in the LR sample domain.
            strength: Curriculum strength ∈ [0, 1]. Use compute_epoch_strength()
                to derive from epoch. Default 1.0 = full-intensity augmentation.

        Returns:
            (view_a, view_b, d_map_a, d_map_b, r_peaks_a, r_peaks_b).
            All signals have shape (L_LR,), both d_maps have shape (L_LR,).
        """
        strength = float(np.clip(strength, 0.0, 1.0))
        sig = np.asarray(signal, dtype=np.float32)
        view_a, r_peaks_a, d_map_a = self._augment_single(sig.copy(), r_peaks.copy(), strength)
        view_b, r_peaks_b, d_map_b = self._augment_single(sig.copy(), r_peaks.copy(), strength)
        return view_a, view_b, d_map_a, d_map_b, r_peaks_a, r_peaks_b

    # ------------------------------------------------------------------
    # Internal single-view augmentation
    # ------------------------------------------------------------------

    def _augment_single(
        self,
        signal: NDArray[np.float32],
        r_peaks: NDArray[np.intp],
        strength: float,
    ) -> tuple[NDArray[np.float32], NDArray[np.intp], NDArray[np.float32]]:
        """Apply one random draw of augmentations to produce a single view.

        Augmentation order is fixed: amplitude/frequency corruptions first,
        time warping last so it acts on the fully-corrupted signal.
        Parameter ranges are interpolated between a mild baseline (strength=0)
        and the full config values (strength=1) using _lerp.

        Returns:
            (augmented_signal, r_peaks, d_map)
        """
        p = self.params
        sig = signal.copy()
        peaks = r_peaks.copy()

        cfg = p.get("gaussian_noise", {})
        if self.rng.random() < float(cfg.get("p", 0.0)):
            snr_max = float(cfg.get("snr_db_max", 45.0))
            snr_min = _lerp(snr_max, float(cfg.get("snr_db_min", 25.0)), strength)
            sig = add_gaussian_noise(
                sig,
                snr_db_min=snr_min,
                snr_db_max=snr_max,
                rng=self.rng,
            )

        cfg = p.get("amplitude_scaling", {})
        if self.rng.random() < float(cfg.get("p", 0.0)):
            full_half = (float(cfg.get("scale_max", 1.3)) - float(cfg.get("scale_min", 0.7))) / 2.0
            half = _lerp(0.02, full_half, strength)
            sig = amplitude_scaling(
                sig,
                scale_min=1.0 - half,
                scale_max=1.0 + half,
                rng=self.rng,
            )

        cfg = p.get("baseline_wander", {})
        if self.rng.random() < float(cfg.get("p", 0.0)):
            sig = baseline_wander(
                sig,
                fs=self.fs,
                freq_min=float(cfg.get("freq_min", 0.05)),
                freq_max=float(cfg.get("freq_max", 0.8)),
                amplitude_frac=_lerp(0.01, float(cfg.get("amplitude_frac", 0.10)), strength),
                rng=self.rng,
            )

        cfg = p.get("powerline_noise", {})
        if self.rng.random() < float(cfg.get("p", 0.0)):
            sig = powerline_noise(
                sig,
                fs=self.fs,
                freq=float(cfg.get("freq", 50.0)),
                amplitude_frac=_lerp(0.005, float(cfg.get("amplitude_frac", 0.05)), strength),
                rng=self.rng,
            )

        cfg = p.get("segment_masking", {})
        if self.rng.random() < float(cfg.get("p", 0.0)):
            sig = segment_masking(
                sig,
                max_mask_frac=_lerp(0.01, float(cfg.get("max_mask_frac", 0.10)), strength),
                rng=self.rng,
            )

        # Time warping runs last — acts on the fully-corrupted signal.
        # R-peaks are re-detected inside time_warping; d_map is recomputed below.
        cfg = p.get("time_warping", {})
        if self.rng.random() < float(cfg.get("p", 0.0)):
            sig, peaks = time_warping(
                sig,
                peaks,
                fs=self.fs,
                warp_frac=_lerp(0.005, float(cfg.get("warp_frac", 0.05)), strength),
                n_knots=int(cfg.get("n_knots", 4)),
                rng=self.rng,
            )

        # Recompute d_map from current peaks.
        # Required after time warping (CLAUDE.md constraint #10);
        # computed unconditionally since it is cheap (~1 ms) and keeps this
        # method stateless regardless of which augmentations fired.
        d_map = compute_r_distance_map(peaks, length=len(sig))

        return sig, peaks, d_map
