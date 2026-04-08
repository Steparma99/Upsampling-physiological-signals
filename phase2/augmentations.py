"""
ECG augmentations for contrastive pre-training.

Each augmentation is applied independently with probability p=0.5.
All operations are numpy-based (float32/float64) and return the same
dtype/shape as the input.  Conversion to tensors is done in the training
loop, not here.

Augmentations:
  1. gaussian_noise    — additive white Gaussian noise at random SNR
  2. amplitude_scaling — uniform amplitude scale factor
  3. baseline_wander   — sum of low-frequency sinusoids
  4. powerline_noise   — 50 Hz sinusoidal interference
  5. time_warping      — cubic-spline temporal distortion
  6. segment_masking   — zero-out a contiguous segment
"""

import numpy as np
from scipy.interpolate import interp1d


class ECGAugmentations:
    """
    Callable that applies a random subset of ECG augmentations to a 1-D signal.

    Each augmentation fires independently with probability ``p`` (default 0.5).
    The order is fixed; which augmentations are active is random.

    Parameters
    ----------
    p :    Per-augmentation application probability.
    seed : Optional integer seed for the internal RNG (np.random.default_rng).
    """

    def __init__(self, p: float = 0.5, seed: int | None = None) -> None:
        self.p = p
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Static augmentation primitives
    # ------------------------------------------------------------------

    @staticmethod
    def gaussian_noise(
        x: np.ndarray,
        rng: np.random.Generator,
        snr_db_range: tuple[float, float] = (25.0, 45.0),
    ) -> np.ndarray:
        """
        Add white Gaussian noise at a random SNR.

            SNR_dB  ~ Uniform(snr_db_range)
            sigma    = std(x) * 10^(−SNR_dB / 20)
            x'       = x + N(0, sigma²)
        """
        snr_db = rng.uniform(*snr_db_range)
        sigma = float(np.std(x)) * 10.0 ** (-snr_db / 20.0)
        noise = rng.normal(0.0, sigma, size=x.shape)
        return (x + noise).astype(x.dtype)

    @staticmethod
    def amplitude_scaling(
        x: np.ndarray,
        rng: np.random.Generator,
        alpha_range: tuple[float, float] = (0.7, 1.3),
    ) -> np.ndarray:
        """
        Multiply the signal by a random scalar.

            alpha ~ Uniform(alpha_range)
            x' = alpha * x
        """
        alpha = rng.uniform(*alpha_range)
        return (alpha * x).astype(x.dtype)

    @staticmethod
    def baseline_wander(
        x: np.ndarray,
        rng: np.random.Generator,
        fs: float = 100.0,
        n_components: int = 3,
        freq_range: tuple[float, float] = (0.05, 0.8),
        amp_factor: float = 0.1,
    ) -> np.ndarray:
        """
        Add a synthetic baseline wander as a sum of low-frequency sinusoids.

            For k in range(n_components):
              f_k  ~ Uniform(freq_range)
              A_k  ~ Uniform(0, amp_factor * std(x))
              φ_k  ~ Uniform(0, 2π)
              wander += A_k * sin(2π f_k t + φ_k)
            x' = x + wander
        """
        t = np.arange(len(x), dtype=np.float64) / fs
        wander = np.zeros_like(x, dtype=np.float64)
        std_x = float(np.std(x))
        for _ in range(n_components):
            f_k = rng.uniform(*freq_range)
            A_k = rng.uniform(0.0, amp_factor * std_x)
            phi_k = rng.uniform(0.0, 2.0 * np.pi)
            wander += A_k * np.sin(2.0 * np.pi * f_k * t + phi_k)
        return (x + wander).astype(x.dtype)

    @staticmethod
    def powerline_noise(
        x: np.ndarray,
        rng: np.random.Generator,
        fs: float = 100.0,
        freq: float = 50.0,
        amp_factor: float = 0.05,
    ) -> np.ndarray:
        """
        Add a 50 Hz sinusoidal powerline artefact.

            A   ~ Uniform(0, amp_factor * std(x))
            φ   ~ Uniform(0, 2π)
            x'  = x + A * sin(2π * freq * t + φ)

        Note: with fs_LR = 100 Hz the Nyquist limit is 50 Hz exactly.
        The 50 Hz component aliases to DC.  This augmentation is physically
        motivated only when this signal is used at fs_HR = 500 Hz; at
        fs_LR = 100 Hz it adds near-DC bias, which is still a valid
        robustness test.
        """
        t = np.arange(len(x), dtype=np.float64) / fs
        A = rng.uniform(0.0, amp_factor * float(np.std(x)))
        phi = rng.uniform(0.0, 2.0 * np.pi)
        noise = A * np.sin(2.0 * np.pi * freq * t + phi)
        return (x + noise).astype(x.dtype)

    @staticmethod
    def time_warping(
        x: np.ndarray,
        rng: np.random.Generator,
        max_warp: float = 0.05,
        num_knots: int = 5,
    ) -> np.ndarray:
        """
        Apply a smooth temporal distortion via cubic-spline warping.

        Algorithm:
          1. Place ``num_knots`` control points uniformly in [0, L−1].
          2. Perturb each control point by ±max_warp·L; clip to [0, L−1].
          3. Sort the perturbed positions (maintain monotonicity).
          4. Build a cubic mapping: original → warped using interp1d.
          5. Evaluate the mapping on the dense integer grid [0, L−1].
          6. Resample x on the resulting warped grid with linear interp.

        The output has the same length L as the input (no padding/trimming).
        """
        L = len(x)
        knot_orig = np.linspace(0.0, L - 1.0, num_knots)

        # Perturb knots
        delta = max_warp * L
        perturbation = rng.uniform(-delta, delta, size=num_knots)
        knot_warped = np.clip(knot_orig + perturbation, 0.0, L - 1.0)
        knot_warped = np.sort(knot_warped)   # enforce monotonicity

        # Build dense warp mapping: position i maps to warped position
        warp_fn = interp1d(
            knot_orig, knot_warped, kind="cubic",
            bounds_error=False,
            fill_value=(knot_warped[0], knot_warped[-1]),
        )
        dense_grid = np.arange(L, dtype=np.float64)
        warped_positions = np.clip(warp_fn(dense_grid), 0.0, L - 1.0)

        # Resample x at warped positions
        resample_fn = interp1d(
            dense_grid, x.astype(np.float64), kind="linear",
            bounds_error=False, fill_value=(float(x[0]), float(x[-1])),
        )
        return resample_fn(warped_positions).astype(x.dtype)

    @staticmethod
    def segment_masking(
        x: np.ndarray,
        rng: np.random.Generator,
        max_mask_ratio: float = 0.10,
    ) -> np.ndarray:
        """
        Zero out a randomly located contiguous segment.

            P     = int(max_mask_ratio * L)
            start ~ Uniform(0, L − P)
            x'[start : start+P] = 0
        """
        x_out = x.copy()
        P = int(max_mask_ratio * len(x))
        if P > 0:
            start = int(rng.integers(0, len(x) - P))
            x_out[start : start + P] = 0.0
        return x_out

    # ------------------------------------------------------------------
    # Main callable
    # ------------------------------------------------------------------

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply each augmentation independently with probability ``self.p``.

        The order is fixed:
          gaussian_noise → amplitude_scaling → baseline_wander →
          powerline_noise → time_warping → segment_masking

        Returns an array with the same dtype and shape as ``x``.
        """
        x = x.copy()
        augmentations = [
            lambda arr: self.gaussian_noise(arr, self.rng),
            lambda arr: self.amplitude_scaling(arr, self.rng),
            lambda arr: self.baseline_wander(arr, self.rng),
            lambda arr: self.powerline_noise(arr, self.rng),
            lambda arr: self.time_warping(arr, self.rng),
            lambda arr: self.segment_masking(arr, self.rng),
        ]
        for aug in augmentations:
            if self.rng.random() < self.p:
                x = aug(x)
        return x
