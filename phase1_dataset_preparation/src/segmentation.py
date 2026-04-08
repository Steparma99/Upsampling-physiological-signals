"""
Segmentation and normalization of ECG signals into fixed-length windows.

Reference: project specification, Section 3.

Window parameters:
  T_w = 10 s
  L_HR = T_w × fs_hr = 5000 samples
  L_LR = T_w × fs_lr = 1000 samples
  Overlap = 50%  →  stride = 2500 HR samples

Z-score normalization uses HR statistics and applies them to both HR and LR
to preserve the linear scale relationship between input and ground truth.
"""

import numpy as np
from typing import List, Tuple


def extract_windows(
    x_hr: np.ndarray,
    x_lr: np.ndarray,
    fs_hr: int = 500,
    fs_lr: int = 100,
    window_duration_s: float = 10.0,
    overlap_ratio: float = 0.5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract overlapping windows from aligned (x_HR, x_LR) signal pair.

    The LR signal is assumed to be already decimated so that:
        x_lr[m]  ↔  x_hr[m × R],  R = fs_hr / fs_lr

    Parameters
    ----------
    x_hr : np.ndarray
        HR signal at fs_hr (1-D, length L_hr_total).
    x_lr : np.ndarray
        LR signal at fs_lr (1-D, length L_lr_total ≈ L_hr_total / R).
    fs_hr : int
        High-resolution sampling frequency (Hz).
    fs_lr : int
        Low-resolution sampling frequency (Hz).
    window_duration_s : float
        Window duration in seconds.
    overlap_ratio : float
        Fraction of window to overlap (0.5 = 50%).

    Returns
    -------
    windows : list of (win_hr, win_lr) tuples
        Each tuple contains aligned HR and LR windows.
    """
    R = fs_hr // fs_lr
    L_HR = int(window_duration_s * fs_hr)   # 5000
    L_LR = int(window_duration_s * fs_lr)   # 1000
    stride_HR = int((1.0 - overlap_ratio) * L_HR)  # 2500

    windows: List[Tuple[np.ndarray, np.ndarray]] = []
    start_hr = 0

    while start_hr + L_HR <= len(x_hr):
        win_hr = x_hr[start_hr : start_hr + L_HR]

        # Corresponding LR start index (exact, since R divides evenly)
        start_lr = start_hr // R
        win_lr = x_lr[start_lr : start_lr + L_LR]

        if len(win_hr) == L_HR and len(win_lr) == L_LR:
            windows.append((win_hr.copy(), win_lr.copy()))

        start_hr += stride_HR

    return windows


def zscore_normalize_window(
    win_hr: np.ndarray,
    win_lr: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Z-score normalize an (x_HR, x_LR) window pair using HR statistics.

    Computing normalization parameters from x_HR and applying them to both
    ensures the linear relationship x_LR ≈ x_HR (after upsampling) is
    preserved in the normalized space.

    Parameters
    ----------
    win_hr : np.ndarray
        HR window (1-D, L_HR samples).
    win_lr : np.ndarray
        LR window (1-D, L_LR samples).
    eps : float
        Floor for sigma to avoid division by zero.

    Returns
    -------
    win_hr_norm : np.ndarray
        Normalized HR window.
    win_lr_norm : np.ndarray
        Normalized LR window (using HR statistics).
    mu_w : float
        Per-window mean (from HR), needed for denormalization.
    sigma_w : float
        Per-window std (from HR), needed for denormalization.
    """
    mu_w = float(np.mean(win_hr))
    sigma_w = float(np.std(win_hr, ddof=1))

    if sigma_w < eps:
        sigma_w = eps

    win_hr_norm = ((win_hr - mu_w) / sigma_w).astype(np.float32)
    win_lr_norm = ((win_lr - mu_w) / sigma_w).astype(np.float32)

    return win_hr_norm, win_lr_norm, mu_w, sigma_w
