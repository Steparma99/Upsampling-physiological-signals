"""
Pan-Tompkins R-peak detector.

Implements the algorithm from:
  Pan J. & Tompkins W.J. (1985). A real-time QRS detection algorithm.
  IEEE Trans. Biomed. Eng., 32(3), 230–236.

Used by the artifact rejection module to compute RMSSD for window quality check.
"""

import numpy as np
from scipy import signal as sp_signal


def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """
    Detect R-peaks in an ECG signal using the Pan-Tompkins algorithm.

    Pipeline:
      1. Bandpass filter     5–15 Hz (isolates QRS complex)
      2. Differentiation     5-point derivative (slope information)
      3. Squaring            emphasizes large slopes, makes all values positive
      4. Moving integration  150 ms window (smooths & widens peaks)
      5. Adaptive thresholding + peak search with minimum RR distance

    Peak locations are then refined by searching for the maximum absolute
    amplitude in the original signal within a ±50 ms window.

    Parameters
    ----------
    ecg : np.ndarray
        1-D ECG signal.
    fs : float
        Sampling frequency of the ECG signal (Hz).

    Returns
    -------
    r_peaks : np.ndarray
        Array of R-peak sample indices (sorted, deduplicated).
        Empty array if fewer than 2 peaks are found.
    """
    if len(ecg) < int(0.5 * fs):
        return np.array([], dtype=np.int32)

    # 1. Bandpass filter (5–15 Hz, 1st-order Butterworth, zero-phase)
    nyq = fs / 2.0
    low = 5.0 / nyq
    high = min(15.0 / nyq, 0.99)
    b, a = sp_signal.butter(1, [low, high], btype="band")
    filtered = sp_signal.filtfilt(b, a, ecg.astype(np.float64))

    # 2. Five-point derivative
    #    y[n] = (−2x[n−2] − x[n−1] + x[n+1] + 2x[n+2]) / 8
    diff = np.zeros_like(filtered)
    diff[2:-2] = (
        -2.0 * filtered[:-4]
        - filtered[1:-3]
        + filtered[3:-1]
        + 2.0 * filtered[4:]
    ) / 8.0

    # 3. Squaring
    squared = diff ** 2

    # 4. Moving window integration (150 ms)
    win_size = max(1, int(0.150 * fs))
    kernel = np.ones(win_size) / win_size
    integrated = np.convolve(squared, kernel, mode="same")

    # 5. Peak detection
    # Initial threshold from first 2 s of signal
    init_len = min(int(2.0 * fs), len(integrated))
    threshold = 0.5 * np.max(integrated[:init_len])
    if threshold == 0.0:
        threshold = 0.5 * np.max(integrated)

    min_rr_samples = max(1, int(0.20 * fs))  # 200 ms → 300 bpm max

    candidate_peaks, _ = sp_signal.find_peaks(
        integrated, height=threshold, distance=min_rr_samples
    )

    if len(candidate_peaks) == 0:
        return np.array([], dtype=np.int32)

    # Refine to true R-peak in the original signal (search ±50 ms)
    search_radius = max(1, int(0.05 * fs))
    r_peaks = []
    for p in candidate_peaks:
        start = max(0, p - search_radius)
        end = min(len(ecg), p + search_radius + 1)
        local_idx = start + int(np.argmax(np.abs(ecg[start:end])))
        r_peaks.append(local_idx)

    return np.array(sorted(set(r_peaks)), dtype=np.int32)
