"""
DatasetBuilder — per-recording pipeline orchestration.

For each recording:
  1. Resample to fs_hr if needed (e.g. PTB Diagnostic 1000 Hz → 500 Hz)
  2. Run the degrading chain  D = Q ∘ N ∘ S ∘ F
  3. Extract overlapping windows
  4. Z-score normalize each window
  5. Run artifact rejection
  6. Detect R-peaks for annotation
  7. Collect accepted windows

Split management:
  - Patient-level split (80/10/10) to prevent data leakage

Output:
  - One HDF5 file per split: {train,val,test}.h5
  - Datasets: x_hr, x_lr, r_peaks, mu_w, sigma_w, record_ids, window_idxs
"""

import logging
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import h5py
import numpy as np
from scipy import signal as sp_signal

from .artifact_rejection import is_valid_window
from .degrading import degrade_signal, design_antialiasing_filter
from .pan_tompkins import detect_r_peaks
from .segmentation import extract_windows, zscore_normalize_window

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patient-level train/val/test split
# ---------------------------------------------------------------------------

def split_patients(
    patient_ids: List[str],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split a list of patient IDs into train / val / test sets.

    The split is by *patient*, not by window, to prevent data leakage.
    test_ratio is implicitly 1 - train_ratio - val_ratio.

    Parameters
    ----------
    patient_ids : list of str
        All unique patient identifiers.
    train_ratio, val_ratio : float
        Fraction of patients for train and validation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (train_ids, val_ids, test_ids) : tuple of lists
    """
    rng = np.random.default_rng(seed)
    ids = list(patient_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return ids[:n_train], ids[n_train : n_train + n_val], ids[n_train + n_val :]


# ---------------------------------------------------------------------------
# Main orchestration class
# ---------------------------------------------------------------------------

class DatasetBuilder:
    """
    Orchestrates the full Phase 1 pipeline for any number of datasets.

    Usage
    -----
    builder = DatasetBuilder(config)
    windows = builder.process_recording(signal, fs_source, record_id)
    builder.save_split(windows, "train", Path("data/processed"))
    builder.log_stats()
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.fs_hr: int = config["fs_hr"]
        self.fs_lr: int = config["fs_lr"]
        self.R: int = config["upsample_factor"]
        self.rng = np.random.default_rng(config["split"]["random_seed"])

        # Pre-design the anti-aliasing filter once (shared across all recordings)
        flt = config["filter"]
        self.h_aa = design_antialiasing_filter(
            fs_hr=float(self.fs_hr),
            fs_lr=float(self.fs_lr),
            stopband_attenuation_db=flt["stopband_attenuation_db"],
            transition_width_normalized=flt["transition_width_normalized"],
        )
        logger.info(
            "Anti-aliasing filter: %d taps, fc=%.3f (normalized)",
            len(self.h_aa),
            flt.get("cutoff_normalized", 0.1),
        )

        # Noise kwargs (forwarded to add_composite_noise)
        n = config["noise"]
        bw = n["baseline_wander"]
        self.noise_kwargs: dict = {
            "snr_db_range": tuple(n["snr_db_range"]),
            "powerline_freq": n["powerline_freq"],
            "powerline_amp_range": tuple(n["powerline_amp_range"]),
            "bw_freq_range": tuple(bw["freq_range"]),
            "bw_amp_range": tuple(bw["amp_range"]),
            "n_bw_components": bw["n_components"],
        }

        # Quantization kwargs
        q = config["quantization"]
        self.quantize: bool = q["enabled"]
        self.quant_kwargs: dict = {
            "bits": q["bits"],
            "v_min": q["voltage_range"][0],
            "v_max": q["voltage_range"][1],
        }

        # Segmentation config
        self.seg_cfg = config["segmentation"]

        # Artifact rejection config
        self.ar_cfg = config["artifact_rejection"]

        # Running statistics
        self.stats: Dict[str, int] = {
            "total": 0,
            "accepted": 0,
            "rejected_saturation": 0,
            "rejected_variance": 0,
            "rejected_sqi": 0,
            "rejected_rmssd": 0,
        }

    # ------------------------------------------------------------------

    def process_recording(
        self,
        x_hr: np.ndarray,
        fs_source: int,
        record_id: str,
    ) -> List[dict]:
        """
        Process a single recording through the full Phase 1 pipeline.

        Steps:
          1. Resample to fs_hr (if fs_source ≠ fs_hr)
          2. Generate x_lr via degrading chain
          3. Extract overlapping windows
          4. Z-score normalize each window
          5. Artifact rejection
          6. R-peak detection

        Parameters
        ----------
        x_hr : np.ndarray
            1-D signal at fs_source (mV, float).
        fs_source : int
            Sampling frequency of the input signal (Hz).
        record_id : str
            Unique identifier (for metadata in HDF5).

        Returns
        -------
        accepted : list of dict
            Each dict has keys: record_id, window_idx, x_hr, x_lr,
            r_peaks, mu_w, sigma_w.
        """
        x_hr = x_hr.astype(np.float64)

        # Step 1: resample to fs_hr if needed
        if fs_source != self.fs_hr:
            n_target = int(len(x_hr) * self.fs_hr / fs_source)
            x_hr = sp_signal.resample(x_hr, n_target)
            logger.debug(
                "%s: resampled %d Hz → %d Hz (%d → %d samples)",
                record_id, fs_source, self.fs_hr, len(x_hr), n_target,
            )

        # Step 2: generate LR signal via D = Q ∘ N ∘ S ∘ F
        x_lr = degrade_signal(
            x_hr=x_hr,
            h=self.h_aa,
            upsample_factor=self.R,
            fs_lr=float(self.fs_lr),
            noise_kwargs=self.noise_kwargs,
            quantize=self.quantize,
            quant_kwargs=self.quant_kwargs,
            rng=self.rng,
        )

        # Step 3: extract overlapping windows
        raw_windows = extract_windows(
            x_hr=x_hr,
            x_lr=x_lr,
            fs_hr=self.fs_hr,
            fs_lr=self.fs_lr,
            window_duration_s=self.seg_cfg["window_duration_s"],
            overlap_ratio=self.seg_cfg["overlap_ratio"],
        )

        accepted: List[dict] = []
        ar = self.ar_cfg

        for win_idx, (win_hr, win_lr) in enumerate(raw_windows):
            self.stats["total"] += 1

            # Step 4: z-score normalize
            win_hr_norm, win_lr_norm, mu_w, sigma_w = zscore_normalize_window(
                win_hr, win_lr
            )

            # Step 5: artifact rejection
            valid, reason = is_valid_window(
                win_hr_norm=win_hr_norm,
                win_hr_raw=win_hr,
                sigma_w=sigma_w,
                fs_hr=float(self.fs_hr),
                saturation_threshold=ar["saturation_zscore"],
                saturation_max_fraction=ar.get("saturation_max_fraction", 0.01),
                min_std=ar["min_std_mv"],
                max_std=ar["max_std_mv"],
                sqi_threshold=ar["sqi_threshold"],
                ecg_band=tuple(ar["sqi_ecg_band"]),
                rmssd_threshold_ms=ar["rmssd_threshold_ms"],
            )

            if not valid:
                key = f"rejected_{reason}"
                self.stats[key] = self.stats.get(key, 0) + 1
                continue

            # Step 6: R-peak detection for annotation
            r_peaks = detect_r_peaks(win_hr, float(self.fs_hr))

            self.stats["accepted"] += 1
            accepted.append(
                {
                    "record_id": record_id,
                    "window_idx": win_idx,
                    "x_hr": win_hr_norm,  # float32
                    "x_lr": win_lr_norm,  # float32
                    "r_peaks": r_peaks,   # int32
                    "mu_w": mu_w,
                    "sigma_w": sigma_w,
                }
            )

        return accepted

    # ------------------------------------------------------------------

    def save_split(
        self,
        windows: List[dict],
        split_name: str,
        output_dir: Path,
    ) -> Path:
        """
        Save a list of window dicts to an HDF5 file.

        HDF5 datasets:
          x_hr        (N, L_HR)  float32
          x_lr        (N, L_LR)  float32
          r_peaks     (N,)       variable-length int32
          mu_w        (N,)       float64
          sigma_w     (N,)       float64
          record_ids  (N,)       string
          window_idxs (N,)       int32

        Parameters
        ----------
        windows : list of dict
            Accepted windows from one or more recordings.
        split_name : str
            "train", "val", or "test".
        output_dir : Path
            Directory for the output .h5 file.

        Returns
        -------
        output_path : Path
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{split_name}.h5"

        n = len(windows)
        if n == 0:
            logger.warning("Split '%s': no windows to save — skipping.", split_name)
            return output_path

        L_HR = len(windows[0]["x_hr"])
        L_LR = len(windows[0]["x_lr"])

        with h5py.File(output_path, "w") as f:
            # Fixed-shape datasets
            f.create_dataset("x_hr", shape=(n, L_HR), dtype=np.float32,
                             chunks=(min(n, 256), L_HR), compression="gzip",
                             compression_opts=4)
            f.create_dataset("x_lr", shape=(n, L_LR), dtype=np.float32,
                             chunks=(min(n, 256), L_LR), compression="gzip",
                             compression_opts=4)
            f.create_dataset("mu_w", shape=(n,), dtype=np.float64)
            f.create_dataset("sigma_w", shape=(n,), dtype=np.float64)
            f.create_dataset("window_idxs", shape=(n,), dtype=np.int32)

            # Variable-length R-peaks
            vlen_int = h5py.vlen_dtype(np.dtype("int32"))
            f.create_dataset("r_peaks", shape=(n,), dtype=vlen_int)

            # Record IDs (string)
            record_ids_arr = np.array(
                [w["record_id"] for w in windows], dtype=h5py.string_dtype()
            )
            f.create_dataset("record_ids", data=record_ids_arr)

            # Fill row by row
            for i, w in enumerate(windows):
                f["x_hr"][i] = w["x_hr"]
                f["x_lr"][i] = w["x_lr"]
                f["mu_w"][i] = w["mu_w"]
                f["sigma_w"][i] = w["sigma_w"]
                f["window_idxs"][i] = w["window_idx"]
                f["r_peaks"][i] = w["r_peaks"]

            # Global metadata attributes
            f.attrs["split"] = split_name
            f.attrs["n_windows"] = n
            f.attrs["L_HR"] = L_HR
            f.attrs["L_LR"] = L_LR
            f.attrs["fs_hr"] = self.fs_hr
            f.attrs["fs_lr"] = self.fs_lr

        logger.info("Saved %d windows → %s", n, output_path)
        return output_path

    # ------------------------------------------------------------------

    def log_stats(self) -> None:
        """Print pipeline acceptance/rejection statistics."""
        total = self.stats["total"]
        accepted = self.stats["accepted"]
        rejected = total - accepted

        logger.info("=" * 55)
        logger.info("PHASE 1 PIPELINE STATISTICS")
        logger.info("  Total windows extracted : %10d", total)
        logger.info(
            "  Accepted                : %10d  (%5.1f%%)",
            accepted, 100.0 * accepted / max(total, 1),
        )
        logger.info(
            "  Rejected                : %10d  (%5.1f%%)",
            rejected, 100.0 * rejected / max(total, 1),
        )
        logger.info("  Rejection breakdown:")
        for reason in ("saturation", "variance", "sqi", "rmssd"):
            count = self.stats.get(f"rejected_{reason}", 0)
            logger.info(
                "    %-15s : %8d  (%5.1f%%)",
                reason, count, 100.0 * count / max(total, 1),
            )
        logger.info("=" * 55)
