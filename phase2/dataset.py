"""
Phase 2 dataset wrapper.

Reads Phase 1 HDF5 output files (train.h5, val.h5, test.h5) and exposes
them as a PyTorch Dataset suitable for self-supervised pre-training.

Additions over the raw HDF5:
  - Loads or computes the physiological feature vector f from r_peaks.
  - Applies z-score normalisation to f using f_stats.npz from Phase 1
    (or a default identity transform if f_stats.npz is not yet available).
  - Converts r_peaks to the (K, 2) annotation format expected by Phase 2.
  - Supports curriculum source-filtering (Phase A: "native" only).

f computation fallback:
  Phase 1 saves r_peaks (HR-space integer positions) but not the full f
  vector.  When f is absent from the HDF5, this module derives a partial f
  from r_peaks (features f[0]–f[7]).  Features f[8]–f[12] (QT, QTc, A_QRS,
  T_peak, Pol_T) require PQRST detection and are set to 0 until Phase 1 is
  extended to produce them.
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_F_DIM = 13
_FS_HR = 500.0      # HR sampling frequency (Hz)
_L_HR  = 5000       # HR window length (samples)


# ---------------------------------------------------------------------------
# Feature computation from R-peaks
# ---------------------------------------------------------------------------

def _compute_f_from_r_peaks(r_peaks: np.ndarray) -> np.ndarray:
    """
    Compute the physiological feature vector f from R-peak positions.

    Args:
        r_peaks: 1-D int array of R-peak positions in HR samples (500 Hz).

    Returns:
        f: float32 array shape (13,).  Features f[8]–f[12] are 0 (need PQRST).
    """
    f = np.zeros(_F_DIM, dtype=np.float32)

    if len(r_peaks) < 2:
        return f

    # RR intervals in seconds
    rr_s = np.diff(r_peaks.astype(np.float64)) / _FS_HR

    f[0] = 60.0 / float(np.mean(rr_s))     # HR (bpm)
    f[1] = float(np.std(rr_s))              # SDNN (s)

    if len(rr_s) >= 2:
        drr = np.diff(rr_s)
        f[2] = float(np.sqrt(np.mean(drr ** 2)))       # RMSSD (s)
        f[3] = float(np.mean(np.abs(drr) > 0.02))      # pNN20 (fraction)

    # Locate the R-peak closest to the window centre (sample 2500)
    mid = _L_HR // 2
    dists = np.abs(r_peaks - mid)
    c = int(np.argmin(dists))   # index of central R-peak within r_peaks array

    rr_mean = float(np.mean(rr_s))

    # RR_prev: interval ending at the central R-peak
    f[4] = float(rr_s[c - 1]) if (c > 0 and c - 1 < len(rr_s)) else rr_mean
    # RR_curr: interval starting at the central R-peak
    f[5] = float(rr_s[c])     if (c < len(rr_s))                 else rr_mean
    # RR_next: interval after the central R-peak
    f[6] = float(rr_s[c + 1]) if (c + 1 < len(rr_s))            else rr_mean
    # delta_RR: relative deviation of RR_curr from the mean
    f[7] = (f[5] - rr_mean) / (rr_mean + 1e-8)

    # f[8]–f[12]: QT, QTc, A_QRS, T_peak, Pol_T — require PQRST detection
    # Leave as 0 until Phase 1 is extended to export these features.

    return f


# ---------------------------------------------------------------------------
# Custom collate function
# ---------------------------------------------------------------------------

def phase2_collate_fn(batch: list[dict]) -> dict:
    """
    Collate a list of Phase2Dataset items into a batch dict.

    The annotation tensor ``a`` has variable number of rows K per sample
    and cannot be stacked with the default collate.  This function pads
    ``a`` to the maximum K in the batch with zeros.

    Returns:
        {
            "x_lr":   FloatTensor (B, L),
            "f":      FloatTensor (B, 13),
            "a":      FloatTensor (B, max_K, 2),  zero-padded
            "source": list[str]  length B,
        }
    """
    x_lr   = torch.stack([item["x_lr"] for item in batch])   # (B, L)
    f      = torch.stack([item["f"]    for item in batch])   # (B, 13)
    source = [item["source"] for item in batch]

    max_k = max(item["a"].shape[0] for item in batch)
    B = len(batch)

    if max_k > 0:
        a_padded = torch.zeros(B, max_k, 2, dtype=torch.float32)
        for i, item in enumerate(batch):
            k = item["a"].shape[0]
            if k > 0:
                a_padded[i, :k, :] = item["a"]
    else:
        a_padded = torch.zeros(B, 0, 2, dtype=torch.float32)

    return {"x_lr": x_lr, "f": f, "a": a_padded, "source": source}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Phase2Dataset(Dataset):
    """
    PyTorch Dataset over a Phase 1 HDF5 split file.

    Parameters
    ----------
    h5_path:       Path to the HDF5 file (e.g. data/processed/train.h5).
    f_stats_path:  Path to f_stats.npz (mean, std shape (13,)).
                   If None, no normalisation is applied to f.
    source:        Source tag for all samples in this file ("native" or
                   "simulated").  Used by the curriculum Phase A filter.
    filter_source: If set, __getitem__ raises IndexError for items whose
                   source does not match.  Prefer filtering at DataLoader
                   level via the training loop instead.

    __getitem__ returns:
        {
            "x_lr":   FloatTensor (1000,)   — normalised LR ECG
            "f":      FloatTensor (13,)     — normalised physio features
            "a":      FloatTensor (K, 2)    — PQRST annotations (col0=type, col1=pos_hr)
            "source": str
        }

    Note: x_hr, mu_w, sigma_w are not loaded — not needed for pre-training.
    Augmentations are applied in the training loop (not here) so two views
    of the same sample can be generated in the same training step.
    """

    def __init__(
        self,
        h5_path: str | Path,
        f_stats_path: str | Path | None = None,
        source: str = "native",
        filter_source: str | None = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.source = source
        self.filter_source = filter_source

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 not found: {self.h5_path}")

        with h5py.File(self.h5_path, "r") as hf:
            self.n = int(hf.attrs.get("n_windows", hf["x_lr"].shape[0]))
            self._has_f = "f" in hf
            self._has_a = "a" in hf

        logger.info(
            "Phase2Dataset: %s | n=%d | has_f=%s | has_a=%s | source=%s",
            self.h5_path.name, self.n, self._has_f, self._has_a, source,
        )

        # f normalisation statistics
        if f_stats_path is not None:
            stats = np.load(f_stats_path)
            self._f_mean = stats["mean"].astype(np.float32)   # (13,)
            self._f_std  = stats["std"].astype(np.float32)    # (13,)
        else:
            self._f_mean = np.zeros(_F_DIM, dtype=np.float32)
            self._f_std  = np.ones(_F_DIM,  dtype=np.float32)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        with h5py.File(self.h5_path, "r") as hf:
            x_lr = hf["x_lr"][idx].astype(np.float32)       # (1000,)

            # Load or compute f
            if self._has_f:
                f_raw = hf["f"][idx].astype(np.float32)      # (13,)
            else:
                r_peaks = np.array(hf["r_peaks"][idx], dtype=np.int32)
                f_raw = _compute_f_from_r_peaks(r_peaks)

            # Load or construct annotations a
            if self._has_a:
                a = hf["a"][idx].astype(np.float32)           # (K, 2)
            else:
                r_peaks_arr = np.array(hf["r_peaks"][idx], dtype=np.float32)
                K = len(r_peaks_arr)
                if K > 0:
                    a = np.column_stack([
                        np.zeros(K, dtype=np.float32),   # type = 0 (R-peak)
                        r_peaks_arr,                     # position in HR samples
                    ])                                   # (K, 2)
                else:
                    a = np.zeros((0, 2), dtype=np.float32)

        # Normalise f: z-score for f[0:12], identity for f[12] (Pol_T: already ±1)
        f_norm = (f_raw - self._f_mean) / (self._f_std + 1e-8)
        f_norm[12] = f_raw[12]   # Pol_T: pass-through

        return {
            "x_lr":   torch.from_numpy(x_lr),
            "f":      torch.from_numpy(f_norm),
            "a":      torch.from_numpy(a),
            "source": self.source,
        }

    # ------------------------------------------------------------------
    # Statistics utility
    # ------------------------------------------------------------------

    @staticmethod
    def compute_and_save_f_stats(
        train_h5_path: str | Path,
        output_path: str | Path,
    ) -> None:
        """
        Compute f statistics over a training HDF5 file and save as f_stats.npz.

        Statistics are computed from raw (un-normalised) f values or from the
        r_peaks fallback when f is not present in the HDF5.

        Args:
            train_h5_path: Path to train.h5 (must NOT have f_stats applied).
            output_path:   Where to save f_stats.npz.

        Saves:
            f_stats.npz with arrays ``mean`` (13,) and ``std`` (13,).
            std[12] is forced to 1 and mean[12] to 0 so Pol_T passes through
            unchanged.
        """
        # Use a fresh dataset without any normalisation applied
        tmp = Phase2Dataset(train_h5_path, f_stats_path=None)
        n = len(tmp)
        logger.info("Computing f_stats over %d training samples…", n)

        f_sum    = np.zeros(_F_DIM, dtype=np.float64)
        f_sq_sum = np.zeros(_F_DIM, dtype=np.float64)

        for i in range(n):
            item = tmp[i]
            f = item["f"].numpy().astype(np.float64)   # no-op normalisation (mean=0,std=1)
            f_sum    += f
            f_sq_sum += f ** 2

        mean = (f_sum / n).astype(np.float32)
        var  = np.maximum(f_sq_sum / n - (f_sum / n) ** 2, 0.0)
        std  = np.sqrt(var).astype(np.float32)

        # Prevent division by zero for constant features
        std[std < 1e-8] = 1.0

        # Pol_T (index 12): already ±1, no scaling
        mean[12] = 0.0
        std[12]  = 1.0

        np.savez(output_path, mean=mean, std=std)
        logger.info("f_stats saved → %s  (mean range [%.3f, %.3f])",
                    output_path, float(mean.min()), float(mean.max()))
