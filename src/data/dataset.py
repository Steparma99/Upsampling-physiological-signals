"""
dataset.py — PyTorch Dataset classes for the ECG super-resolution pipeline.

Four Dataset classes + one tuple-assembly function + one DataLoader factory:

  ECGDataset          — abstract base; handles __getitem__ via build_training_tuple
  PTBXLDataset        — native 100/500 Hz pairs; reads CSV metadata; no downsampling
  MITBIHDataset       — 360 Hz wfdb records; resampled to FS_HR; LR simulated
  CPSC2018Dataset     — 500 Hz wfdb records; LR simulated
  build_training_tuple() — applies zscore, detect_r_peaks, d_map, f_signal, f_meta
  get_dataloader()    — DataLoader factory with collate_fn for the training tuple

Training tuple returned by __getitem__:
  (x_LR_hat, x_HR_hat, d_map, f_signal, f_meta,
   pqrst_annotations, mu_w, sigma_w, source_flag)
  source_flag: SOURCE_NATIVE (0) or SOURCE_SIMULATED (1)
  pqrst_annotations: dict with HR-domain sample indices, or None
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import wfdb
from numpy.typing import NDArray
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset

from .features import DIAGNOSTIC_CLASSES, build_f_meta, build_f_signal
from .preprocessing import (
    FS_HR,
    FS_LR,
    L_HR,
    L_LR,
    apply_normalization,
    compute_r_distance_map,
    detect_r_peaks,
    downsample_signal,
    zscore_normalize,
)

logger = logging.getLogger(__name__)

# Integer encoding of source_flag (easier to batch than strings)
SOURCE_NATIVE: int = 0
SOURCE_SIMULATED: int = 1


def _resample_signal(
    signal: NDArray[np.floating],
    fs_in: float,
    fs_out: float,
    target_len: int,
) -> NDArray[np.float64]:
    """Resample with polyphase filtering when rates differ, else trim/pad safely."""
    sig = np.asarray(signal, dtype=np.float64)

    if abs(fs_in - fs_out) <= 1e-6:
        if len(sig) >= target_len:
            return sig[:target_len].astype(np.float64, copy=False)
        padded = np.zeros(target_len, dtype=np.float64)
        padded[: len(sig)] = sig
        return padded

    resampled = resample_poly(sig, up=int(round(fs_out)), down=int(round(fs_in)))
    if len(resampled) >= target_len:
        return resampled[:target_len].astype(np.float64, copy=False)

    padded = np.zeros(target_len, dtype=np.float64)
    padded[: len(resampled)] = resampled
    return padded


# =============================================================================
# Training tuple assembly
# =============================================================================

def build_training_tuple(
    signal_hr: NDArray[np.floating],
    signal_lr: NDArray[np.floating],
    age: float | None = None,
    sex: float | None = None,
    diagnostic_labels: dict[str, bool] | None = None,
    pqrst_annotations: dict[str, Any] | None = None,
    source_flag: int = SOURCE_NATIVE,
    p_drop_meta: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[
    NDArray[np.float32],   # x_LR_hat  (L_LR,)
    NDArray[np.float32],   # x_HR_hat  (L_HR,)
    NDArray[np.float32],   # d_map     (L_LR,)
    NDArray[np.float32],   # f_signal  (12,)
    NDArray[np.float32],   # f_meta    (7,)
    dict[str, Any] | None, # pqrst_annotations
    np.float32,            # mu_w
    np.float32,            # sigma_w
    int,                   # source_flag
]:
    """Assemble one complete training example from a raw signal pair.

    Single entry point for building training examples — both Dataset subclasses
    and external callers use this function to guarantee a consistent tuple layout.

    Normalisation parameters mu_w and sigma_w are always computed from x_LR
    only (Eq. 3 PDF) and then applied to both x_LR and x_HR (Eq. 4 PDF).
    This preserves the amplitude relationship and ensures inference is possible
    without access to x_HR (CLAUDE.md constraint #1).

    Args:
        signal_hr: Raw HR window at FS_HR Hz, shape (L_HR,) = (5000,).
        signal_lr: Raw LR window at FS_LR Hz, shape (L_LR,) = (1000,).
        age: Patient age in years, or None.
        sex: 0.0 (male) / 1.0 (female), or None.
        diagnostic_labels: Multi-hot dict keyed by DIAGNOSTIC_CLASSES, or None.
        pqrst_annotations: PQRST landmark sample indices in the HR domain, or None.
        source_flag: SOURCE_NATIVE (0) or SOURCE_SIMULATED (1).
        p_drop_meta: Metadata dropout probability (0.0 at inference, 0.3 in training).
        rng: Optional seeded numpy Generator; used for metadata dropout.

    Returns:
        9-element tuple as documented in the module docstring.
    """
    # --- Z-score normalisation (Eq. 3-4 PDF) ---
    # Parameters from x_LR; applied to both signals to keep scale consistent.
    x_lr_hat, mu_w, sigma_w = zscore_normalize(signal_lr)
    x_hr_hat = apply_normalization(signal_hr, mu_w, sigma_w)

    # --- R-peak detection and R-distance map ---
    r_peaks = detect_r_peaks(signal_lr, FS_LR)
    d_map = compute_r_distance_map(r_peaks, length=L_LR)

    # --- Conditioning vectors ---
    f_signal = build_f_signal(signal_lr, mu_w, sigma_w, r_peaks, FS_LR)
    f_meta = build_f_meta(age, sex, diagnostic_labels, p_drop=p_drop_meta, rng=rng)

    return (
        x_lr_hat,
        x_hr_hat,
        d_map,
        f_signal,
        f_meta,
        pqrst_annotations,
        np.float32(mu_w),
        np.float32(sigma_w),
        source_flag,
    )


# =============================================================================
# Abstract base Dataset
# =============================================================================

class ECGDataset(Dataset, ABC):
    """Abstract base for ECG super-resolution datasets.

    Subclasses implement _load_record(idx) which returns the raw signal pair
    plus a metadata dict; __getitem__ delegates to build_training_tuple.

    The per-sample RNG is derived from (seed, idx) so that dropout is
    deterministic across runs and fork-safe with num_workers > 0.
    """

    def __init__(
        self,
        p_drop_meta: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.p_drop_meta = p_drop_meta
        self._seed = seed

    def __len__(self) -> int:
        return len(self._records)  # type: ignore[attr-defined]

    def __getitem__(self, idx: int) -> tuple:
        # Fork-safe: each sample gets its own deterministic rng
        rng = (
            np.random.default_rng(self._seed + idx)
            if self._seed is not None
            else None
        )
        signal_hr, signal_lr, meta = self._load_record(idx)
        return build_training_tuple(
            signal_hr=signal_hr,
            signal_lr=signal_lr,
            age=meta.get("age"),
            sex=meta.get("sex"),
            diagnostic_labels=meta.get("diagnostic_labels"),
            pqrst_annotations=meta.get("pqrst_annotations"),
            source_flag=meta.get("source_flag", SOURCE_NATIVE),
            p_drop_meta=self.p_drop_meta,
            rng=rng,
        )

    @abstractmethod
    def _load_record(
        self, idx: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
        """Load raw (signal_hr, signal_lr, metadata) for record at idx."""


# =============================================================================
# PTB-XL — native 100/500 Hz pairs
# =============================================================================

class PTBXLDataset(ECGDataset):
    """PTB-XL dataset with native 100/500 Hz recording pairs.

    PTB-XL provides both lr (100 Hz) and hr (500 Hz) recordings for every
    record, so no simulated downsampling is needed (CLAUDE.md constraint #8).
    Patient age, sex, and superclass-level diagnostic labels are loaded from
    the accompanying CSV files.

    Standard PTB-XL directory layout::

        root/
          ptbxl_database.csv
          scp_statements.csv
          records100/00000/00001_lr.dat  00001_lr.hea  ...
          records500/00000/00001_hr.dat  00001_hr.hea  .../clear

    Stratified 10-fold split follows the PTB-XL paper convention:
      train → folds 1-8, val → fold 9, test → fold 10.

    Args:
        root: Path to the PTB-XL root directory.
        split: "train", "val", or "test".
        channel: Lead index to load (default 0 = Lead I; 1 = Lead II).
        p_drop_meta: Metadata dropout probability (training only; 0.0 at inference).
        seed: Random seed for deterministic dropout.
    """

    _FOLDS: dict[str, list[int]] = {
        "train": list(range(1, 9)),
        "val": [9],
        "test": [10],
    }

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        channel: int = 0,
        p_drop_meta: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(p_drop_meta=p_drop_meta, seed=seed)
        if split not in self._FOLDS:
            raise ValueError(f"Invalid PTB-XL split: {split}. Expected one of {tuple(self._FOLDS)}")
        self.root = Path(root)
        self.channel = channel
        self._scp_map = self._build_scp_map()
        self._metadata, self._records = self._build_index(split)

    # ------------------------------------------------------------------

    def _build_scp_map(self) -> dict[str, str]:
        """Build {scp_code: diagnostic_superclass} from scp_statements.csv."""
        scp_path = self.root / "scp_statements.csv"
        if not scp_path.exists():
            logger.warning("scp_statements.csv not found at %s — diagnostics will be empty.", scp_path)
            return {}
        df = pd.read_csv(scp_path, index_col=0)
        if "diagnostic_class" not in df.columns:
            return {}
        return {
            str(code): str(row["diagnostic_class"])
            for code, row in df.iterrows()
            if pd.notna(row.get("diagnostic_class"))
            and str(row["diagnostic_class"]) in DIAGNOSTIC_CLASSES
        }

    def _build_index(self, split: str) -> tuple[pd.DataFrame, list[int]]:
        csv_path = self.root / "ptbxl_database.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"ptbxl_database.csv not found: {csv_path}")
        df = pd.read_csv(csv_path, index_col="ecg_id")
        folds = self._FOLDS[split]
        df = df[df["strat_fold"].isin(folds)].copy()
        return df, list(range(len(df)))

    def _parse_diagnostics(self, scp_codes_raw: str) -> dict[str, bool]:
        """Convert scp_codes JSON string → binary DIAGNOSTIC_CLASSES dict."""
        labels: dict[str, bool] = {cls: False for cls in DIAGNOSTIC_CLASSES}
        try:
            # PTB-XL stores scp_codes as a Python repr string; normalise to JSON
            codes: dict[str, float] = json.loads(scp_codes_raw.replace("'", '"'))
            for code, likelihood in codes.items():
                superclass = self._scp_map.get(code)
                if superclass and float(likelihood) >= 50.0:
                    labels[superclass] = True
        except Exception as exc:
            logger.debug("scp_codes parse failed: %s", exc)
        return labels

    def _load_record(
        self, idx: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
        row = self._metadata.iloc[idx]

        # HR signal (500 Hz) — native, no resampling needed
        rec_hr = wfdb.rdrecord(str(self.root / row["filename_hr"]))
        sig_hr = rec_hr.p_signal[:L_HR, self.channel].astype(np.float64)

        # LR signal (100 Hz) — native pair, no downsampling (constraint #8)
        rec_lr = wfdb.rdrecord(str(self.root / row["filename_lr"]))
        sig_lr = rec_lr.p_signal[:L_LR, self.channel].astype(np.float64)

        age = float(row["age"]) if pd.notna(row.get("age")) else None
        sex = float(row["sex"]) if pd.notna(row.get("sex")) else None
        diag = self._parse_diagnostics(str(row.get("scp_codes", "{}")))

        meta: dict[str, Any] = {
            "age": age,
            "sex": sex,
            "diagnostic_labels": diag,
            "pqrst_annotations": None,
            "source_flag": SOURCE_NATIVE,
        }
        return sig_hr, sig_lr, meta


# =============================================================================
# MIT-BIH — simulated LR-HR pairs
# =============================================================================

class MITBIHDataset(ECGDataset):
    """MIT-BIH Arrhythmia Database with simulated LR-HR pairs.

    MIT-BIH records are natively at 360 Hz. To produce a 500 Hz HR target,
    the native signal is resampled to FS_HR via scipy.signal.resample. The LR
    counterpart is then derived by applying downsample_signal() (anti-alias
    FIR + decimation), satisfying CLAUDE.md constraints #8 and #9.

    Long recordings (≈30 minutes per record) are split into non-overlapping
    10-second windows during index construction. The window list is built
    lazily from the wfdb headers at construction time.

    Standard MIT-BIH layout::

        root/
          100.dat  100.hea
          101.dat  101.hea
          ...

    Args:
        root: Path to MIT-BIH data directory.
        channel: Lead index (default 0 = MLII).
        window_sec: Window duration in seconds (default 10.0).
        overlap_sec: Window overlap in seconds (default 0.0 = non-overlapping).
        p_drop_meta: Metadata dropout probability (no diagnostics for MIT-BIH).
        seed: Random seed.
    """

    def __init__(
        self,
        root: str | Path,
        channel: int = 0,
        window_sec: float = 10.0,
        overlap_sec: float = 0.0,
        p_drop_meta: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(p_drop_meta=p_drop_meta, seed=seed)
        if overlap_sec < 0.0 or overlap_sec >= window_sec:
            raise ValueError("overlap_sec must satisfy 0 <= overlap_sec < window_sec")
        self.root = Path(root)
        self.channel = channel
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec
        self._records = self._build_index()

    def _build_index(self) -> list[tuple[Path, int, float]]:
        """Return list of (record_path, start_sample_at_FS_HR, native_fs)."""
        windows: list[tuple[Path, int, float]] = []
        for hea in sorted(self.root.glob("*.hea")):
            rec_path = hea.with_suffix("")
            try:
                hdr = wfdb.rdheader(str(rec_path))
            except Exception as exc:
                logger.warning("Cannot read header %s: %s", rec_path, exc)
                continue
            native_fs = float(hdr.fs)
            n_hr = int(hdr.sig_len * FS_HR / native_fs)
            win_hr = int(self.window_sec * FS_HR)
            step_hr = int((self.window_sec - self.overlap_sec) * FS_HR)
            for start in range(0, n_hr - win_hr + 1, step_hr):
                windows.append((rec_path, start, native_fs))
        if not windows:
            logger.warning("MITBIHDataset: no records found in %s", self.root)
        return windows

    def _load_record(
        self, idx: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
        rec_path, start_hr, native_fs = self._records[idx]

        # Map window start/end from FS_HR domain back to native sample domain
        start_native = int(start_hr * native_fs / FS_HR)
        n_native_needed = int(self.window_sec * native_fs) + 2

        rec = wfdb.rdrecord(
            str(rec_path),
            sampfrom=start_native,
            sampto=start_native + n_native_needed,
            channels=[self.channel],
        )
        sig_native = rec.p_signal[:, 0].astype(np.float64)

        # Resample native → FS_HR (synthetic HR ground truth)
        sig_hr = _resample_signal(sig_native, fs_in=native_fs, fs_out=FS_HR, target_len=L_HR)

        # Simulate LR via anti-alias FIR + decimation (constraints #8, #9)
        sig_lr = downsample_signal(sig_hr).astype(np.float64)

        meta: dict[str, Any] = {
            "age": None,
            "sex": None,
            "diagnostic_labels": None,
            "pqrst_annotations": None,
            "source_flag": SOURCE_SIMULATED,
        }
        return sig_hr, sig_lr, meta


# =============================================================================
# CPSC2018 — simulated LR-HR pairs
# =============================================================================

class CPSC2018Dataset(ECGDataset):
    """CPSC 2018 ECG Challenge dataset with simulated LR-HR pairs.

    CPSC2018 records are natively at 500 Hz (= FS_HR), so no resampling is
    needed. LR is derived by applying downsample_signal() directly.

    Recordings vary in length (6-60 s); the index construction windows each
    recording into 10-second segments, discarding the last partial window.

    Standard CPSC2018 WFDB layout::

        root/
          A0001.hea  A0001.mat   (WFDB + MATLAB data)
          A0002.hea  A0002.mat
          ...

    Args:
        root: Path to CPSC2018 data directory.
        channel: Lead index (default 0 = Lead I).
        window_sec: Window duration in seconds (default 10.0).
        overlap_sec: Window overlap in seconds (default 0.0).
        p_drop_meta: Metadata dropout probability.
        seed: Random seed.
    """

    def __init__(
        self,
        root: str | Path,
        channel: int = 0,
        window_sec: float = 10.0,
        overlap_sec: float = 0.0,
        p_drop_meta: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(p_drop_meta=p_drop_meta, seed=seed)
        if overlap_sec < 0.0 or overlap_sec >= window_sec:
            raise ValueError("overlap_sec must satisfy 0 <= overlap_sec < window_sec")
        self.root = Path(root)
        self.channel = channel
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec
        self._records = self._build_index()

    def _build_index(self) -> list[tuple[Path, int]]:
        """Return list of (record_path, start_sample_at_FS_HR)."""
        windows: list[tuple[Path, int]] = []
        for hea in sorted(self.root.glob("*.hea")):
            rec_path = hea.with_suffix("")
            try:
                hdr = wfdb.rdheader(str(rec_path))
            except Exception as exc:
                logger.warning("Cannot read header %s: %s", rec_path, exc)
                continue
            native_fs = float(hdr.fs)
            if abs(native_fs - FS_HR) > 1.0:
                logger.warning(
                    "CPSC2018 record %s has fs=%.0f Hz (expected %d Hz) — will resample.",
                    rec_path.name, native_fs, FS_HR,
                )
            n_samples = int(hdr.sig_len * FS_HR / native_fs)
            win = int(self.window_sec * FS_HR)
            step = int((self.window_sec - self.overlap_sec) * FS_HR)
            for start in range(0, n_samples - win + 1, step):
                windows.append((rec_path, start))
        if not windows:
            logger.warning("CPSC2018Dataset: no records found in %s", self.root)
        return windows

    def _load_record(
        self, idx: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
        rec_path, start_hr = self._records[idx]

        hdr = wfdb.rdheader(str(rec_path))
        native_fs = float(hdr.fs)
        start_native = int(start_hr * native_fs / FS_HR)
        n_native_needed = int(self.window_sec * native_fs) + 2

        rec = wfdb.rdrecord(
            str(rec_path),
            sampfrom=start_native,
            sampto=start_native + n_native_needed,
            channels=[self.channel],
        )
        sig_native = rec.p_signal[:, 0].astype(np.float64)

        # Resample to FS_HR if needed (handles records not exactly at 500 Hz)
        sig_hr = _resample_signal(sig_native, fs_in=native_fs, fs_out=FS_HR, target_len=L_HR)

        sig_lr = downsample_signal(sig_hr).astype(np.float64)

        meta: dict[str, Any] = {
            "age": None,
            "sex": None,
            "diagnostic_labels": None,
            "pqrst_annotations": None,
            "source_flag": SOURCE_SIMULATED,
        }
        return sig_hr, sig_lr, meta


# =============================================================================
# Collate function and DataLoader factory
# =============================================================================

def _annotations_to_tensor(
    pqrst_list: list[dict[str, Any] | None],
    default_half_window: int = 25,
) -> torch.Tensor | None:
    """Convert a batch of PQRST annotation dicts to a (B, K, 2) tensor.

    Each dict may contain keys like "P", "Q", "R", "S", "T" mapping to lists
    of HR-domain sample indices. All dicts are padded to the same K (max
    landmarks in the batch) with position = -1 for missing entries.

    Position -1 is the sentinel used by MorphologicalLoss to skip invalid
    landmarks (matching MorphologicalLoss.forward's ``valid_mask = pos >= 0``).

    Args:
        pqrst_list: B-length list of annotation dicts or None values.
        default_half_window: Half-window size applied to all landmarks.

    Returns:
        (B, K, 2) int32 tensor with columns [position, half_window],
        or None if every element in pqrst_list is None.
    """
    if all(a is None for a in pqrst_list):
        return None

    # Collect all (position, half_window) pairs per sample
    per_sample: list[list[tuple[int, int]]] = []
    for ann in pqrst_list:
        pairs: list[tuple[int, int]] = []
        if ann is not None:
            for positions in ann.values():
                if isinstance(positions, (list, np.ndarray)):
                    for pos in positions:
                        pairs.append((int(pos), default_half_window))
                elif isinstance(positions, (int, float, np.integer)):
                    pairs.append((int(positions), default_half_window))
        per_sample.append(pairs)

    K = max((len(p) for p in per_sample), default=0)
    if K == 0:
        return None

    B = len(pqrst_list)
    out = torch.full((B, K, 2), fill_value=-1, dtype=torch.int32)
    for b, pairs in enumerate(per_sample):
        for k, (pos, hw) in enumerate(pairs):
            out[b, k, 0] = pos
            out[b, k, 1] = hw
    return out


def _ecg_collate_fn(
    batch: list[tuple],
) -> tuple[
    torch.Tensor,              # x_lr    (B, L_LR)
    torch.Tensor,              # x_hr    (B, L_HR)
    torch.Tensor,              # d_map   (B, L_LR)
    torch.Tensor,              # f_signal (B, 12)
    torch.Tensor,              # f_meta  (B, 7)
    torch.Tensor | None,       # pqrst_annotations — (B, K, 2) or None
    torch.Tensor,              # mu_w    (B,)
    torch.Tensor,              # sigma_w (B,)
    torch.Tensor,              # source_flag (B,)
]:
    """Collate training tuples into a batched PyTorch tuple.

    PQRST annotations are converted from list[dict | None] to a
    (B, K, 2) int32 tensor [position, half_window] via _annotations_to_tensor,
    so that MorphologicalLoss can consume them directly. Returns None when the
    entire batch has no annotations.
    """
    x_lr, x_hr, d_map, f_sig, f_meta, pqrst, mu, sigma, flags = zip(*batch)

    return (
        torch.from_numpy(np.stack(x_lr)),                       # (B, L_LR)
        torch.from_numpy(np.stack(x_hr)),                       # (B, L_HR)
        torch.from_numpy(np.stack(d_map)),                      # (B, L_LR)
        torch.from_numpy(np.stack(f_sig)),                      # (B, 12)
        torch.from_numpy(np.stack(f_meta)),                     # (B, 7)
        _annotations_to_tensor(list(pqrst)),                    # (B, K, 2) | None
        torch.tensor(mu, dtype=torch.float32),                  # (B,)
        torch.tensor(sigma, dtype=torch.float32),               # (B,)
        torch.tensor(flags, dtype=torch.int64),                 # (B,)
    )


def get_dataloader(
    dataset: ECGDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
    prefetch_factor: int | None = None,
) -> DataLoader:
    """Create a DataLoader for an ECGDataset with the appropriate collate_fn.

    Args:
        dataset: Any ECGDataset subclass (PTBXLDataset, MITBIHDataset, CPSC2018Dataset).
        batch_size: Number of examples per batch.
        shuffle: Shuffle at each epoch (True for training, False for val/test).
        num_workers: Parallel data-loading workers (0 = main process only).
        pin_memory: Enable pinned memory for faster CPU→GPU transfer.
        drop_last: Drop the last incomplete batch (recommended for training).
        prefetch_factor: Batches to prefetch per worker (None = PyTorch default).

    Returns:
        Configured DataLoader instance.
    """
    kwargs: dict[str, Any] = {}
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_ecg_collate_fn,
        **kwargs,
    )
