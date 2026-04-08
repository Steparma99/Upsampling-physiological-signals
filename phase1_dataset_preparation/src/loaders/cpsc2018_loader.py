"""
CPSC2018 Challenge Dataset Loader

Dataset: China Physiological Signal Challenge 2018
  URL:  http://2018.icbeb.org/Challenge.html
  fs:   500 Hz
  N:    6 877 recordings (12-lead ECG, variable length)
  Leads: 12 standard leads

CPSC2018 data is distributed in WFDB-compatible format.

Expected directory structure after download and extraction:
  data/raw/cpsc2018/
    TrainingSet/
      A0001.mat (or A0001.dat + A0001.hea)
      A0002.mat
      ...
      REFERENCE.csv   ← rhythm labels per record
    RECORDS            ← optional record list

Note on format: CPSC2018 originally provides MATLAB .mat files.
If WFDB .hea/.dat files are available (e.g., via PhysioNet mirror),
this loader uses wfdb.rdrecord directly.
For raw .mat files, the loader falls back to scipy.io.loadmat.

Lead index mapping (standard order for CPSC2018):
  0: I,  1: II,  2: III,  3: aVR,  4: aVL,  5: aVF,
  6: V1, 7: V2,  8: V3,   9: V4,  10: V5,  11: V6
"""

import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CPSC2018Loader:
    """
    Iterates over CPSC2018 recordings, yielding (patient_id, signal, fs).

    Supports both WFDB format (.hea/.dat) and MATLAB format (.mat).

    Parameters
    ----------
    data_dir : Path or str
        Root directory of the CPSC2018 download.
    leads : list of int
        Channel indices to load (default: [0, 1] = Lead I, Lead II).
    max_records : int, optional
        Stop after this many records (for dry runs).
    """

    FS = 500  # Hz

    def __init__(
        self,
        data_dir: Path,
        leads: List[int] = None,
        max_records: Optional[int] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.leads = leads if leads is not None else [0, 1]
        self.max_records = max_records

    def _find_records(self) -> List[Tuple[str, Path, str]]:
        """
        Discover records in data_dir.
        Returns list of (record_id, path_stem, format) where format in {"wfdb", "mat"}.
        """
        results = []

        # Prefer WFDB .hea files
        for hea_file in sorted(self.data_dir.rglob("*.hea")):
            stem = hea_file.with_suffix("")
            record_id = hea_file.stem
            results.append((record_id, stem, "wfdb"))

        if results:
            return results

        # Fallback: MATLAB .mat files
        for mat_file in sorted(self.data_dir.rglob("*.mat")):
            record_id = mat_file.stem
            results.append((record_id, mat_file, "mat"))

        if not results:
            logger.warning(
                "No CPSC2018 records found in %s. "
                "Download from http://2018.icbeb.org/Challenge.html",
                self.data_dir,
            )
        return results

    def patient_ids(self) -> List[str]:
        """
        Return unique patient IDs.
        CPSC2018 uses one record per patient; record ID used as patient ID.
        """
        return [rid for rid, _, _ in self._find_records()]

    def iter_records(
        self,
    ) -> Generator[Tuple[str, np.ndarray, int], None, None]:
        """
        Iterate over all CPSC2018 records.

        Yields
        ------
        (record_id, signal, fs) where fs = 500.
        """
        records = self._find_records()
        count = 0

        for record_id, path, fmt in records:
            if self.max_records is not None and count >= self.max_records:
                break

            if fmt == "wfdb":
                signal_matrix = _load_wfdb(path, record_id)
            else:
                signal_matrix = _load_mat(path, record_id)

            if signal_matrix is None:
                continue

            for lead_idx in self.leads:
                if lead_idx >= signal_matrix.shape[1]:
                    logger.warning(
                        "CPSC2018 %s: lead %d out of range (%d leads)",
                        record_id, lead_idx, signal_matrix.shape[1],
                    )
                    continue
                signal = signal_matrix[:, lead_idx].astype(np.float64)
                signal = _interpolate_nan(signal)
                yield f"cpsc2018_{record_id}_L{lead_idx}", signal, self.FS

            count += 1

        logger.info("CPSC2018: yielded %d records", count)


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

def _load_wfdb(path: Path, record_id: str) -> Optional[np.ndarray]:
    """Load WFDB record, return physical signal matrix (L, n_leads) in mV."""
    import wfdb
    try:
        record = wfdb.rdrecord(str(path))
        return record.p_signal
    except Exception as exc:
        logger.warning("CPSC2018 WFDB load failed (%s): %s", record_id, exc)
        return None


def _load_mat(path: Path, record_id: str) -> Optional[np.ndarray]:
    """
    Load CPSC2018 MATLAB .mat file.

    CPSC2018 .mat files store ECG as a (12, L) int16 array under key 'val'.
    The gain and baseline are stored in the companion .hea file if available;
    otherwise we apply the standard CPSC2018 gain: 1000 ADC units = 1 mV.
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        logger.error("scipy not available; cannot load .mat files")
        return None

    try:
        mat = loadmat(str(path))
        if "val" not in mat:
            logger.warning("CPSC2018 .mat %s: key 'val' not found", record_id)
            return None

        ecg_raw = mat["val"].astype(np.float64)  # (12, L) or (L, 12)

        # Ensure shape (L, 12)
        if ecg_raw.shape[0] == 12 and ecg_raw.shape[1] != 12:
            ecg_raw = ecg_raw.T

        # Standard CPSC2018 gain: 1 mV = 1000 ADC units
        ecg_mv = ecg_raw / 1000.0
        return ecg_mv

    except Exception as exc:
        logger.warning("CPSC2018 .mat load failed (%s): %s", record_id, exc)
        return None


def _interpolate_nan(x: np.ndarray) -> np.ndarray:
    nans = np.isnan(x)
    if not nans.any():
        return x
    idx = np.arange(len(x))
    x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x
