"""
PTB Diagnostic ECG Database Loader

Dataset: PTB Diagnostic ECG Database (PhysioNet)
  URL:  https://physionet.org/content/ptbdb/1.0.0/
  fs:   1000 Hz
  N:    549 records from 290 patients
  Leads: 15 (12 standard + Frank leads VX, VY, VZ)

Since fs = 1000 Hz > fs_hr = 500 Hz, recordings are used as high-fidelity
ground truth by resampling 1000 → 500 Hz in DatasetBuilder.

Expected directory structure:
  data/raw/ptb_diagnostic/
    patient001/
      s0010lre.dat
      s0010lre.hea
      ...
    patient002/
      ...
    RECORDS        ← list of all record paths (one per line)

Lead index mapping (standard WFDB order for PTB):
  0: I,   1: II,  2: III,  3: aVR, 4: aVL,  5: aVF,
  6: V1,  7: V2,  8: V3,   9: V4, 10: V5,  11: V6,
 12: VX, 13: VY, 14: VZ
"""

import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import wfdb

logger = logging.getLogger(__name__)


class PTBDiagnosticLoader:
    """
    Iterates over PTB Diagnostic ECG recordings, yielding (patient_id, signal, fs).

    The DatasetBuilder will resample from 1000 Hz → 500 Hz.

    Parameters
    ----------
    data_dir : Path or str
        Root directory of the PTB Diagnostic download.
    leads : list of int
        WFDB channel indices to load (default: [0, 1] = Lead I, Lead II).
    max_records : int, optional
        Stop after this many records (for dry runs).
    """

    FS = 1000  # Hz — will be resampled to 500 Hz by DatasetBuilder

    def __init__(
        self,
        data_dir: Path,
        leads: List[int] = None,
        max_records: Optional[int] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.leads = leads if leads is not None else [0, 1]
        self.max_records = max_records

    def _find_records(self) -> List[Tuple[str, Path]]:
        """
        Discover all WFDB records.
        Returns list of (patient_id, record_stem_path).
        """
        records_file = self.data_dir / "RECORDS"
        if records_file.exists():
            with open(records_file) as f:
                rel_paths = [line.strip() for line in f if line.strip()]
            result = []
            for rel in rel_paths:
                # rel is like "patient001/s0010lre"
                parts = Path(rel).parts
                patient_id = parts[0] if len(parts) > 1 else "unknown"
                full_path = self.data_dir / rel
                result.append((patient_id, full_path))
            return result

        # Fallback: scan directory tree for .hea files
        result = []
        for hea_file in sorted(self.data_dir.rglob("*.hea")):
            patient_id = hea_file.parent.name
            result.append((patient_id, hea_file.with_suffix("")))
        return result

    def patient_ids(self) -> List[str]:
        """Return sorted list of unique patient ID strings."""
        records = self._find_records()
        return sorted(set(pid for pid, _ in records))

    def iter_records(
        self,
    ) -> Generator[Tuple[str, np.ndarray, int], None, None]:
        """
        Iterate over all PTB Diagnostic records.

        Yields
        ------
        (record_id, signal, fs) where:
          record_id : str  — "{patient_id}_{record_stem}_L{lead}"
          signal    : np.ndarray  — 1-D float64 (mV)
          fs        : int  — 1000
        """
        records = self._find_records()
        count = 0

        for patient_id, record_path in records:
            if self.max_records is not None and count >= self.max_records:
                break

            try:
                record = wfdb.rdrecord(str(record_path))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", record_path, exc)
                continue

            signal_matrix = record.p_signal  # (L, n_leads), mV
            record_stem = record_path.stem

            for lead_idx in self.leads:
                if lead_idx >= signal_matrix.shape[1]:
                    logger.warning(
                        "%s/%s: lead %d out of range",
                        patient_id, record_stem, lead_idx,
                    )
                    continue
                signal = signal_matrix[:, lead_idx].astype(np.float64)
                signal = _interpolate_nan(signal)
                record_id = f"{patient_id}_{record_stem}_L{lead_idx}"
                yield record_id, signal, self.FS

            count += 1

        logger.info("PTB Diagnostic: yielded %d records", count)


def _interpolate_nan(x: np.ndarray) -> np.ndarray:
    nans = np.isnan(x)
    if not nans.any():
        return x
    idx = np.arange(len(x))
    x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x
