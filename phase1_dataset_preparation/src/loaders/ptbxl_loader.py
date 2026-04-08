"""
PTB-XL Dataset Loader

Dataset: PTB-XL (PhysioNet)
  URL:  https://physionet.org/content/ptb-xl/1.0.3/
  fs:   500 Hz
  N:    21 799 10-second recordings from 18 885 patients
  Leads: 12-lead standard ECG

Expected directory structure after download:
  data/raw/ptbxl/
    ptbxl_database.csv       ← metadata (patient IDs, diagnoses, etc.)
    records500/              ← 500 Hz WFDB records
      00000/
        00001_hr.dat
        00001_hr.hea
        ...
    records100/              ← 100 Hz WFDB records (not used here)

Lead index mapping (WFDB channel order for PTB-XL):
  0: I,  1: II,  2: III,  3: aVR,  4: aVL,  5: aVF,
  6: V1, 7: V2,  8: V3,   9: V4,  10: V5,  11: V6
"""

import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb

logger = logging.getLogger(__name__)

# Nominal ADC gain for PTB-XL (mV per ADC unit, from WFDB header)
# wfdb.rdrecord already applies gain and returns physical units (mV)


class PTBXLLoader:
    """
    Iterates over PTB-XL recordings, yielding (patient_id, signal, fs).

    Parameters
    ----------
    data_dir : Path or str
        Root directory of the PTB-XL download.
    leads : list of int
        WFDB channel indices to load (default: [0, 1] = Lead I, Lead II).
    max_records : int, optional
        If set, stop after this many records (for dry runs / testing).
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
        self._metadata: Optional[pd.DataFrame] = None

    def _load_metadata(self) -> pd.DataFrame:
        csv_path = self.data_dir / "ptbxl_database.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"PTB-XL metadata not found: {csv_path}\n"
                "Download from https://physionet.org/content/ptb-xl/1.0.3/"
            )
        df = pd.read_csv(csv_path, index_col="ecg_id")
        logger.info("PTB-XL metadata: %d records, %d unique patients",
                    len(df), df["patient_id"].nunique())
        return df

    @property
    def metadata(self) -> pd.DataFrame:
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return self._metadata

    def patient_ids(self) -> List[str]:
        """Return sorted list of unique patient ID strings (integer format, e.g. '15709')."""
        return sorted(
            self.metadata["patient_id"].dropna().astype(int).astype(str).unique().tolist()
        )

    def records_for_patient(self, patient_id: str) -> List[str]:
        """Return list of filename stems for all records of a given patient."""
        pid = int(patient_id)
        sub = self.metadata[self.metadata["patient_id"] == pid]
        return sub["filename_hr"].tolist()

    def _record_path(self, filename_hr: str) -> Path:
        """Convert relative filename_hr from CSV to absolute path."""
        # filename_hr is like "records500/00000/00001_hr"
        return self.data_dir / filename_hr

    def iter_records(
        self,
    ) -> Generator[Tuple[str, np.ndarray, int], None, None]:
        """
        Iterate over all records.

        Yields
        ------
        (record_id, signal, fs) where:
          record_id : str  — "{patient_id}_{ecg_id}"
          signal    : np.ndarray shape (L,)  — single-channel float64 (mV)
          fs        : int  — 500
        """
        count = 0
        for ecg_id, row in self.metadata.iterrows():
            if self.max_records is not None and count >= self.max_records:
                break

            record_path = self._record_path(str(row["filename_hr"]))
            try:
                record = wfdb.rdrecord(str(record_path))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", record_path, exc)
                continue

            signal_matrix = record.p_signal  # shape (L, n_leads), physical units (mV)
            patient_id = str(int(row["patient_id"]))
            record_id = f"{patient_id}_{ecg_id}"

            for lead_idx in self.leads:
                if lead_idx >= signal_matrix.shape[1]:
                    logger.warning(
                        "%s: lead index %d out of range (%d leads)",
                        record_id, lead_idx, signal_matrix.shape[1],
                    )
                    continue
                signal = signal_matrix[:, lead_idx].astype(np.float64)
                # Replace NaN (missing samples) with linear interpolation
                signal = _interpolate_nan(signal)
                yield f"{record_id}_L{lead_idx}", signal, self.FS

            count += 1

        logger.info("PTB-XL: yielded %d records", count)


def _interpolate_nan(x: np.ndarray) -> np.ndarray:
    """Replace NaN values with linear interpolation."""
    nans = np.isnan(x)
    if not nans.any():
        return x
    idx = np.arange(len(x))
    x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x
