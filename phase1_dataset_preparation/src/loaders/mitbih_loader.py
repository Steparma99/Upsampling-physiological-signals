"""
MIT-BIH Arrhythmia Database Loader

Dataset: MIT-BIH Arrhythmia Database (PhysioNet)
  URL:  https://physionet.org/content/mitdb/1.0.0/
  fs:   360 Hz
  N:    48 records (30 min each) from 47 patients
  Leads: 2 (MLII + V1 or V2 or V4 or V5)

Since fs = 360 Hz < fs_hr = 500 Hz, this dataset cannot provide real
500 Hz ground truth. Role: unsupervised pre-training only.

In unsupervised mode, the pipeline creates pseudo-pairs:
  x_pseudo_HR at 360 Hz (the original signal)
  x_pseudo_LR at 72 Hz  (360/5)

These pseudo-pairs are stored in a separate HDF5 file:
  data/processed/mitbih_pretrain.h5

Expected directory structure:
  data/raw/mitbih/
    100.dat, 100.hea, 100.atr
    101.dat, 101.hea, 101.atr
    ...
    234.dat, 234.hea, 234.atr
    RECORDS

The 48 records are: 100–124, 200–234 (some numbers skipped).
"""

import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import wfdb

logger = logging.getLogger(__name__)

# All 48 MIT-BIH record IDs
MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221", "222", "223",
    "228", "230", "231", "232", "233", "234",
]

# Patient mapping: MIT-BIH does not provide explicit patient IDs in the header.
# We use the record number as a surrogate patient ID (each record = one patient).
# Records 201 and 202 are the same patient — deduplicate in split if needed.
SAME_PATIENT_GROUPS = [("201", "202")]


class MITBIHLoader:
    """
    Iterates over MIT-BIH recordings for unsupervised pre-training pairs.

    Since fs = 360 Hz < fs_hr = 500 Hz, yields (patient_id, signal, 360).
    The DatasetBuilder uses role="pretraining" to handle these differently:
      - Creates pseudo-HR/LR pairs at 360 Hz / 72 Hz (×5)
      - Stored separately (mitbih_pretrain.h5), not mixed with supervised splits

    Parameters
    ----------
    data_dir : Path or str
        Root directory containing MIT-BIH .dat/.hea files.
    leads : list of int
        Channel indices to load (default: [0] = MLII only).
    max_records : int, optional
        Stop after this many records (for dry runs).
    """

    FS = 360       # Hz (actual sampling frequency)
    FS_LR = 72     # Hz (360 / 5 for pseudo-pair generation)

    def __init__(
        self,
        data_dir: Path,
        leads: List[int] = None,
        max_records: Optional[int] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.leads = leads if leads is not None else [0]
        self.max_records = max_records

    def _find_records(self) -> List[str]:
        """Return list of record IDs that exist in data_dir."""
        present = []
        for rid in MITBIH_RECORDS:
            hea = self.data_dir / f"{rid}.hea"
            if hea.exists():
                present.append(rid)
        if not present:
            logger.warning(
                "No MIT-BIH records found in %s. "
                "Download with: python -c \"import wfdb; "
                "wfdb.dl_database('mitdb', '%s')\"",
                self.data_dir, self.data_dir,
            )
        return present

    def patient_ids(self) -> List[str]:
        """Return unique pseudo-patient IDs (one per record for MIT-BIH)."""
        return self._find_records()

    def iter_records(
        self,
    ) -> Generator[Tuple[str, np.ndarray, int], None, None]:
        """
        Iterate over all MIT-BIH records.

        Yields
        ------
        (record_id, signal, fs) where fs = 360.
        """
        record_ids = self._find_records()
        count = 0

        for rid in record_ids:
            if self.max_records is not None and count >= self.max_records:
                break

            record_path = self.data_dir / rid
            try:
                record = wfdb.rdrecord(str(record_path))
            except Exception as exc:
                logger.warning("Failed to load MIT-BIH %s: %s", rid, exc)
                continue

            signal_matrix = record.p_signal  # (L, 2), mV
            patient_id = rid  # surrogate

            for lead_idx in self.leads:
                if lead_idx >= signal_matrix.shape[1]:
                    logger.warning("MIT-BIH %s: lead %d out of range", rid, lead_idx)
                    continue
                signal = signal_matrix[:, lead_idx].astype(np.float64)
                signal = _interpolate_nan(signal)
                record_id = f"mitbih_{rid}_L{lead_idx}"
                yield record_id, signal, self.FS

            count += 1

        logger.info("MIT-BIH: yielded %d records", count)


def _interpolate_nan(x: np.ndarray) -> np.ndarray:
    nans = np.isnan(x)
    if not nans.any():
        return x
    idx = np.arange(len(x))
    x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x
