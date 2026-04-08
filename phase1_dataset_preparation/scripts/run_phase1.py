"""
Phase 1 — Main Pipeline Entry Point

Builds the supervised (x_LR, x_HR) dataset for ECG upsampling.

Usage
-----
From phase1_dataset_preparation/:

    python scripts/run_phase1.py --config config/config.yaml

Options:
    --config CONFIG        Path to YAML config (default: config/config.yaml)
    --datasets DS1,DS2     Comma-separated subset of datasets to process
                           Choices: ptbxl, ptb_diagnostic, mitbih, cpsc2018
                           (default: all enabled in config)
    --output_dir DIR       Override output directory from config
    --dry_run              Process only the first 10 records per dataset

Output:
    data/processed/train.h5
    data/processed/val.h5
    data/processed/test.h5
    data/processed/pipeline_stats.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure Unicode log output works on Windows terminals (cp1252 → utf-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import yaml

# _PHASE1_DIR = phase1_dataset_preparation/
# Ensures imports work AND that relative paths (config, data/) resolve correctly
# regardless of the CWD the user launches the script from.
_PHASE1_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PHASE1_DIR))


def _abs(p: Path) -> Path:
    """Return p unchanged if absolute; otherwise resolve relative to _PHASE1_DIR."""
    return p if p.is_absolute() else _PHASE1_DIR / p

from src.dataset_builder import DatasetBuilder, split_patients
from src.loaders import (
    CPSC2018Loader,
    MITBIHLoader,
    PTBDiagnosticLoader,
    PTBXLLoader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_phase1")


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------

def build_loader(name: str, cfg: dict, data_root: Path, max_records: Optional[int]):
    """Instantiate the correct loader for the given dataset name."""
    ds_cfg = cfg["datasets"][name]
    # Allow dataset-specific absolute path override (e.g. external drive).
    override = ds_cfg.get("data_dir")
    data_dir = Path(override) if override else data_root / name
    leads = ds_cfg.get("leads", [0, 1])

    loaders = {
        "ptbxl": PTBXLLoader,
        "ptb_diagnostic": PTBDiagnosticLoader,
        "mitbih": MITBIHLoader,
        "cpsc2018": CPSC2018Loader,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}")

    return loaders[name](data_dir=data_dir, leads=leads, max_records=max_records)


# ---------------------------------------------------------------------------
# Process one supervised dataset (role = "supervised")
# ---------------------------------------------------------------------------

def process_supervised_dataset(
    name: str,
    loader,
    builder: DatasetBuilder,
    split_cfg: dict,
) -> Dict[str, List[dict]]:
    """
    Process a supervised dataset:
      1. Enumerate all patient IDs
      2. Split patients → train / val / test
      3. Iterate records, process each through the builder
      4. Return windows keyed by split

    Returns
    -------
    {"train": [...], "val": [...], "test": [...]}
    """
    all_patient_ids = loader.patient_ids()
    if not all_patient_ids:
        logger.warning("%s: no patients found — skipping", name)
        return {"train": [], "val": [], "test": []}

    train_ids, val_ids, test_ids = split_patients(
        all_patient_ids,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        seed=split_cfg["random_seed"],
    )
    patient_to_split = (
        {pid: "train" for pid in train_ids}
        | {pid: "val" for pid in val_ids}
        | {pid: "test" for pid in test_ids}
    )

    logger.info(
        "%s: %d patients → train=%d  val=%d  test=%d",
        name, len(all_patient_ids), len(train_ids), len(val_ids), len(test_ids),
    )

    split_windows: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    ds_cfg = builder.cfg["datasets"][name]
    fs_source = ds_cfg["fs"]

    for record_id, signal, fs in loader.iter_records():
        # Derive patient_id from record_id:
        # Format varies by dataset; use first component before "_"
        patient_id = record_id.split("_")[0]
        # PTB-XL: record_id = "{patient_id}_{ecg_id}_L{lead}" → patient_id correct
        # PTB Diag: record_id = "{patient_id}_{stem}_L{lead}" → patient_id is "patientXXX"
        # CPSC2018: record_id = "cpsc2018_{record}_L{lead}" → use full record without lead
        if name == "cpsc2018":
            # patient_id = record without lead suffix (1 patient per record)
            patient_id = "_".join(record_id.split("_")[1:-1])  # "A0001"
        elif name == "ptb_diagnostic":
            patient_id = record_id.split("_")[0]  # "patientXXX"

        split_name = patient_to_split.get(str(patient_id))
        if split_name is None:
            # Patient not in any split (shouldn't happen)
            logger.debug("Patient %s not in any split — skipping %s", patient_id, record_id)
            continue

        windows = builder.process_recording(signal, fs_source, record_id)
        split_windows[split_name].extend(windows)

        logger.debug(
            "%s / %s → %d windows accepted",
            name, record_id, len(windows),
        )

    for split_name, wlist in split_windows.items():
        logger.info("%s / %s: %d windows", name, split_name, len(wlist))

    return split_windows


# ---------------------------------------------------------------------------
# Process MIT-BIH (role = "pretraining")
# ---------------------------------------------------------------------------

def process_pretraining_dataset(
    loader: MITBIHLoader,
    cfg: dict,
    output_dir: Path,
) -> None:
    """
    Build unsupervised pseudo-pairs for MIT-BIH (360 Hz → 72 Hz, ×5).

    A separate DatasetBuilder is constructed with fs_hr=360 / fs_lr=72
    to match the MIT-BIH sampling rate.
    No train/val/test split is performed — all data goes into one file.
    """
    import copy

    pretrain_cfg = copy.deepcopy(cfg)
    pretrain_cfg["fs_hr"] = MITBIHLoader.FS        # 360
    pretrain_cfg["fs_lr"] = MITBIHLoader.FS_LR     # 72
    pretrain_cfg["upsample_factor"] = 5

    pretrain_builder = DatasetBuilder(pretrain_cfg)
    all_windows: List[dict] = []

    for record_id, signal, fs in loader.iter_records():
        windows = pretrain_builder.process_recording(signal, fs, record_id)
        all_windows.extend(windows)

    logger.info("MIT-BIH pretraining: %d windows total", len(all_windows))
    pretrain_builder.save_split(all_windows, "mitbih_pretrain", output_dir)
    pretrain_builder.log_stats()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 — Build ECG upsampling dataset"
    )
    parser.add_argument(
        "--config",
        default=str(_PHASE1_DIR / "config" / "config.yaml"),
        help="Path to YAML config (default: <phase1_dir>/config/config.yaml)",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated datasets to process (default: all enabled)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Process only first 10 records per dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve output directory (relative paths are anchored to _PHASE1_DIR)
    output_dir = _abs(Path(args.output_dir) if args.output_dir else Path(cfg["paths"]["processed_data"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = _abs(Path(cfg["paths"]["raw_data"]))

    # Decide which datasets to process
    enabled = {
        name for name, ds in cfg["datasets"].items() if ds.get("enabled", False)
    }
    if args.datasets:
        requested = set(args.datasets.split(","))
        enabled = enabled & requested
        unknown = requested - set(cfg["datasets"].keys())
        if unknown:
            logger.error("Unknown datasets: %s", unknown)
            sys.exit(1)

    logger.info("Datasets to process: %s", sorted(enabled))
    max_records = 10 if args.dry_run else None

    builder = DatasetBuilder(cfg)
    all_splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}

    # ---- Supervised datasets ----
    supervised = [n for n in enabled if cfg["datasets"][n].get("role") == "supervised"]
    for name in sorted(supervised):
        logger.info("=" * 55)
        logger.info("Processing dataset: %s", name)
        loader = build_loader(name, cfg, data_root, max_records)
        split_windows = process_supervised_dataset(
            name, loader, builder, cfg["split"]
        )
        for split_name, windows in split_windows.items():
            all_splits[split_name].extend(windows)

    # ---- Save supervised splits ----
    logger.info("=" * 55)
    logger.info("Saving supervised splits …")
    saved_paths = {}
    for split_name, windows in all_splits.items():
        path = builder.save_split(windows, split_name, output_dir)
        saved_paths[split_name] = str(path)

    # ---- MIT-BIH unsupervised pretraining ----
    if "mitbih" in enabled and cfg["datasets"]["mitbih"].get("role") == "pretraining":
        logger.info("=" * 55)
        logger.info("Processing MIT-BIH (pretraining) …")
        mitbih_loader = build_loader("mitbih", cfg, data_root, max_records)
        process_pretraining_dataset(mitbih_loader, cfg, output_dir)

    # ---- Final stats ----
    builder.log_stats()

    # ---- Save pipeline stats to JSON ----
    stats_path = output_dir / "pipeline_stats.json"
    stats_out = {
        "config": str(config_path.resolve()),
        "dry_run": args.dry_run,
        "datasets_processed": sorted(enabled),
        "output_files": saved_paths,
        "window_counts": {k: len(v) for k, v in all_splits.items()},
        "pipeline_stats": builder.stats,
    }
    with open(stats_path, "w") as f:
        json.dump(stats_out, f, indent=2)
    logger.info("Pipeline stats → %s", stats_path)
    logger.info("Phase 1 complete.")


if __name__ == "__main__":
    main()
