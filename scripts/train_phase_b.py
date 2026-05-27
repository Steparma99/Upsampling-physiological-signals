"""
train_phase_b.py — Entry point for Phase B supervised fine-tuning.

This script is intentionally thin: it wires together the config, data, models,
and W&B run, then delegates all training logic to src/training/phase_b.py.

The encoder weights from Phase A are loaded via the internal _load_encoder()
function (strict=False, with missing/unexpected key warnings).

Usage
-----
    python scripts/train_phase_b.py \\
        --data /path/to/ptbxl \\
        --encoder_ckpt runs/phase_a/encoder.pt \\
        --save_dir runs/phase_b

    # Resume an interrupted Phase B run:
    python scripts/train_phase_b.py \\
        --data /path/to/ptbxl \\
        --encoder_ckpt runs/phase_a/encoder.pt \\
        --save_dir runs/phase_b \\
        --resume runs/phase_b/last.pt

    # Disable W&B:
    python scripts/train_phase_b.py \\
        --data /path/to/ptbxl \\
        --encoder_ckpt runs/phase_a/encoder.pt \\
        --save_dir runs/phase_b \\
        --no_wandb
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# ── Make project root importable regardless of working directory ──────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import PTBXLDataset, get_dataloader
from src.training.phase_b import train_phase_b

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =============================================================================
# Argument parsing
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase B: supervised fine-tuning with U-Net decoder + PatchGAN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data", type=str, required=True,
        help="Path to PTB-XL root directory (must contain ptbxl_database.csv).",
    )
    p.add_argument(
        "--encoder_ckpt", type=str, required=True,
        help="Path to encoder.pt from train_phase_a.py (required).",
    )
    p.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to OmegaConf config YAML.",
    )
    p.add_argument(
        "--save_dir", type=str, default="runs/phase_b",
        help="Directory where checkpoints and final models will be saved.",
    )
    p.add_argument(
        "--gpu", type=int, default=0,
        help="CUDA device index.  Use -1 to force CPU.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed for reproducibility.",
    )
    p.add_argument(
        "--resume", type=str, default=None,
        help="Path to last.pt checkpoint to resume Phase B training.",
    )
    p.add_argument(
        "--run_name", type=str, default=None,
        help="W&B run name (auto-generated if omitted).",
    )
    p.add_argument(
        "--wandb_project", type=str, default="ecg-superresolution",
        help="W&B project name.",
    )
    p.add_argument(
        "--wandb_entity", type=str, default=None,
        help="W&B entity (team or user). Uses your default entity if omitted.",
    )
    p.add_argument(
        "--no_wandb", action="store_true",
        help="Disable W&B logging even if the library is installed.",
    )
    return p.parse_args()


# =============================================================================
# Reproducibility
# =============================================================================

def _set_seed(seed: int) -> None:
    """Fix random seeds for Python, NumPy, PyTorch (CPU + GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %d", seed)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = _parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    logger.info("Loaded config from %s", cfg_path)

    # ── Encoder checkpoint validation ────────────────────────────────────────
    encoder_ckpt = Path(args.encoder_ckpt)
    if not encoder_ckpt.exists():
        raise FileNotFoundError(
            f"Encoder checkpoint not found: {encoder_ckpt}\n"
            "Run train_phase_a.py first to generate encoder.pt."
        )

    # ── Seed ─────────────────────────────────────────────────────────────────
    _set_seed(args.seed)

    # ── Device ───────────────────────────────────────────────────────────────
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # ── Datasets and DataLoaders ──────────────────────────────────────────────
    data_root = Path(args.data)
    logger.info("Loading PTB-XL from %s", data_root)

    train_ds = PTBXLDataset(
        root=data_root,
        split="train",
        channel=cfg.data.channel,
        p_drop_meta=cfg.data.p_drop_meta,
        seed=args.seed,
    )
    val_ds = PTBXLDataset(
        root=data_root,
        split="val",
        channel=cfg.data.channel,
        p_drop_meta=0.0,
        seed=args.seed + 1,
    )
    logger.info("Train: %d records | Val: %d records", len(train_ds), len(val_ds))

    train_loader = get_dataloader(
        train_ds,
        batch_size=cfg.phase_b.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=cfg.data.drop_last,
    )
    val_loader = get_dataloader(
        val_ds,
        batch_size=cfg.phase_b.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=False,
    )

    # ── W&B ──────────────────────────────────────────────────────────────────
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                resume="allow",
            )
            logger.info("W&B run initialised: %s", wandb.run.name)
        except Exception as exc:
            logger.warning("W&B init failed (%s) — continuing without logging.", exc)

    # ── Train ─────────────────────────────────────────────────────────────────
    # train_phase_b instantiates the models internally, loads encoder weights,
    # and runs the full fine-tuning loop.
    train_phase_b(
        cfg=cfg,
        encoder_ckpt=encoder_ckpt,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=args.save_dir,
        device=device,
        resume_ckpt=args.resume,
    )

    # ── Finish W&B run ────────────────────────────────────────────────────────
    if not args.no_wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

    logger.info(
        "Phase B training complete. encoder_finetuned.pt and decoder.pt saved in %s",
        args.save_dir,
    )


if __name__ == "__main__":
    main()
