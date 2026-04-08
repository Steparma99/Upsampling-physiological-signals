"""
Entry point for Phase 2 self-supervised pre-training.

Usage:
    python train_phase2.py \\
        --config  configs/pretrain_config.yaml \\
        --data_dir  phase1_dataset_preparation/data/processed \\
        [--resume checkpoints/phase2/epoch_0049.pt]

Expected files in data_dir:
    train.h5        — Phase 1 training split
    val.h5          — Phase 1 validation split
    f_stats.npz     — f normalisation stats (computed automatically if absent)

Outputs (written to config.checkpoint.output_dir):
    epoch_XXXX.pt          — rolling checkpoints (keep_last N)
    encoder_phase3.pt      — encoder-only weights for Phase 3

Sanity check (run before a full training run):
    python -c "
    import torch
    from omegaconf import OmegaConf
    from phase2.encoder import HybridEncoder
    cfg = OmegaConf.load('configs/pretrain_config.yaml')
    model = HybridEncoder(cfg)
    x = torch.randn(2, 1000)
    f = torch.randn(2, 13)
    z_seq, z_cls = model(x, f)
    assert z_seq.shape == (2, 21, 256), z_seq.shape
    assert z_cls.shape == (2, 256),     z_cls.shape
    print('Shapes OK')
    "
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from phase2.augmentations import ECGAugmentations
from phase2.dataset import Phase2Dataset, phase2_collate_fn
from phase2.pretrain import (
    PretrainModel,
    export_encoder_for_phase3,
    save_pretrain_checkpoint,
    train_epoch,
)
from phase2.utils import (
    cosine_schedule_with_warmup,
    linear_probe_eval,
    rotate_checkpoints,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 ECG pre-training")
    p.add_argument("--config",   default="configs/pretrain_config.yaml",
                   help="Path to pretrain_config.yaml")
    p.add_argument("--data_dir", required=True,
                   help="Directory containing train.h5, val.h5, f_stats.npz")
    p.add_argument("--resume",   default=None,
                   help="Path to a checkpoint to resume from")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    config   = OmegaConf.load(args.config)
    data_dir = Path(args.data_dir)
    out_dir  = Path(config.checkpoint.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA not available — training on CPU will be extremely slow. "
            "Set amp: false in the config when running without a GPU."
        )

    # ------------------------------------------------------------------ #
    # f_stats: compute from training set if absent                        #
    # ------------------------------------------------------------------ #
    f_stats_path = data_dir / "f_stats.npz"
    if not f_stats_path.exists():
        logger.info("f_stats.npz not found — computing from training split …")
        Phase2Dataset.compute_and_save_f_stats(
            train_h5_path=data_dir / "train.h5",
            output_path=f_stats_path,
        )

    # ------------------------------------------------------------------ #
    # Datasets and DataLoaders                                            #
    # ------------------------------------------------------------------ #
    train_dataset = Phase2Dataset(
        data_dir / "train.h5",
        f_stats_path=f_stats_path,
        source="native",
    )
    val_dataset = Phase2Dataset(
        data_dir / "val.h5",
        f_stats_path=f_stats_path,
        source="native",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.contrastive.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        drop_last=True,             # keep batch sizes consistent for BatchNorm1d
        collate_fn=phase2_collate_fn,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.contrastive.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
        collate_fn=phase2_collate_fn,
        persistent_workers=True,
    )

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    model = PretrainModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_enc    = sum(p.numel() for p in model.encoder.parameters())
    logger.info("Total parameters: %d  (encoder: %d)", n_params, n_enc)

    # ------------------------------------------------------------------ #
    # Optimiser, AMP scaler, LR scheduler                                 #
    # ------------------------------------------------------------------ #
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        betas=tuple(config.optimizer.betas),
        eps=config.optimizer.eps,
    )

    amp_enabled = config.amp and device.type == "cuda"
    scaler      = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    scheduler = cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=config.scheduler.warmup_epochs,
        total_epochs=config.scheduler.total_epochs,
        min_lr=config.scheduler.min_lr,
        base_lr=config.optimizer.lr,
    )

    # ------------------------------------------------------------------ #
    # Resume                                                               #
    # ------------------------------------------------------------------ #
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["full_model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        # Fast-forward scheduler to the correct epoch
        for _ in range(start_epoch):
            scheduler.step()
        logger.info("Resumed from checkpoint %s (epoch %d)", args.resume, start_epoch)

    augmentations = ECGAugmentations(p=0.5)

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    for epoch in range(start_epoch, config.scheduler.total_epochs):
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            epoch=epoch,
            augmentations=augmentations,
            device=device,
        )
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %04d/%04d | loss=%.4f | lr=%.2e",
            epoch, config.scheduler.total_epochs - 1, train_loss, lr_now,
        )

        # Probe evaluation every 10 epochs
        if (epoch + 1) % 10 == 0:
            metrics = linear_probe_eval(model, val_loader, device, config)
            logger.info(
                "Probe [epoch %04d] | HR MAE=%.4f (norm) | PRD=%.2f%%",
                epoch, metrics["hr_mae"], metrics["prd"],
            )

        # Checkpoint
        if (epoch + 1) % config.checkpoint.save_every == 0:
            ckpt_path = out_dir / f"epoch_{epoch:04d}.pt"
            save_pretrain_checkpoint(model, optimizer, epoch, ckpt_path)
            rotate_checkpoints(out_dir, config.checkpoint.keep_last)

    # ------------------------------------------------------------------ #
    # Export encoder for Phase 3                                           #
    # ------------------------------------------------------------------ #
    export_encoder_for_phase3(model, out_dir / "encoder_phase3.pt")
    logger.info("Pre-training complete.")


if __name__ == "__main__":
    main()
