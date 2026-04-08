"""
Training utilities for Phase 2 pre-training.

  cosine_schedule_with_warmup  — cosine LR schedule with linear warmup
  save_checkpoint              — full checkpoint save (model + optimizer)
  rotate_checkpoints           — keep only the N most recent checkpoints
  linear_probe_eval            — lightweight evaluation of representation quality
"""

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float,
    base_lr: float,
) -> LambdaLR:
    """
    Cosine annealing schedule with linear warm-up.

    During warm-up (epochs 0 … warmup_epochs−1) the LR increases linearly
    from 0 to base_lr.  After warm-up it follows a cosine decay down to
    min_lr at total_epochs.

    Returns a LambdaLR scheduler (call scheduler.step() once per epoch).
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / base_lr, cosine_val)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    path: Path,
    extra: dict | None = None,
) -> None:
    """Save a training checkpoint to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch":               epoch,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    logger.info("Checkpoint saved → %s", path)


def rotate_checkpoints(output_dir: Path, keep_last: int, prefix: str = "epoch_") -> None:
    """Delete oldest checkpoints so at most ``keep_last`` files remain."""
    ckpts = sorted(
        output_dir.glob(f"{prefix}*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    while len(ckpts) > keep_last:
        oldest = ckpts.pop(0)
        oldest.unlink()
        logger.debug("Removed old checkpoint: %s", oldest)


# ---------------------------------------------------------------------------
# Linear probe evaluation
# ---------------------------------------------------------------------------

def linear_probe_eval(
    pretrain_model: nn.Module,
    dataloader_val: Any,
    device: torch.device,
    config: Any,
) -> dict[str, float | None]:
    """
    Evaluate representation quality via lightweight linear probes.

    Called every ``N`` epochs during pre-training to track whether the
    representations are becoming semantically meaningful.

    Probe 1 — HR regression
        A single Linear(d_model, 1) is trained for 50 gradient steps on
        frozen z_cls features.  The metric is MAE in the normalised f[0]
        space (monotone with bpm, so lower = better).  Typical target: MAE
        corresponds to < 5 bpm when denormalised.

    Probe 2 — MAE reconstruction PRD
        Masks are generated with the configured mask_ratio (no rhythm-aware
        weighting for consistency across calls).  The decoder reconstructs
        the masked patches and PRD is computed on those patches only.

            PRD = 100 · ||x_rec − x_orig||₂ / ||x_orig||₂

        Computed per-batch and averaged.  Target: PRD < 15 % on masked patches.

    Probe 3 — diagnostic classification: not implemented here (requires PTB-XL
        labels which are not loaded in Phase2Dataset).  Returns None.

    Args:
        pretrain_model: PretrainModel instance (encoder + mae_decoder + mask_token).
        dataloader_val: Validation DataLoader yielding phase2_collate_fn batches.
        device:         torch.device for inference.
        config:         Full OmegaConf pretrain config.

    Returns:
        dict with keys "hr_mae" (float), "prd" (float), "auc" (None).
    """
    from .mae import create_mask, apply_mask_to_tokens

    encoder     = pretrain_model.encoder
    mae_decoder = pretrain_model.mae_decoder
    mask_token  = pretrain_model.mask_token

    d_model     = config.encoder.d_model
    patch_len   = config.data.patch_len
    num_patches = config.data.num_patches
    mask_ratio  = config.mae.mask_ratio

    encoder.eval()
    mae_decoder.eval()

    # ---- Probe 1: collect frozen z_cls representations -------------------
    z_list:  list[torch.Tensor] = []
    hr_list: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader_val:
            x_lr = batch["x_lr"].to(device)
            f    = batch["f"].to(device)
            _, z_cls = encoder(x_lr, f)
            z_list.append(z_cls.cpu())
            hr_list.append(f[:, 0].cpu())   # f[0] is normalised HR

    if not z_list:
        return {"hr_mae": float("inf"), "prd": float("inf"), "auc": None}

    z_all  = torch.cat(z_list,  dim=0)         # (N_val, d_model)
    hr_all = torch.cat(hr_list, dim=0).unsqueeze(1)   # (N_val, 1)

    # Train a fresh linear regression probe
    hr_probe     = nn.Linear(d_model, 1)
    hr_optimizer = torch.optim.Adam(hr_probe.parameters(), lr=1e-3)
    for _ in range(50):
        hr_pred = hr_probe(z_all)
        loss    = nn.functional.mse_loss(hr_pred, hr_all)
        hr_optimizer.zero_grad()
        loss.backward()
        hr_optimizer.step()

    with torch.no_grad():
        hr_pred = hr_probe(z_all)
        hr_mae  = float(nn.functional.l1_loss(hr_pred, hr_all).item())

    # ---- Probe 2: PRD on MAE reconstruction --------------------------------
    rng      = np.random.default_rng(42)
    prd_list: list[float] = []

    with torch.no_grad():
        for batch in dataloader_val:
            x_lr = batch["x_lr"].to(device)
            f    = batch["f"].to(device)
            B    = x_lr.shape[0]

            # Fixed masks (no rhythm-awareness for eval consistency)
            masks = np.stack([
                create_mask(num_patches, mask_ratio, rng=rng)
                for _ in range(B)
            ])
            mask_t = torch.tensor(masks, dtype=torch.bool, device=device)

            z_seq, _ = encoder(x_lr, f)
            z_masked = apply_mask_to_tokens(z_seq, mask_t, mask_token)
            x_rec    = mae_decoder(z_masked)   # (B, num_patches, patch_len)

            # PRD only on masked patches
            x_patches    = x_lr.unfold(-1, patch_len, patch_len)   # (B, num_patches, patch_len)
            x_orig_masked = x_patches[mask_t]   # (N_masked_total, patch_len)
            x_rec_masked  = x_rec[mask_t]

            num = torch.norm(x_rec_masked - x_orig_masked, dim=-1)
            den = torch.norm(x_orig_masked, dim=-1) + 1e-8
            prd_list.append(float((100.0 * num / den).mean().item()))

    prd = float(np.mean(prd_list)) if prd_list else float("inf")

    # Restore training mode
    encoder.train()
    mae_decoder.train()

    return {"hr_mae": hr_mae, "prd": prd, "auc": None}
