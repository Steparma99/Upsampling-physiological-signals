"""
phase_a.py — Self-supervised pre-training loop (Phase A) for ECG super-resolution.

Combines SimCLR (NT-Xent contrastive loss) and MAE (masked auto-encoder) in a
single training loop.  After training, only the encoder weights are saved as
encoder.pt; the projection head and MAE decoder are discarded (per CLAUDE.md).

Training objective:
    L_pretrain = λ_ntxent × L_NT-Xent + λ_mae × L_MAE

Entry point:
    train_phase_a(encoder, proj_head, mae_decoder,
                  train_loader, val_loader,
                  cfg, save_dir, device, resume_ckpt=None)

SimCLR path (per batch):
  1. Apply ECGAugmentation → two independently augmented views (view_a, view_b)
  2. Full encoder forward for each view → z_cls_a, z_cls_b
  3. ProjectionHead → h_a, h_b (raw, unnormalised)
  4. NTXentLoss(h_a, h_b) — L2 norm applied inside the loss (CLAUDE.md #3)

MAE path (per batch, original non-augmented input):
  1. feature_mlp → f_emb (CLS token)  [never masked — CLAUDE.md #2]
  2. cnn_encoder → CNN patches (B, P, d_model)
  3. Rhythm-aware mask: QRS patches get higher probability (qrs_masking_boost)
  4. MAEDecoder.prepare_masked_input → masked patches
  5. Concatenate [f_emb, masked_patches] → Transformer → z_seq
  6. MAEDecoder → patch_preds (B, P, patch_len)
  7. MAELoss: per-patch z-score MSE using TARGET stats (CLAUDE.md #4) + FFT
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from scipy.signal import argrelmin
from torch.utils.data import DataLoader

from src.data.augmentations import ECGAugmentation
from src.data.preprocessing import FS_LR, L_LR
from src.losses.contrastive import NTXentLoss
from src.losses.mae_loss import MAELoss
from src.models.encoder import ECGEncoder
from src.models.mae_decoder import MAEDecoder
from src.models.projection_head import ProjectionHead

logger = logging.getLogger(__name__)

try:
    import wandb as _wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


# =============================================================================
# Private helpers
# =============================================================================

def _wandb_log(metrics: dict[str, Any], step: int) -> None:
    """Log to W&B only when a run is active; silently skips otherwise."""
    if _HAS_WANDB and _wandb.run is not None:
        _wandb.log(metrics, step=step)


def _approx_r_peaks_from_dmap(
    d_map: np.ndarray,
    order: int = 20,
    threshold: float = 0.25,
) -> np.ndarray:
    """Recover approximate R-peak sample indices from a pre-computed d_map.

    d_map[m] = min_k|m − r_k| / mean_RR, so R-peaks sit at positions m where
    d_map is a local minimum near zero.  This avoids re-running the full
    R-peak detector in the training loop.

    Args:
        d_map:     (L_LR,) array of normalised R-distances.
        order:     argrelmin neighbourhood half-width in samples.
        threshold: d_map value below which a local minimum is accepted as a peak.

    Returns:
        1-D int array of sample indices, dtype np.intp.
    """
    local_mins = argrelmin(d_map, order=order)[0]
    r_peaks = local_mins[d_map[local_mins] < threshold]
    if len(r_peaks) == 0:
        # Fallback: at least return the global minimum so augmentation doesn't crash
        r_peaks = np.array([int(np.argmin(d_map))], dtype=np.intp)
    return r_peaks.astype(np.intp)


def _augment_batch(
    x_lr: torch.Tensor,      # (B, L_LR)  z-score normalised, on device
    d_map: torch.Tensor,     # (B, L_LR)  on device
    mu_w: torch.Tensor,      # (B,)       per-sample normalisation mean
    sigma_w: torch.Tensor,   # (B,)       per-sample normalisation std
    augmentor: ECGAugmentation,
    strength: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produce two independently augmented views for the entire batch.

    Steps per sample:
      1. Denormalise x_LR → raw signal (ECGAugmentation needs physical units).
      2. Recover approximate R-peaks from d_map (avoids re-detecting on raw signal).
      3. Call ECGAugmentation → view_a, view_b, d_map_a, d_map_b.
      4. Re-normalise each view independently with its own z-score parameters.

    All numpy operations run on CPU; results are moved to `device`.

    Returns:
        view_a:  (B, L_LR) normalised augmented view A
        dmap_a:  (B, L_LR) R-distance map for view A
        view_b:  (B, L_LR) normalised augmented view B
        dmap_b:  (B, L_LR) R-distance map for view B
    """
    B, L = x_lr.shape

    x_np  = x_lr.cpu().numpy()
    dm_np = d_map.cpu().numpy()
    mu_np = mu_w.cpu().numpy()
    sig_np = sigma_w.cpu().numpy()

    va_buf  = np.empty((B, L), dtype=np.float32)
    vb_buf  = np.empty((B, L), dtype=np.float32)
    dma_buf = np.empty((B, L), dtype=np.float32)
    dmb_buf = np.empty((B, L), dtype=np.float32)

    for b in range(B):
        # Denormalise
        x_raw = (x_np[b] * float(sig_np[b]) + float(mu_np[b])).astype(np.float64)
        r_peaks = _approx_r_peaks_from_dmap(dm_np[b])

        view_a, view_b, dmap_a, dmap_b, _, _ = augmentor(x_raw, r_peaks, strength=strength)

        # Re-normalise each view with its own z-score parameters (SimCLR convention)
        for view, dm_view, x_buf, dm_buf in [
            (view_a, dmap_a, va_buf,  dma_buf),
            (view_b, dmap_b, vb_buf,  dmb_buf),
        ]:
            mu_v  = float(view.mean())
            sig_v = max(float(view.std()), 1e-6)
            x_buf[b]  = ((view  - mu_v) / sig_v).astype(np.float32)
            dm_buf[b] = dm_view.astype(np.float32)

    return (
        torch.from_numpy(va_buf).to(device),
        torch.from_numpy(dma_buf).to(device),
        torch.from_numpy(vb_buf).to(device),
        torch.from_numpy(dmb_buf).to(device),
    )


def _sample_mae_mask(
    d_map: torch.Tensor,   # (B, L_LR)  on device
    n_patches: int,
    patch_len: int,
    masking_ratio: float,
    qrs_boost: float,
) -> torch.Tensor:         # (B, n_patches) bool, on same device as d_map
    """Sample a rhythm-aware boolean mask for MAE.

    QRS patches (those whose d_map minimum falls below a proximity threshold)
    receive probability weight = qrs_boost relative to non-QRS patches.
    Sampling is without replacement so that exactly round(ratio × P) patches
    are masked per sample.

    Position 0 (f_emb / CLS token) is never in `patches` and is therefore
    never affected here — CLAUDE.md constraint #2 is guaranteed by design.

    Args:
        d_map:         (B, L_LR) R-distance map (normalised units).
        n_patches:     Total number of patches P.
        patch_len:     Samples per patch.
        masking_ratio: Fraction of patches to mask (e.g. 0.4 → 8/20 patches).
        qrs_boost:     Probability weight multiplier for QRS-aligned patches.

    Returns:
        Boolean mask (B, n_patches), True at masked positions.
    """
    B = d_map.shape[0]
    n_masked = max(1, round(masking_ratio * n_patches))
    mask = torch.zeros(B, n_patches, dtype=torch.bool, device=d_map.device)

    # Reshape to (B, n_patches, patch_len) for vectorised per-patch minimum
    dm_np = d_map.cpu().numpy().reshape(B, n_patches, patch_len)
    patch_min = dm_np.min(axis=2)                          # (B, n_patches)

    # QRS patches: local minimum d_map < threshold → higher weight
    weights = np.where(patch_min < 0.3, qrs_boost, 1.0)   # (B, n_patches)
    weights = weights / weights.sum(axis=1, keepdims=True) # normalise per sample

    rng = np.random.default_rng()                          # thread-safe, no fixed seed
    for b in range(B):
        idx = rng.choice(n_patches, size=n_masked, replace=False, p=weights[b])
        mask[b, idx] = True

    return mask


def _unpack_batch_a(
    batch: tuple,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Move Phase-A-relevant fields to device; discard x_hr, annotations, flag.

    Tuple layout (dataset.py):
      x_lr, x_hr, d_map, f_signal, f_meta, annotations, mu_w, sigma_w, source_flag

    Returns:
        x_lr, d_map, f_signal, f_meta, mu_w, sigma_w  — all on device
    """
    x_lr, _x_hr, d_map, f_signal, f_meta, _ann, mu_w, sigma_w, _flag = batch
    return (
        x_lr.to(device,     non_blocking=True),
        d_map.to(device,    non_blocking=True),
        f_signal.to(device, non_blocking=True),
        f_meta.to(device,   non_blocking=True),
        mu_w.to(device,     non_blocking=True),
        sigma_w.to(device,  non_blocking=True),
    )


def _save_checkpoint_a(
    path: Path,
    encoder: ECGEncoder,
    proj_head: ProjectionHead,
    mae_decoder: MAEDecoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_loss: float,
) -> None:
    torch.save({
        "epoch":          epoch,
        "best_val_loss":  best_val_loss,
        "encoder":        encoder.state_dict(),
        "proj_head":      proj_head.state_dict(),
        "mae_decoder":    mae_decoder.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "scheduler":      scheduler.state_dict(),
    }, path)


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def _validate_phase_a(
    encoder: ECGEncoder,
    proj_head: ProjectionHead,
    mae_decoder: MAEDecoder,
    ntxent_loss: NTXentLoss,
    mae_loss_fn: MAELoss,
    val_loader: DataLoader,
    cfg: DictConfig,
    device: torch.device,
    augmentor: ECGAugmentation,
    n_patches: int,
    patch_len: int,
) -> dict[str, float]:
    """One validation epoch — both SimCLR and MAE losses, no weight updates.

    Returns:
        Dict with mean per-batch losses: total, ntxent, mae, mae_fft.
    """
    encoder.eval()
    proj_head.eval()
    mae_decoder.eval()

    pa  = cfg.phase_a
    w   = pa.loss_weights
    mae = pa.mae

    sums = {"total": 0.0, "ntxent": 0.0, "mae": 0.0, "mae_fft": 0.0}
    n = 0

    for batch in val_loader:
        x_lr, d_map, f_signal, f_meta, mu_w, sigma_w = _unpack_batch_a(batch, device)

        # ── SimCLR validation ────────────────────────────────────────────────
        va, dma, vb, dmb = _augment_batch(x_lr, d_map, mu_w, sigma_w, augmentor, 1.0, device)
        X_in_a = torch.stack([va, dma], dim=1)
        X_in_b = torch.stack([vb, dmb], dim=1)

        _, z_cls_a, _, _ = encoder(X_in_a, f_signal, f_meta)
        _, z_cls_b, _, _ = encoder(X_in_b, f_signal, f_meta)
        h_a = proj_head(z_cls_a)
        h_b = proj_head(z_cls_b)
        l_ntxent = ntxent_loss(h_a, h_b)

        # ── MAE validation ───────────────────────────────────────────────────
        X_in    = torch.stack([x_lr, d_map], dim=1)             # (B, 2, L_LR)
        f_emb   = encoder.feature_mlp(f_signal, f_meta)         # (B, d_model)
        p_seq, _= encoder.cnn_encoder(X_in)                     # (B, d_model, P)
        patches = p_seq.transpose(1, 2)                         # (B, P, d_model)

        mask = _sample_mae_mask(d_map, n_patches, patch_len, mae.masking_ratio, mae.qrs_masking_boost)
        masked = mae_decoder.prepare_masked_input(patches, mask)
        seq    = torch.cat([f_emb.unsqueeze(1), masked], dim=1) # (B, P+1, d_model)
        z_seq, _, _ = encoder.transformer(seq)
        preds  = mae_decoder(z_seq)                             # (B, P, patch_len)
        target = x_lr.unfold(-1, patch_len, patch_len)          # (B, P, patch_len)
        l_mae, _, l_mae_fft = mae_loss_fn(preds, target, mask)

        total = w.ntxent * l_ntxent + w.mae * l_mae
        sums["total"]    += total.item()
        sums["ntxent"]   += l_ntxent.item()
        sums["mae"]      += l_mae.item()
        sums["mae_fft"]  += l_mae_fft.item()
        n += 1

    # Restore training mode
    encoder.train()
    proj_head.train()
    mae_decoder.train()

    return {k: v / max(n, 1) for k, v in sums.items()}


# =============================================================================
# Main training loop
# =============================================================================

def train_phase_a(
    encoder: ECGEncoder,
    proj_head: ProjectionHead,
    mae_decoder: MAEDecoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: DictConfig,
    save_dir: str | Path,
    device: str | torch.device = "cuda",
    resume_ckpt: str | Path | None = None,
) -> None:
    """Phase A self-supervised pre-training loop.

    Trains encoder + projection head + MAE decoder jointly.
    At the end, saves only encoder.pt (projection head and MAE decoder are
    discarded — CLAUDE.md design).

    Args:
        encoder:      ECGEncoder instance (uninitialised or warm-started).
        proj_head:    ProjectionHead instance.
        mae_decoder:  MAEDecoder instance (holds the learnable mask token).
        train_loader: DataLoader for the training split.
        val_loader:   DataLoader for the validation split.
        cfg:          Top-level OmegaConf DictConfig (configs/default.yaml).
        save_dir:     Directory for checkpoints and final encoder.pt.
        device:       'cuda' or 'cpu'.
        resume_ckpt:  Optional path to last_phase_a.pt to resume an interrupted run.
    """
    device   = torch.device(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pa        = cfg.phase_a
    mae_cfg   = pa.mae
    patch_len = cfg.model.encoder.patch_len
    n_patches = L_LR // patch_len                     # 1000 // 50 = 20

    # ── Move models to device ─────────────────────────────────────────────────
    encoder     = encoder.to(device)
    proj_head   = proj_head.to(device)
    mae_decoder = mae_decoder.to(device)

    # ── Loss functions ────────────────────────────────────────────────────────
    ntxent_loss = NTXentLoss(temperature=pa.simclr.temperature)
    mae_loss_fn = MAELoss(fft_loss_weight=mae_cfg.fft_loss_weight)

    # ── Augmentor (curriculum aware) ──────────────────────────────────────────
    aug_params = OmegaConf.to_container(cfg.augmentations, resolve=True)
    augmentor  = ECGAugmentation(
        aug_params,
        fs=FS_LR,
        ramp_epochs=cfg.augmentations.curriculum_ramp_epochs,
    )

    # ── Optimizer: all three modules jointly ─────────────────────────────────
    all_params = (
        list(encoder.parameters())
        + list(proj_head.parameters())
        + list(mae_decoder.parameters())
    )
    optimizer = torch.optim.AdamW(
        all_params,
        lr=pa.learning_rate,
        weight_decay=pa.weight_decay,
    )

    # ── Cosine LR schedule with linear warmup ────────────────────────────────
    def _lr_lambda(epoch: int) -> float:
        if epoch < pa.warmup_epochs:
            return (epoch + 1) / pa.warmup_epochs
        progress = (epoch - pa.warmup_epochs) / max(1, pa.epochs - pa.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    start_epoch   = 0
    best_val_loss = float("inf")

    # ── Optional resume ───────────────────────────────────────────────────────
    if resume_ckpt is not None:
        ckpt = torch.load(Path(resume_ckpt), map_location="cpu", weights_only=True)
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        encoder.load_state_dict(ckpt["encoder"])
        proj_head.load_state_dict(ckpt["proj_head"])
        mae_decoder.load_state_dict(ckpt["mae_decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        logger.info("Resumed Phase A from %s — starting at epoch %d", resume_ckpt, start_epoch)

    w = pa.loss_weights

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, pa.epochs):
        encoder.train()
        proj_head.train()
        mae_decoder.train()

        strength = ECGAugmentation.compute_epoch_strength(
            epoch, cfg.augmentations.curriculum_ramp_epochs
        )

        sums: dict[str, float] = {
            "total": 0.0, "ntxent": 0.0, "mae": 0.0,
            "mae_rec": 0.0, "mae_fft": 0.0,
        }
        n_batches = 0

        for batch in train_loader:
            x_lr, d_map, f_signal, f_meta, mu_w, sigma_w = _unpack_batch_a(batch, device)

            # ── SimCLR path ───────────────────────────────────────────────────
            va, dma, vb, dmb = _augment_batch(
                x_lr, d_map, mu_w, sigma_w, augmentor, strength, device
            )
            X_in_a = torch.stack([va,   dma], dim=1)   # (B, 2, L_LR)
            X_in_b = torch.stack([vb,   dmb], dim=1)   # (B, 2, L_LR)

            _, z_cls_a, _, _ = encoder(X_in_a, f_signal, f_meta)
            _, z_cls_b, _, _ = encoder(X_in_b, f_signal, f_meta)
            h_a = proj_head(z_cls_a)                   # (B, proj_dim)  raw
            h_b = proj_head(z_cls_b)                   # (B, proj_dim)  raw
            l_ntxent = ntxent_loss(h_a, h_b)

            # ── MAE path (original, non-augmented input) ──────────────────────
            X_in    = torch.stack([x_lr, d_map], dim=1)              # (B, 2, L_LR)
            f_emb   = encoder.feature_mlp(f_signal, f_meta)          # (B, d_model)
            p_seq, _ = encoder.cnn_encoder(X_in)                     # (B, d_model, P)
            patches  = p_seq.transpose(1, 2)                         # (B, P, d_model)

            mask = _sample_mae_mask(
                d_map, n_patches, patch_len,
                mae_cfg.masking_ratio, mae_cfg.qrs_masking_boost,
            )
            # Replace masked positions with learnable mask token (constraint #2 safe)
            masked = mae_decoder.prepare_masked_input(patches, mask)  # (B, P, d_model)

            # Build Transformer input: f_emb at position 0, never masked (constraint #2)
            cls_token = f_emb.unsqueeze(1)                            # (B, 1, d_model)
            seq       = torch.cat([cls_token, masked], dim=1)        # (B, P+1, d_model)
            z_seq, _, _ = encoder.transformer(seq)                   # (B, P+1, d_model)

            patch_preds   = mae_decoder(z_seq)                       # (B, P, patch_len)
            target_patches = x_lr.unfold(-1, patch_len, patch_len)   # (B, P, patch_len)
            l_mae, l_mae_rec, l_mae_fft = mae_loss_fn(patch_preds, target_patches, mask)

            # ── Combined loss and update ──────────────────────────────────────
            loss = w.ntxent * l_ntxent + w.mae * l_mae

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            sums["total"]    += loss.item()
            sums["ntxent"]   += l_ntxent.item()
            sums["mae"]      += l_mae.item()
            sums["mae_rec"]  += l_mae_rec.item()
            sums["mae_fft"]  += l_mae_fft.item()
            n_batches        += 1

        scheduler.step()

        means = {k: v / n_batches for k, v in sums.items()} if n_batches else sums
        val   = _validate_phase_a(
            encoder, proj_head, mae_decoder,
            ntxent_loss, mae_loss_fn,
            val_loader, cfg, device,
            augmentor, n_patches, patch_len,
        )
        lr_now = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch %3d | LR=%.2e | aug_strength=%.2f | "
            "Train: total=%.4f  ntxent=%.4f  mae=%.4f  mae_rec=%.4f  mae_fft=%.4f | "
            "Val:   total=%.4f  ntxent=%.4f  mae=%.4f",
            epoch, lr_now, strength,
            means["total"], means["ntxent"], means["mae"], means["mae_rec"], means["mae_fft"],
            val["total"],   val["ntxent"],   val["mae"],
        )

        _wandb_log({
            "epoch":            epoch,
            "lr":               lr_now,
            "aug_strength":     strength,
            "train/total":      means["total"],
            "train/ntxent":     means["ntxent"],
            "train/mae":        means["mae"],
            "train/mae_rec":    means["mae_rec"],
            "train/mae_fft":    means["mae_fft"],
            "val/total":        val["total"],
            "val/ntxent":       val["ntxent"],
            "val/mae":          val["mae"],
            "val/mae_fft":      val["mae_fft"],
        }, step=epoch)

        # ── Checkpoint every epoch ────────────────────────────────────────────
        _save_checkpoint_a(
            save_dir / "last_phase_a.pt",
            encoder, proj_head, mae_decoder,
            optimizer, scheduler, epoch, best_val_loss,
        )
        if val["total"] < best_val_loss:
            best_val_loss = val["total"]
            _save_checkpoint_a(
                save_dir / "best_phase_a.pt",
                encoder, proj_head, mae_decoder,
                optimizer, scheduler, epoch, best_val_loss,
            )
            logger.info("Epoch %d: new best val_loss=%.4f → best_phase_a.pt", epoch, best_val_loss)

    # ── Export encoder only (proj_head and mae_decoder discarded) ─────────────
    torch.save(encoder.state_dict(), save_dir / "encoder.pt")
    logger.info(
        "Phase A complete (%d epochs). Saved encoder.pt to %s",
        pa.epochs, save_dir,
    )
