"""
phase_b.py — Supervised fine-tuning loop for ECG super-resolution (Phase B).

Loads the encoder trained in Phase A (encoder.pt) and fine-tunes it alongside
a new UNetDecoder and PatchGANDiscriminator using paired LR-HR ECG data.

Training schedule
-----------------
Epochs 0 .. encoder_freeze_epochs-1:
    Encoder frozen.  Only decoder (and discriminator) parameters are updated.

Epochs encoder_freeze_epochs .. end:
    Encoder unfrozen.  Optimizer rebuilt once with layer-wise LR decay so that
    deeper encoder layers receive lower learning rates:
        lr_group_i = base_lr × decay^(N-1-i)   (i=0 is deepest, i=N-1 is shallowest)

Epochs 0 .. discriminator_warmup_epochs-1:
    No adversarial or feature-matching loss — pure reconstruction training.
    Ensures the generator has reasonable outputs before the discriminator is enabled.

Epochs discriminator_warmup_epochs .. end:
    Full generator loss:  L_G = w.l1×L1 + w.fft×FFT + w.stft×STFT
                               + w.morphological×Lmorph + w.adversarial×Ladv + w.fm×LFM

Entry point
-----------
    train_phase_b(cfg, encoder_ckpt, train_loader, val_loader, save_dir, device, resume_ckpt)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.models.decoder import UNetDecoder
from src.models.discriminator import PatchGANDiscriminator
from src.models.encoder import ECGEncoder
from src.losses.adversarial import FeatureMatchingLoss, HingeLoss
from src.losses.morphological import MorphologicalLoss
from src.losses.reconstruction import ReconstructionLoss

logger = logging.getLogger(__name__)


# =============================================================================
# Encoder loading
# =============================================================================

def _load_encoder(encoder: ECGEncoder, ckpt_path: str | Path) -> None:
    """Load encoder weights from a Phase A checkpoint.

    Accepts checkpoints saved as a bare state_dict or a dict with an
    'encoder', 'model_state_dict', or 'state_dict' key.

    Args:
        encoder:   Encoder module to populate.
        ckpt_path: Path to encoder.pt from train_phase_a.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if isinstance(ckpt, dict):
        for key in ("encoder", "model_state_dict", "state_dict"):
            if key in ckpt:
                state = ckpt[key]
                break
        else:
            state = ckpt   # top-level keys are the state dict
    else:
        state = ckpt       # bare OrderedDict

    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing encoder keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected encoder keys: %s", unexpected)
    logger.info("Loaded encoder from %s", ckpt_path)


# =============================================================================
# Layer-wise LR decay
# =============================================================================

def _encoder_param_groups(
    encoder: ECGEncoder,
    base_lr: float,
    decay: float,
) -> list[dict[str, Any]]:
    """Return AdamW parameter groups with layer-wise LR decay.

    Groups ordered deepest → shallowest:
      0   feature_mlp
      1   cnn_encoder.initial_conv
      2-5 cnn_encoder.blocks[0..3]        (n_residual_blocks entries)
      6   cnn_encoder.patch_conv
      7-12 transformer.layers[0..5]       (n_layers entries, final_norm merged into last)

    lr_group_i = base_lr × decay^(N-1-i)
    Positional encoding has no learnable parameters and is skipped.

    Args:
        encoder:  ECGEncoder instance (must be fully initialised before call).
        base_lr:  LR for the shallowest group (shallowest layer gets full base_lr).
        decay:    Per-layer decay factor (cfg.phase_b.lr_layer_decay = 0.8).

    Returns:
        List of {'params': [...], 'lr': float} dicts ready for an optimizer.
    """
    raw: list[list[nn.Parameter]] = []

    raw.append(list(encoder.feature_mlp.parameters()))
    raw.append(list(encoder.cnn_encoder.initial_conv.parameters()))
    for block in encoder.cnn_encoder.blocks:
        raw.append(list(block.parameters()))
    raw.append(list(encoder.cnn_encoder.patch_conv.parameters()))
    # pos_enc uses only register_buffer — no learnable params, skip
    for layer in encoder.transformer.layers:
        raw.append(list(layer.parameters()))
    # Merge final_norm into the topmost transformer layer group
    raw[-1] = raw[-1] + list(encoder.transformer.final_norm.parameters())

    raw = [g for g in raw if g]   # drop empty groups (safety)
    N = len(raw)

    return [
        {"params": params, "lr": base_lr * (decay ** (N - 1 - i))}
        for i, params in enumerate(raw)
    ]


# =============================================================================
# Batch helpers
# =============================================================================

def _unpack_batch(
    batch: tuple,
    device: torch.device,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
    torch.Tensor | None,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Move training tuple to device.

    Tuple layout (dataset.py):
      x_LR, x_HR, d_map, f_signal, f_meta, annotations, mu_w, sigma_w, source_flag
    """
    x_lr, x_hr, d_map, f_signal, f_meta, annotations, mu_w, sigma_w, source_flag = batch
    x_lr     = x_lr.to(device, non_blocking=True)
    x_hr     = x_hr.to(device, non_blocking=True)
    d_map    = d_map.to(device, non_blocking=True)
    f_signal = f_signal.to(device, non_blocking=True)
    f_meta   = f_meta.to(device, non_blocking=True)
    mu_w     = mu_w.to(device, non_blocking=True)
    sigma_w  = sigma_w.to(device, non_blocking=True)
    if annotations is not None:
        annotations = annotations.to(device, non_blocking=True)
    return x_lr, x_hr, d_map, f_signal, f_meta, annotations, mu_w, sigma_w, source_flag


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def _validate(
    encoder: ECGEncoder,
    decoder: UNetDecoder,
    rec_loss_fn: ReconstructionLoss,
    morph_loss_fn: MorphologicalLoss,
    val_loader: DataLoader,
    w: DictConfig,
    device: torch.device,
) -> dict[str, float]:
    """One validation epoch — reconstruction losses only, no adversarial.

    Returns:
        Dict with mean per-batch losses: total, l1, fft, stft, morph.
    """
    encoder.eval()
    decoder.eval()

    sums: dict[str, float] = {"total": 0.0, "l1": 0.0, "fft": 0.0, "stft": 0.0, "morph": 0.0}
    n = 0

    for batch in val_loader:
        x_lr, x_hr, d_map, f_signal, f_meta, annotations, _mu, _sigma, _flag = (
            _unpack_batch(batch, device)
        )
        X_in = torch.stack([x_lr, d_map], dim=1)         # (B, 2, L_LR)

        _, z_cls, z_patches, skips = encoder(X_in, f_signal, f_meta)
        x_hat = decoder(z_patches, z_cls, skips)          # (B, L_HR)

        l1, fft, stft = rec_loss_fn(x_hat, x_hr)
        morph = morph_loss_fn(x_hat, x_hr, annotations)
        total = w.l1 * l1 + w.fft * fft + w.stft * stft + w.morphological * morph

        sums["total"] += total.item()
        sums["l1"]    += l1.item()
        sums["fft"]   += fft.item()
        sums["stft"]  += stft.item()
        sums["morph"] += morph.item()
        n += 1

    if n == 0:
        return sums
    return {k: v / n for k, v in sums.items()}


# =============================================================================
# Checkpoint helpers
# =============================================================================

def _save_checkpoint(
    path: Path,
    encoder: ECGEncoder,
    decoder: UNetDecoder,
    discriminator: PatchGANDiscriminator,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    encoder_unfrozen: bool,
) -> None:
    torch.save({
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "encoder_unfrozen": encoder_unfrozen,
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
    }, path)


# =============================================================================
# Main training loop
# =============================================================================

def train_phase_b(
    cfg: DictConfig,
    encoder_ckpt: str | Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_dir: str | Path,
    device: str | torch.device = "cuda",
    resume_ckpt: str | Path | None = None,
) -> None:
    """Phase B supervised fine-tuning loop.

    Args:
        cfg:          Top-level OmegaConf DictConfig (configs/default.yaml).
        encoder_ckpt: Path to encoder.pt from train_phase_a.
        train_loader: DataLoader for the training split.
        val_loader:   DataLoader for the validation split.
        save_dir:     Directory for checkpoints and final model files.
        device:       'cuda' or 'cpu'.
        resume_ckpt:  Optional path to last.pt to resume an interrupted run.
    """
    device = torch.device(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pb = cfg.phase_b
    w  = pb.loss_weights

    # ── Build models ──────────────────────────────────────────────────────────
    encoder       = ECGEncoder(cfg).to(device)
    decoder       = UNetDecoder(cfg).to(device)
    discriminator = PatchGANDiscriminator(cfg).to(device)

    _load_encoder(encoder, encoder_ckpt)

    # ── Loss functions ────────────────────────────────────────────────────────
    rec_loss_fn   = ReconstructionLoss(cfg).to(device)
    morph_loss_fn = MorphologicalLoss(cfg)
    hinge_loss    = HingeLoss()
    fm_loss       = FeatureMatchingLoss()

    # ── Initial state (encoder frozen) ───────────────────────────────────────
    encoder.requires_grad_(False)
    encoder_unfrozen = False

    opt_g = torch.optim.AdamW(
        decoder.parameters(),
        lr=pb.learning_rate_generator,
        weight_decay=pb.weight_decay,
    )
    opt_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=pb.learning_rate_discriminator,
        weight_decay=pb.weight_decay,
    )

    start_epoch   = 0
    best_val_loss = float("inf")

    # ── Optional resume ───────────────────────────────────────────────────────
    if resume_ckpt is not None:
        ckpt = torch.load(Path(resume_ckpt), map_location="cpu", weights_only=True)
        start_epoch       = ckpt["epoch"] + 1
        best_val_loss     = ckpt.get("best_val_loss", float("inf"))
        encoder_unfrozen  = ckpt.get("encoder_unfrozen", False)

        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        discriminator.load_state_dict(ckpt["discriminator"])

        # Rebuild opt_g to match the saved group structure before loading state
        if encoder_unfrozen:
            encoder.requires_grad_(True)
            enc_groups = _encoder_param_groups(encoder, pb.learning_rate_generator, pb.lr_layer_decay)
            opt_g = torch.optim.AdamW(
                enc_groups + [{"params": list(decoder.parameters()), "lr": pb.learning_rate_generator}],
                weight_decay=pb.weight_decay,
            )

        try:
            opt_g.load_state_dict(ckpt["opt_g"])
            opt_d.load_state_dict(ckpt["opt_d"])
        except (ValueError, KeyError) as exc:
            logger.warning("Optimizer state not restored (%s); starting optimizers fresh.", exc)

        logger.info("Resumed from %s — starting at epoch %d", resume_ckpt, start_epoch)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, pb.epochs):

        # ── Progressive unfreezing: rebuild optimizer once ────────────────────
        if not encoder_unfrozen and epoch >= pb.encoder_freeze_epochs:
            encoder.requires_grad_(True)
            encoder_unfrozen = True
            enc_groups = _encoder_param_groups(encoder, pb.learning_rate_generator, pb.lr_layer_decay)
            opt_g = torch.optim.AdamW(
                enc_groups + [{"params": list(decoder.parameters()), "lr": pb.learning_rate_generator}],
                weight_decay=pb.weight_decay,
            )
            logger.info(
                "Epoch %d: encoder unfrozen with layer-wise LR decay (γ=%.2f, %d groups)",
                epoch, pb.lr_layer_decay, len(enc_groups),
            )

        use_adv = epoch >= pb.discriminator_warmup_epochs

        encoder.train()
        decoder.train()
        discriminator.train()

        sums: dict[str, float] = {
            "g_total": 0.0, "l1": 0.0, "fft": 0.0, "stft": 0.0,
            "morph": 0.0, "adv_g": 0.0, "fm": 0.0, "d_loss": 0.0,
        }
        n_batches = 0

        for batch in train_loader:
            x_lr, x_hr, d_map, f_signal, f_meta, annotations, _mu, _sigma, _flag = (
                _unpack_batch(batch, device)
            )
            X_in = torch.stack([x_lr, d_map], dim=1)     # (B, 2, L_LR)

            # ── Generator update ──────────────────────────────────────────────
            _, z_cls, z_patches, skips = encoder(X_in, f_signal, f_meta)
            x_hat = decoder(z_patches, z_cls, skips)      # (B, L_HR)

            l1, fft, stft = rec_loss_fn(x_hat, x_hr)
            morph         = morph_loss_fn(x_hat, x_hr, annotations)

            loss_g    = w.l1 * l1 + w.fft * fft + w.stft * stft + w.morphological * morph
            adv_g_val = x_hat.new_zeros(1)
            fm_val    = x_hat.new_zeros(1)

            if use_adv:
                # Forward through D with gradient flowing back to generator
                logits_fake_g, feats_fake = discriminator(x_hat)
                with torch.no_grad():
                    _, feats_real = discriminator(x_hr)
                adv_g_val = hinge_loss.generator_loss(logits_fake_g)
                fm_val    = fm_loss(feats_fake, feats_real)
                loss_g    = loss_g + w.adversarial * adv_g_val + w.feature_matching * fm_val

            opt_g.zero_grad()
            loss_g.backward()
            trainable_g = [p for p in list(encoder.parameters()) + list(decoder.parameters())
                           if p.requires_grad]
            nn.utils.clip_grad_norm_(trainable_g, max_norm=1.0)
            opt_g.step()

            # ── Discriminator update ──────────────────────────────────────────
            d_loss_val = x_hat.new_zeros(1)
            if use_adv:
                logits_real,   _ = discriminator(x_hr)
                logits_fake_d, _ = discriminator(x_hat.detach())   # detach: no G gradient
                d_loss_val = hinge_loss.discriminator_loss(logits_real, logits_fake_d)
                opt_d.zero_grad()
                d_loss_val.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                opt_d.step()

            sums["g_total"] += loss_g.item()
            sums["l1"]      += l1.item()
            sums["fft"]     += fft.item()
            sums["stft"]    += stft.item()
            sums["morph"]   += morph.item()
            sums["adv_g"]   += adv_g_val.item()
            sums["fm"]      += fm_val.item()
            sums["d_loss"]  += d_loss_val.item()
            n_batches += 1

        # ── Epoch logging ─────────────────────────────────────────────────────
        means = {k: v / n_batches for k, v in sums.items()} if n_batches else sums
        val   = _validate(encoder, decoder, rec_loss_fn, morph_loss_fn, val_loader, w, device)

        logger.info(
            "Epoch %3d | "
            "G=%.4f  l1=%.4f  fft=%.4f  stft=%.4f  morph=%.4f  "
            "adv_G=%.4f  FM=%.4f  D=%.4f | "
            "Val=%.4f  val_l1=%.4f",
            epoch,
            means["g_total"], means["l1"], means["fft"],
            means["stft"],    means["morph"],
            means["adv_g"],   means["fm"],  means["d_loss"],
            val["total"],     val["l1"],
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        _save_checkpoint(
            save_dir / "last.pt",
            encoder, decoder, discriminator, opt_g, opt_d,
            epoch, best_val_loss, encoder_unfrozen,
        )
        if val["total"] < best_val_loss:
            best_val_loss = val["total"]
            _save_checkpoint(
                save_dir / "best.pt",
                encoder, decoder, discriminator, opt_g, opt_d,
                epoch, best_val_loss, encoder_unfrozen,
            )
            logger.info("Epoch %d: new best val_loss=%.4f → best.pt", epoch, best_val_loss)

    # ── Final model export ────────────────────────────────────────────────────
    torch.save(encoder.state_dict(),       save_dir / "encoder_finetuned.pt")
    torch.save(decoder.state_dict(),       save_dir / "decoder.pt")
    logger.info("Phase B complete. Exported encoder_finetuned.pt and decoder.pt to %s", save_dir)
