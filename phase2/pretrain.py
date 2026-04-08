"""
PretrainModel and training loop for Phase 2 self-supervised pre-training.

PretrainModel wraps all components:
  - HybridEncoder   (kept for Phase 3)
  - ProjectionHead  (discarded after pre-training)
  - MAEDecoder      (discarded after pre-training)
  - mask_token      (learnable Parameter, discarded after pre-training)

Training uses a three-phase curriculum:
  Phase A (epochs 0–49):      MAE-only, mask_ratio=0.25, native data only.
  Phase B (epochs 50–149):    MAE + NT-Xent, mask_ratio=0.40, all data.
  Phase C (epochs 150–199):   MAE rhythm-aware + NT-Xent, all data.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .contrastive import nt_xent_loss
from .encoder import HybridEncoder
from .mae import MAEDecoder, apply_mask_to_tokens, create_mask, get_r_peak_patches, mae_loss
from .projector import ProjectionHead

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wrapper model
# ---------------------------------------------------------------------------

class PretrainModel(nn.Module):
    """
    Composite model for Phase 2 pre-training.

    Attributes saved/exported:
      encoder         → encoder_phase3.pt  (Phase 3 fine-tuning)
      projection_head → discarded
      mae_decoder     → discarded
      mask_token      → discarded
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        d_model = config.encoder.d_model

        self.encoder         = HybridEncoder(config)
        self.projection_head = ProjectionHead(list(config.contrastive.projection_head_dims))
        self.mae_decoder     = MAEDecoder(
            d_model         = d_model,
            decoder_d_model = config.mae.decoder_d_model,
            num_layers      = config.mae.decoder_num_layers,
            num_heads       = config.mae.decoder_num_heads,
            patch_len       = config.data.patch_len,
        )
        self.mask_token = nn.Parameter(torch.zeros(d_model))

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def forward_contrastive(
        self,
        x_aug1: torch.Tensor,
        x_aug2: torch.Tensor,
        f: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute NT-Xent projection vectors for two augmented views.

        Args:
            x_aug1: (B, L) — first augmented view.
            x_aug2: (B, L) — second augmented view (different augmentation).
            f:      (B, f_dim) — normalised physio features.

        Returns:
            h1, h2: each (B, proj_out_dim) — L2-normalisation is applied in
                    nt_xent_loss, NOT here.
        """
        _, z_cls1 = self.encoder(x_aug1, f)
        _, z_cls2 = self.encoder(x_aug2, f)
        h1 = self.projection_head(z_cls1)   # (B, proj_out)
        h2 = self.projection_head(z_cls2)   # (B, proj_out)
        return h1, h2

    def forward_mae(
        self,
        x_lr: torch.Tensor,
        f: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode x_lr, apply the mask, and decode to patch reconstructions.

        Args:
            x_lr: (B, L)              — original (un-augmented) LR signal.
            f:    (B, f_dim)          — normalised physio features.
            mask: BoolTensor (B, num_patches) — True = patch is masked.

        Returns:
            x_rec: (B, num_patches, patch_len) — reconstruction of ALL patches.
                   The loss function (mae_loss) uses only the masked subset.
        """
        z_seq, _ = self.encoder(x_lr, f)
        z_masked = apply_mask_to_tokens(z_seq, mask, self.mask_token)
        return self.mae_decoder(z_masked)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: PretrainModel,
    dataloader: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    config: Any,
    epoch: int,
    augmentations: Any,
    device: torch.device,
) -> float:
    """
    Train for one epoch with curriculum-aware loss switching.

    Curriculum:
      Phase A (epoch < phase_A_epochs):
        - Loss:       MAE only
        - mask_ratio: 0.25  (easier warm-up task)
        - Data:       source == "native" (PTB-XL ground-truth only)
        - Rhythm:     no rhythm-aware masking

      Phase B (phase_A_epochs ≤ epoch < phase_B_epochs):
        - Loss:       MAE + NT-Xent
        - mask_ratio: config.mae.mask_ratio (0.40)
        - Data:       all sources
        - Rhythm:     no rhythm-aware masking

      Phase C (epoch ≥ phase_B_epochs):
        - Loss:       MAE + NT-Xent
        - mask_ratio: config.mae.mask_ratio (0.40)
        - Data:       all sources
        - Rhythm:     rhythm-aware masking enabled

    Returns:
        Mean training loss for this epoch.
    """
    model.train()
    rng = np.random.default_rng()

    # ---- Curriculum phase determination ----------------------------------
    if epoch < config.curriculum.phase_A_epochs:
        active_losses  = {"mae"}
        mask_ratio     = 0.25
        filter_source  = "native"
        rhythm_aware   = False
    elif epoch < config.curriculum.phase_B_epochs:
        active_losses  = {"mae", "contrastive"}
        mask_ratio     = config.mae.mask_ratio
        filter_source  = None
        rhythm_aware   = False
    else:
        active_losses  = {"mae", "contrastive"}
        mask_ratio     = config.mae.mask_ratio
        filter_source  = None
        rhythm_aware   = config.mae.rhythm_aware_mask

    num_patches = config.data.num_patches
    patch_len   = config.data.patch_len
    temperature = config.contrastive.temperature
    lambda_cl   = config.loss.lambda_cl
    lambda_mae  = config.loss.lambda_mae
    lambda_fft  = config.mae.lambda_fft
    weight_qrs  = config.mae.mask_weight_qrs
    grad_clip   = config.optimizer.grad_clip

    total_loss = 0.0
    n_batches  = 0

    for batch in dataloader:
        x_lr_t  = batch["x_lr"]    # FloatTensor (B, L) on CPU
        f       = batch["f"].to(device)
        a_t     = batch["a"]        # FloatTensor (B, max_K, 2) on CPU
        sources = batch["source"]   # list[str]

        # ---- Source filtering (Phase A) ----------------------------------
        if filter_source is not None:
            keep = [i for i, s in enumerate(sources) if s == filter_source]
            if not keep:
                continue
            keep_idx = torch.tensor(keep, dtype=torch.long)
            x_lr_t = x_lr_t[keep_idx]
            f      = f[keep_idx]
            a_t    = a_t[keep_idx]

        B = x_lr_t.shape[0]
        if B == 0:
            continue

        loss = torch.tensor(0.0, device=device)

        with torch.amp.autocast("cuda", enabled=config.amp):

            # ---- Contrastive branch --------------------------------------
            if "contrastive" in active_losses:
                x_np = x_lr_t.numpy()
                x_aug1_np = np.stack([augmentations(x_np[i]) for i in range(B)])
                x_aug2_np = np.stack([augmentations(x_np[i]) for i in range(B)])
                x_aug1 = torch.tensor(x_aug1_np, dtype=torch.float32, device=device)
                x_aug2 = torch.tensor(x_aug2_np, dtype=torch.float32, device=device)

                h1, h2    = model.forward_contrastive(x_aug1, x_aug2, f)
                loss_cl   = nt_xent_loss(h1, h2, temperature)
                loss      = loss + lambda_cl * loss_cl

            # ---- MAE branch ----------------------------------------------
            if "mae" in active_losses:
                x_lr = x_lr_t.to(device)
                a_np = a_t.numpy()   # (B, max_K, 2)

                masks = []
                for i in range(B):
                    r_patches: list[int] = []
                    if rhythm_aware:
                        r_patches = get_r_peak_patches(
                            a_np[i], num_patches, patch_len
                        )
                    mask_i = create_mask(
                        num_patches, mask_ratio, r_patches, weight_qrs, rng
                    )
                    masks.append(mask_i)

                mask_t = torch.tensor(
                    np.stack(masks), dtype=torch.bool, device=device
                )  # (B, num_patches)

                x_rec    = model.forward_mae(x_lr, f, mask_t)
                loss_mae = mae_loss(x_rec, x_lr, mask_t, lambda_fft)
                loss     = loss + lambda_mae * loss_mae

        # ---- Backward + gradient clipping --------------------------------
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_pretrain_checkpoint(
    model: PretrainModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
) -> None:
    """
    Save a full training checkpoint (all model components + optimizer).

    The checkpoint includes both the full model state dict (for resuming
    pre-training) and the encoder-only state dict (for convenience when
    loading into Phase 3 without the full pretrain model class).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":                  epoch,
            "encoder_state_dict":     model.encoder.state_dict(),
            "full_model_state_dict":  model.state_dict(),
            "optimizer_state_dict":   optimizer.state_dict(),
        },
        path,
    )
    logger.info("Pre-train checkpoint saved → %s (epoch %d)", path, epoch)


def export_encoder_for_phase3(model: PretrainModel, path: Path) -> None:
    """
    Export encoder weights and config for Phase 3 fine-tuning.

    Saves ONLY the encoder (discards ProjectionHead, MAEDecoder, mask_token).
    The config is serialised as a plain dict so it can be reconstructed
    without the full OmegaConf dependency.

    Phase 3 loading:
        ckpt = torch.load("encoder_phase3.pt", weights_only=False)
        cfg  = OmegaConf.create(ckpt["encoder_config"])
        enc  = HybridEncoder(cfg)
        enc.load_state_dict(ckpt["encoder_state_dict"])
    """
    from omegaconf import OmegaConf

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "encoder_config":     OmegaConf.to_container(
                model.encoder.config, resolve=True
            ),
        },
        path,
    )
    logger.info("Encoder exported for Phase 3 → %s", path)
