"""
mae_decoder.py — Lightweight MAE decoder for Phase A self-supervised pre-training.

Reconstructs masked CNN patch embeddings from the full Transformer encoder output.
This module is DISCARDED after Phase A; only encoder weights (encoder.pt) are kept.

Masking protocol (CLAUDE.md constraints #2 and #4):
  - The learnable mask_token (stored here) is substituted into CNN patch embeddings
    BEFORE the Transformer encoder forward pass.  Position 0 (f_emb / CLS) is
    NEVER masked.
  - Reconstruction target: per-patch z-score of x_LR using TARGET patch statistics
    (not prediction statistics).  Loss computation lives in src/losses/mae_loss.py.

Phase A training loop flow:
  1. cnn_encoder(X_in)  → patches (B, P, d_model)  + skips
  2. mae_decoder.prepare_masked_input(patches, mask)  → masked_patches
  3. Build sequence:  cat([f_emb, masked_patches]) → (B, P+1, d_model)
  4. transformer(seq)  → z_seq (B, P+1, d_model)
  5. mae_decoder(z_seq) → patch_preds (B, P, patch_len)
  6. mae_loss(patch_preds, x_LR, mask)  → scalar loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.data.preprocessing import L_LR
from src.models.encoder import SinusoidalPositionalEncoding, _PreLNTransformerLayer


class MAEDecoder(nn.Module):
    """Lightweight MAE decoder: z_seq → per-patch signal reconstructions.

    Input:
        z_seq: (B, P+1, d_model) — full Transformer encoder output.
               Masked patch positions were encoded via the learnable mask_token.

    Output:
        patch_preds: (B, P, patch_len) — predicted signal values for every patch.
                     The training loop selects masked positions for loss computation.

    Architecture:
        Linear(d_model → decoder_hidden_dim)     ← project to lighter decoder space
        + SinusoidalPositionalEncoding           ← fixed, not learnable
        + decoder_n_layers × Pre-LN Transformer  ← lightweight self-attention
        + LayerNorm                              ← final normalisation
        + Linear(decoder_hidden_dim → patch_len) ← per-patch signal head (positions 1:)

    Mask token:
        self.mask_token (d_model,) — learnable, init N(0, 0.02).
        Used via prepare_masked_input() before the Transformer encoder call.

    Args:
        cfg: Full OmegaConf config.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        d_model   = cfg.model.d_model
        mae       = cfg.phase_a.mae
        n_patches = L_LR // cfg.model.encoder.patch_len   # 1000 // 50 = 20
        patch_len = cfg.model.encoder.patch_len            # 50

        # Learnable mask token — substituted for masked CNN patches before encoding
        self.mask_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Project encoder space → lighter decoder space
        self.input_proj = nn.Linear(d_model, mae.decoder_hidden_dim)

        # Positional encoding for decoder sequence (P+1 positions)
        self.pos_enc = SinusoidalPositionalEncoding(
            mae.decoder_hidden_dim,
            max_len=n_patches + 1,
            dropout=0.0,
        )

        # Lightweight Transformer layers (Pre-LN, same style as encoder)
        ffn_dim = mae.decoder_hidden_dim * 4
        self.layers = nn.ModuleList([
            _PreLNTransformerLayer(
                d_model=mae.decoder_hidden_dim,
                n_heads=mae.decoder_n_heads,
                ffn_dim=ffn_dim,
                dropout=cfg.model.transformer.dropout,
                attn_dropout=0.0,
            )
            for _ in range(mae.decoder_n_layers)
        ])
        self.final_norm = nn.LayerNorm(mae.decoder_hidden_dim)

        # Per-patch output head: predicts patch_len signal values
        self.output_head = nn.Linear(mae.decoder_hidden_dim, patch_len)

    # ------------------------------------------------------------------
    # Masking helper (called by training loop before encoder forward)
    # ------------------------------------------------------------------

    def prepare_masked_input(
        self,
        patches: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Substitute masked positions with the learnable mask token.

        CLAUDE.md constraint #2: position 0 (CLS / f_emb) is never in `patches`
        and is therefore never affected by this call.

        Args:
            patches: (B, P, d_model) — CNN patch embeddings from CNNEncoder.
            mask:    (B, P) bool     — True where the patch should be hidden.

        Returns:
            (B, P, d_model) with masked positions replaced by mask_token.
        """
        out = patches.clone()
        # mask_token is (1, 1, d_model); squeeze → (d_model,) broadcasts over n_masked
        out[mask] = self.mask_token.squeeze()
        return out

    # ------------------------------------------------------------------
    # Forward: z_seq → patch signal predictions
    # ------------------------------------------------------------------

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_seq: (B, P+1, d_model) — Transformer encoder output.
                   Position 0 is the CLS token; positions 1:P+1 are patch tokens.

        Returns:
            patch_preds: (B, P, patch_len) — predicted signal values per patch.
                         All P patches are returned; the loss masks externally.
        """
        x = self.input_proj(z_seq)       # (B, P+1, decoder_hidden_dim)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        patch_tokens = x[:, 1:, :]       # (B, P, decoder_hidden_dim) — skip CLS pos
        return self.output_head(patch_tokens)  # (B, P, patch_len)
