"""
encoder.py — Multi-scale CNN + Transformer encoder for ECG super-resolution.

Full pipeline (Section 3.2, architectural PDF):

  ECGEncoder (top level)
    ├── FeatureProjectionMLP  f_signal(R^12) → 192-d, f_meta(R^7) → 64-d → cat → d_model
    ├── CNNEncoder            X_in:(B,2,N_LR) → patch_seq:(B,d_model,P) + skips
    │     ├── initial Conv1d(2→d_model, k=7)
    │     ├── MultiScaleCNNBlock × L  (residual, 3 parallel kernels)
    │     └── patch Conv1d(stride=patch_len) → P patches
    └── TransformerEncoder    [f_emb, patches]:(B,P+1,d_model) → z_seq, z_cls, z_patches

Shapes: (B, C, T) for 1-D signals, (B, T, D) for sequences.
All hyperparameters read from configs/default.yaml via OmegaConf DictConfig.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.data.preprocessing import L_LR


# =============================================================================
# Feature Projection MLPs
# =============================================================================

class FeatureProjectionMLP(nn.Module):
    """Two independent MLPs projecting f_signal and f_meta, outputs concatenated.

    f_signal ∈ R^12 → MLP(12→64→192) → 192-d
    f_meta   ∈ R^7  → MLP(7→32→64)   → 64-d
    f_emb = concat([signal_emb, meta_emb]) ∈ R^d_model

    Eq. 9–11 of architectural PDF.  Signal MLP gets more capacity than meta MLP
    (proportional split configured in f_signal_mlp / f_meta_mlp subcfgs).

    Args:
        cfg: model subcfg (fields: f_signal_mlp, f_meta_mlp).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        s, m = cfg.f_signal_mlp, cfg.f_meta_mlp

        self.signal_mlp = nn.Sequential(
            nn.Linear(s.input_dim, s.hidden_dim),
            nn.GELU(),
            nn.Linear(s.hidden_dim, s.output_dim),
        )
        self.meta_mlp = nn.Sequential(
            nn.Linear(m.input_dim, m.hidden_dim),
            nn.GELU(),
            nn.Linear(m.hidden_dim, m.output_dim),
        )

    def forward(
        self,
        f_signal: torch.Tensor,  # (B, F_SIGNAL_DIM=12)
        f_meta: torch.Tensor,    # (B, F_META_DIM=7)
    ) -> torch.Tensor:
        """Returns f_emb: (B, d_model)."""
        return torch.cat([self.signal_mlp(f_signal), self.meta_mlp(f_meta)], dim=-1)


# =============================================================================
# Multi-Scale CNN Encoder
# =============================================================================

class MultiScaleCNNBlock(nn.Module):
    """Residual block with three parallel multi-scale convolution branches.

    For each branch j with kernel k_j ∈ {k_small, k_medium, k_large}:
        h_j = Conv1d(e, k_j, d_model → branch_dim, pad=k_j//2)

    branch_dim = d_model // n_branches  (e.g. 256 // 3 = 85)
    Concatenation restores d_model via 1×1 conv: cat_dim → d_model.

    Residual with pre-activation pattern (Eq. 15 PDF):
        e_out = e_in + GELU(LN(fuse(concat(branches))))

    Input/output: (B, d_model, T)  — temporal length T is preserved.
    """

    def __init__(self, d_model: int, kernel_sizes: list[int]) -> None:
        super().__init__()
        n = len(kernel_sizes)
        branch_dim = d_model // n       # 256 // 3 = 85
        cat_dim    = branch_dim * n     # 85  *  3 = 255

        self.branches = nn.ModuleList([
            nn.Conv1d(d_model, branch_dim, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.fuse = nn.Conv1d(cat_dim, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_model, T) → (B, d_model, T)."""
        h = torch.cat([b(x) for b in self.branches], dim=1)   # (B, cat_dim, T)
        h = self.fuse(h)                                        # (B, d_model, T)
        # LN over channel dim: transpose to (B, T, d_model), norm, transpose back
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        return x + F.gelu(h)


class CNNEncoder(nn.Module):
    """Multi-scale CNN: initial embedding → L residual blocks → patchification.

    All intermediate residual outputs e[1]…e[L] (each (B, d_model, N_LR)) are
    returned as skip connections for the Phase B U-Net decoder (Eq. 16 PDF).

    Args:
        d_model: Embedding / channel dimension.
        cfg: model subcfg (field: encoder with n_residual_blocks, kernel_sizes, patch_len).
    """

    def __init__(self, d_model: int, cfg: DictConfig) -> None:
        super().__init__()
        enc = cfg.encoder

        # Initial embedding: 2-channel input → d_model feature maps (Eq. 12 PDF)
        self.initial_conv = nn.Conv1d(2, d_model, kernel_size=7, padding=3)

        # Residual blocks (L = n_residual_blocks)
        self.blocks = nn.ModuleList([
            MultiScaleCNNBlock(d_model, list(enc.kernel_sizes))
            for _ in range(enc.n_residual_blocks)
        ])

        # Patchification: non-overlapping stride-conv → P = N_LR // patch_len patches
        self.patch_conv = nn.Conv1d(
            d_model, d_model, kernel_size=enc.patch_len, stride=enc.patch_len
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: (B, 2, N_LR) — channel 0: x_LR, channel 1: d_map.

        Returns:
            patch_seq: (B, d_model, P) — patchified CNN output.
            skips:     list of L tensors, each (B, d_model, N_LR).
        """
        e = self.initial_conv(x)     # (B, d_model, N_LR)
        skips: list[torch.Tensor] = []
        for block in self.blocks:
            e = block(e)
            skips.append(e)          # save every intermediate output
        patch_seq = self.patch_conv(e)   # (B, d_model, P)
        return patch_seq, skips


# =============================================================================
# Transformer Encoder
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017).

    Not learnable.  Stored as a buffer so it moves with the model to GPU/CPU.
    Position 0 is the CLS/f_emb token; positions 1…P are CNN patches.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length (P+1 in normal operation).
        dropout: Dropout applied after adding encodings.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.drop = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()          # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)."""
        return self.drop(x + self.pe[:, : x.size(1), :])


class _PreLNTransformerLayer(nn.Module):
    """Single Pre-LayerNorm Transformer layer with independent attn / FFN dropout.

    Standard Post-LN applies LN after the residual add.  Pre-LN (used here)
    applies LN before each sub-layer, improving gradient flow for deep stacks.

        x = x + drop1(attn(LN1(x)))
        x = x + drop2(ffn(LN2(x)))

    Separate attn_dropout (on attention weights) and dropout (on sub-layer outputs)
    allow the config to set attn_dropout=0 while keeping residual dropout active.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        attn_dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model). Bidirectional (no causal mask)."""
        # Pre-LN self-attention
        h = self.norm1(x)
        h, _ = self.self_attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop1(h)
        # Pre-LN feed-forward
        h = self.norm2(x)
        h = self.linear2(F.gelu(self.linear1(h)))
        x = x + self.drop2(h)
        return x


class TransformerEncoder(nn.Module):
    """Stacked Pre-LN Transformer, bidirectional.

    Input sequence layout:
        position 0    → f_emb (CLS / physiological conditioning token)
        positions 1:P → CNN patch embeddings

    Outputs:
        z_seq:     (B, P+1, d_model)  full sequence
        z_cls:     (B, d_model)       position-0 output (global repr)
        z_patches: (B, P, d_model)    positions 1: (local patch reprs)

    Args:
        d_model: Embedding dimension.
        cfg: model subcfg (field: transformer with n_layers, n_heads,
             ffn_multiplier, dropout, attn_dropout).
        max_len: Maximum sequence length for positional encoding.
    """

    def __init__(self, d_model: int, cfg: DictConfig, max_len: int = 512) -> None:
        super().__init__()
        t = cfg.transformer
        ffn_dim = d_model * t.ffn_multiplier

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=t.dropout)
        self.layers = nn.ModuleList([
            _PreLNTransformerLayer(d_model, t.n_heads, ffn_dim, t.dropout, t.attn_dropout)
            for _ in range(t.n_layers)
        ])
        # Final norm: stabilises output before downstream modules
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            seq: (B, P+1, d_model) — CLS at index 0, patches at 1:P+1.

        Returns:
            z_seq:     (B, P+1, d_model)
            z_cls:     (B, d_model)
            z_patches: (B, P, d_model)
        """
        x = self.pos_enc(seq)
        for layer in self.layers:
            x = layer(x)
        z_seq = self.final_norm(x)
        z_cls     = z_seq[:, 0, :]    # (B, d_model) — off-by-one guard: 0 is CLS
        z_patches = z_seq[:, 1:, :]   # (B, P, d_model)
        return z_seq, z_cls, z_patches


# =============================================================================
# Top-level ECGEncoder
# =============================================================================

class ECGEncoder(nn.Module):
    """Top-level encoder: X_in + f_signal + f_meta → latent representations.

    Orchestration:
      1. FeatureProjectionMLP  f_signal, f_meta → f_emb (B, d_model)
      2. CNNEncoder            X_in → patch_seq (B, d_model, P) + skip list
      3. Build sequence        cat([f_emb, patches]) → (B, P+1, d_model)
      4. TransformerEncoder    → z_seq, z_cls, z_patches

    Input shapes:
        X_in:     (B, 2, N_LR)   channel 0 = x_LR, channel 1 = d_map
        f_signal: (B, 12)        physiological features, computed from x_LR only
        f_meta:   (B, 7)         patient metadata (may be all-zero via dropout)

    Output shapes:
        z_seq:     (B, P+1, d_model)
        z_cls:     (B, d_model)
        z_patches: (B, P, d_model)
        skips:     list of L tensors each (B, d_model, N_LR)

    where P = N_LR // patch_len  (default: 1000 // 10 = 100)
          L = n_residual_blocks   (default: 6)

    Args:
        cfg: Full OmegaConf config (top-level, contains cfg.model subcfg).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        mcfg = cfg.model
        d_model = mcfg.d_model

        # Sanity check: MLP output dims must sum to d_model
        signal_out = mcfg.f_signal_mlp.output_dim
        meta_out   = mcfg.f_meta_mlp.output_dim
        if signal_out + meta_out != d_model:
            raise ValueError(
                f"f_signal_mlp.output_dim ({signal_out}) + "
                f"f_meta_mlp.output_dim ({meta_out}) = {signal_out + meta_out}, "
                f"must equal d_model ({d_model})."
            )

        n_patches = L_LR // mcfg.encoder.patch_len   # 1000 // 10 = 100
        max_len   = n_patches + 1                     # +1 for CLS token

        self.feature_mlp  = FeatureProjectionMLP(mcfg)
        self.cnn_encoder  = CNNEncoder(d_model, mcfg)
        self.transformer  = TransformerEncoder(d_model, mcfg, max_len=max_len)

    def forward(
        self,
        X_in: torch.Tensor,      # (B, 2, N_LR)
        f_signal: torch.Tensor,  # (B, F_SIGNAL_DIM=12)
        f_meta: torch.Tensor,    # (B, F_META_DIM=7)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Returns:
            z_seq:     (B, P+1, d_model)
            z_cls:     (B, d_model)
            z_patches: (B, P, d_model)
            skips:     list of L tensors, each (B, d_model, N_LR)
        """
        # Step 1: conditioning embedding → CLS token
        f_emb = self.feature_mlp(f_signal, f_meta)        # (B, d_model)

        # Step 2: CNN → patchified features + skip connections
        patch_seq, skips = self.cnn_encoder(X_in)          # (B, d_model, P), [...]

        # Step 3: build Transformer input sequence
        # patch_seq (B, d_model, P) → (B, P, d_model) to match sequence convention
        patches   = patch_seq.transpose(1, 2)              # (B, P, d_model)
        cls_token = f_emb.unsqueeze(1)                     # (B, 1, d_model)
        seq       = torch.cat([cls_token, patches], dim=1) # (B, P+1, d_model)

        # Step 4: Transformer encoding
        z_seq, z_cls, z_patches = self.transformer(seq)

        return z_seq, z_cls, z_patches, skips