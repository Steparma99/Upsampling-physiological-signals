"""
HybridEncoder — the core representation model for ECG upsampling pre-training.

Architecture summary:
  f (13-dim physio features)
    └─ PhysioProjector (MLP)
         └─ f_emb (d_model)                         ← conditioning token / CLS
  x_lr (L=1000 LR samples)
    └─ CNNEncoder (multi-scale blocks + strided conv)
         └─ patch_tokens (num_patches=20, d_model)
  [f_emb | patch_tokens]                            ← concatenate (21 tokens)
    └─ SinusoidalPE                                 ← add positional encoding
    └─ TransformerEncoder (Pre-LN, bidirectional)
         ├─ z_seq (B, 21, d_model)                  ← full sequence
         └─ z_cls (B, d_model)  = z_seq[:, 0, :]   ← physiological token (used as global rep)

Design rationale:
  - Using f_emb as the CLS token (Opzione B from spec) conditions the entire
    Transformer stack on patient physiology without adding an extra learnable
    token; z_cls is therefore directly interpretable as a physiology-conditioned
    global representation.
  - Pre-LN (norm_first=True) is more stable than Post-LN for large-scale
    self-supervised pre-training (same finding as in BERT and MAE papers).
  - Multi-scale CNN branches (k=3,7,15) capture local QRS morphology (narrow),
    P/T waves (medium), and inter-beat structure (wide) simultaneously.
"""

import math

import torch
import torch.nn as nn

from .projector import PhysioProjector


# ---------------------------------------------------------------------------
# Multi-scale CNN block
# ---------------------------------------------------------------------------

class MultiScaleCNNBlock(nn.Module):
    """
    Parallel multi-scale convolution block with residual connection.

    Three branches with different kernel sizes are applied in parallel,
    concatenated, and projected back to d_model via a 1×1 conv.  A skip
    connection (Pre-LN style) is applied before the activation.

    Input:  (B, d_model, L)
    Output: (B, d_model, L)   — same shape

    Note: branch_dim = d_model // len(kernel_sizes) may not divide evenly
    (e.g. 256//3 = 85).  concat_dim = 85·3 = 255 ≠ 256.  The pointwise conv
    maps concat_dim → d_model cleanly without information loss.
    """

    def __init__(
        self,
        d_model: int,
        kernel_sizes: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        branch_dim = d_model // len(kernel_sizes)
        concat_dim = branch_dim * len(kernel_sizes)

        self.branches = nn.ModuleList([
            nn.Conv1d(d_model, branch_dim, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.pointwise = nn.Conv1d(concat_dim, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_model, L)
        branch_outs = [b(x) for b in self.branches]
        concat = torch.cat(branch_outs, dim=1)  # (B, concat_dim, L)
        pw = self.pointwise(concat)              # (B, d_model, L)
        residual = x + pw                        # (B, d_model, L)   skip connection

        # LayerNorm on last dim: transpose → norm → transpose back
        out = residual.transpose(1, 2)           # (B, L, d_model)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        return out.transpose(1, 2)               # (B, d_model, L)


# ---------------------------------------------------------------------------
# CNN encoder
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """
    Converts a raw LR ECG segment into a sequence of patch embeddings.

    Input:  (B, L)              L = 1000 (LR samples)
    Output: (B, num_patches, d_model)   num_patches = L // patch_len = 20

    Architecture:
      1. Stem conv: Conv1d(1, d_model, 7, padding=3)   → (B, d_model, L)
      2. N_conv × MultiScaleCNNBlock                   → (B, d_model, L)
      3. Strided conv: Conv1d(d_model, d_model,
                              kernel_size=patch_len,
                              stride=patch_len, padding=0)
                                                        → (B, d_model, num_patches)
      4. Transpose                                      → (B, num_patches, d_model)

    The strided conv in step 3 serves as non-overlapping patch embedding,
    analogous to the patch projection in Vision Transformers.
    With L=1000 and patch_len=50: (1000 − 50) / 50 + 1 = 20 patches exactly.
    """

    def __init__(
        self,
        d_model: int,
        num_blocks: int,
        kernel_sizes: list[int],
        dropout: float,
        patch_len: int,
    ) -> None:
        super().__init__()
        # Step 1: stem (single-channel ECG → d_model feature maps)
        self.stem = nn.Conv1d(1, d_model, kernel_size=7, padding=3)

        # Step 2: multi-scale residual blocks (preserve temporal resolution)
        self.blocks = nn.ModuleList([
            MultiScaleCNNBlock(d_model, kernel_sizes, dropout)
            for _ in range(num_blocks)
        ])

        # Step 3: patch tokenisation via strided convolution
        self.strided = nn.Conv1d(
            d_model, d_model,
            kernel_size=patch_len, stride=patch_len, padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) → add channel dim
        x = x.unsqueeze(1)              # (B, 1, L)
        x = self.stem(x)               # (B, d_model, L)
        for block in self.blocks:
            x = block(x)               # (B, d_model, L)
        x = self.strided(x)            # (B, d_model, num_patches)
        return x.transpose(1, 2)       # (B, num_patches, d_model)


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    """
    Fixed (non-learnable) sinusoidal positional encoding.

    Pre-computed and registered as a buffer; not updated during training.

    PE[pos, 2k]   = sin(pos / 10000^(2k / d_model))
    PE[pos, 2k+1] = cos(pos / 10000^(2k / d_model))

    Buffer shape: (1, max_seq_len, d_model) — broadcasts over batch.

    Input:  (B, seq_len, d_model)
    Output: (B, seq_len, d_model)   — input + PE
    """

    def __init__(self, d_model: int, max_seq_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        # Handle odd d_model: cos entries may be one fewer than sin entries
        pe[:, 1::2] = torch.cos(pos * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


# ---------------------------------------------------------------------------
# Transformer encoder stack
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """
    Stack of standard PyTorch TransformerEncoderLayers.

    Key choices:
    - batch_first=True  so tensors are (B, seq_len, d_model) throughout.
    - norm_first=True   (Pre-LN) for improved gradient flow and training
                        stability compared to Post-LN.
    - No causal mask    — bidirectional attention like BERT (the encoder
                          should see the full context including masked patches
                          when deciding how to fill them in).
    - activation='gelu' — GELU consistently outperforms ReLU in Transformer
                          language/vision encoders.

    Input:  (B, seq_len, d_model)
    Output: (B, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model) → (B, seq_len, d_model)
        return self.transformer(x)


# ---------------------------------------------------------------------------
# Hybrid encoder (full model)
# ---------------------------------------------------------------------------

class HybridEncoder(nn.Module):
    """
    Full hybrid encoder: CNN feature extraction + Transformer global context.

    The physiological feature vector f is projected to d_model and placed
    at position 0 (acting as the CLS / conditioning token).  It is part of
    the input sequence to the Transformer, so every patch attends to the
    physiology unconditionally.

    Input:
        x_lr: (B, L=1000)    — normalised LR ECG segment.
        f:    (B, f_dim=13)  — normalised physiological feature vector.
                               Normalisation must be applied externally
                               (see PhysioProjector docstring).

    Output:
        z_seq: (B, num_patches+1, d_model)  — full token sequence.
        z_cls: (B, d_model)                 — physio-conditioned global rep
                                              = z_seq[:, 0, :].
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config            # stored for export to Phase 3
        d_model = config.encoder.d_model

        self.physio_proj = PhysioProjector(
            f_dim=config.data.f_dim,
            hidden_dim=config.physio_projector.hidden_dim,
            d_model=d_model,
        )
        self.cnn = CNNEncoder(
            d_model=d_model,
            num_blocks=config.encoder.cnn_num_blocks,
            kernel_sizes=list(config.encoder.cnn_kernel_sizes),
            dropout=config.encoder.cnn_dropout,
            patch_len=config.data.patch_len,
        )
        self.pe = SinusoidalPE(d_model, config.encoder.tf_max_seq_len)
        self.transformer = TransformerEncoder(
            d_model=d_model,
            num_layers=config.encoder.tf_num_layers,
            num_heads=config.encoder.tf_num_heads,
            d_ff=config.encoder.tf_d_ff,
            dropout=config.encoder.tf_dropout,
        )

    def forward(
        self, x_lr: torch.Tensor, f: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x_lr: (B, L=1000)
        # f:    (B, f_dim=13)

        f_emb   = self.physio_proj(f)                                # (B, d_model)
        cnn_out = self.cnn(x_lr)                                     # (B, num_patches, d_model)

        # Prepend physio token (position 0 = conditioning/CLS token)
        tokens = torch.cat([f_emb.unsqueeze(1), cnn_out], dim=1)    # (B, 21, d_model)
        tokens = self.pe(tokens)                                     # (B, 21, d_model)

        z_seq = self.transformer(tokens)                             # (B, 21, d_model)
        z_cls = z_seq[:, 0, :]                                       # (B, d_model)

        return z_seq, z_cls
