"""
Projection modules for Phase 2 pre-training.

PhysioProjector  — maps the 13-dim physiological feature vector f into d_model
                   space so it can serve as the conditioning token (CLS) for
                   the Transformer stack.

ProjectionHead   — lightweight MLP with BatchNorm used only during contrastive
                   pre-training (NT-Xent).  Discarded before Phase 3.
"""

import torch
import torch.nn as nn


class PhysioProjector(nn.Module):
    """
    Projects f ∈ R^f_dim → R^d_model via a two-layer MLP with GELU.

    Architecture:
        Linear(f_dim, hidden_dim) → GELU → Linear(hidden_dim, d_model)

    IMPORTANT: the caller must normalise f before passing it here.
    Normalisation is NOT performed inside this module — it is applied in the
    DataLoader using f_stats.npz (mean, std from Phase 1 training split).
    Only f[12] (Pol_T, already ±1) must NOT be z-scored.

    Input:  (B, f_dim)
    Output: (B, d_model)
    """

    def __init__(self, f_dim: int, hidden_dim: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(f_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # f: (B, f_dim) → (B, d_model)
        return self.net(f)


class ProjectionHead(nn.Module):
    """
    MLP projection head used ONLY during contrastive pre-training.
    Discarded after pre-training (not saved in encoder_phase3.pt).

    Architecture (dims = [256, 128, 64]):
        Linear(256, 128) → BatchNorm1d(128) → ReLU → Linear(128, 64)

    Design notes:
    - BatchNorm1d (not LayerNorm): original SimCLR choice; stabilises
      contrastive training by centring representations across the batch.
    - L2 normalisation is NOT applied inside this module — it is applied
      in nt_xent_loss before computing cosine similarities.

    Input:  (B, d_model)   — z_cls from HybridEncoder
    Output: (B, proj_out)  — proj_out = dims[-1] = 64
    """

    def __init__(self, dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:                           # all but last layer
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, d_model) → (B, proj_out)
        return self.net(z)
