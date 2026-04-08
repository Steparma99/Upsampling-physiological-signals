"""
NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for contrastive
pre-training, following the SimCLR formulation.

Reference: Chen et al., "A Simple Framework for Contrastive Learning of
Visual Representations", ICML 2020.
"""

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
    """
    Compute the NT-Xent loss for a batch of N positive pairs.

    Args:
        z1:          (B, proj_dim) — first augmented view embeddings.
        z2:          (B, proj_dim) — second augmented view embeddings.
        temperature: Softmax temperature τ (default 0.07).

    Returns:
        loss: Scalar tensor.

    Algorithm:
      1. L2-normalise z1, z2.
      2. Concatenate into z of shape (2B, proj_dim).
      3. Build cosine similarity matrix sim = z @ z.T / τ  (2B, 2B).
      4. Mask the diagonal (self-similarity → −∞).
      5. Labels: for row i the positive is at i+B (i<B) and i−B (i≥B).
      6. Cross-entropy over the 2B rows.

    Note: F.cross_entropy applies log-softmax internally — do NOT pre-apply
    softmax to sim.
    """
    B = z1.shape[0]

    # 1. L2 normalisation
    z1 = F.normalize(z1, dim=-1)   # (B, proj_dim)
    z2 = F.normalize(z2, dim=-1)   # (B, proj_dim)

    # 2. Concatenate
    z = torch.cat([z1, z2], dim=0)  # (2B, proj_dim)

    # 3. Cosine similarity matrix (dot product of unit vectors)
    sim = torch.mm(z, z.T) / temperature  # (2B, 2B)

    # 4. Mask diagonal (exclude self-similarity)
    diag_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(diag_mask, -1e9)

    # 5. Positive-pair labels
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),   # positives of z1[i] are z2[i] = row i+B
        torch.arange(0,     B, device=z.device),    # positives of z2[i] are z1[i] = row i
    ])  # (2B,)

    # 6. Cross-entropy (includes log-softmax)
    return F.cross_entropy(sim, labels)
