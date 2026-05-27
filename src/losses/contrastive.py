import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NTXentLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for SimCLR.

    For a batch of N signals with 2N augmented views, computes the symmetric
    contrastive loss. L2 normalisation is applied here, not in the model.

    Args:
        temperature: Softmax temperature τ (config: phase_a.simclr.temperature)
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature
        self._criterion = nn.CrossEntropyLoss()

    def forward(self, h_a: Tensor, h_b: Tensor) -> Tensor:
        """
        Args:
            h_a: (N, D) — projection head output for view a, raw (not normalised)
            h_b: (N, D) — projection head output for view b, raw (not normalised)

        Returns:
            Scalar NT-Xent loss.
        """
        N = h_a.shape[0]
        device = h_a.device

        # L2 normalise inside the loss — CLAUDE.md constraint #3
        h_a = F.normalize(h_a, dim=-1)  # (N, D)
        h_b = F.normalize(h_b, dim=-1)  # (N, D)

        # Stack both views: (2N, D)
        h = torch.cat([h_a, h_b], dim=0)

        # Cosine similarity matrix scaled by temperature: (2N, 2N)
        sim = torch.mm(h, h.T) / self.temperature

        # Exclude self-similarity (diagonal) from the softmax denominator
        eye = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(eye, float("-inf"))

        # Positive for row i in [0, N) is at column i+N; for [N, 2N) is at i-N
        labels = torch.cat([
            torch.arange(N, 2 * N, device=device),
            torch.arange(0, N, device=device),
        ])  # (2N,)

        # CrossEntropyLoss averages over 2N rows → equivalent to the (1/2N) factor
        return self._criterion(sim, labels)
