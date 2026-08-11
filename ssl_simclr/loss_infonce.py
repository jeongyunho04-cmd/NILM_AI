#!/usr/bin/env python3
"""
InfoNCE (NT-Xent: Normalized Temperature-scaled Cross Entropy) Loss Module.
Calculates contrastive loss over a batch of L2-normalized projection vectors (z_A, z_B).
Pulls positive pairs together (z_A <-> z_B) while pushing negative pairs apart (z_A <-> z_neg).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_A: torch.Tensor, z_B: torch.Tensor) -> torch.Tensor:
        """
        Args:
          z_A: (B, D) projection vectors for view A (L2 normalized)
          z_B: (B, D) projection vectors for view B (L2 normalized)
        Returns:
          loss: scalar tensor
        """
        batch_size = z_A.shape[0]
        device = z_A.device

        # Concatenate both views: shape (2*B, D)
        z = torch.cat([z_A, z_B], dim=0)

        # Compute Pairwise Cosine Similarity Matrix: shape (2*B, 2*B)
        # Since z is L2 normalized, z @ z.T is cosine similarity
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Create self-contrast mask (exclude diagonal i == j)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        # Positive pair targets:
        # For item i in [0, B-1], target is i + B
        # For item i in [B, 2B-1], target is i - B
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device)
        ], dim=0)

        # Compute Cross Entropy Loss over softmax normalized similarities
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
