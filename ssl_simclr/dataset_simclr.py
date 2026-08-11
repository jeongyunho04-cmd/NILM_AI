#!/usr/bin/env python3
"""
PyTorch Dataset for Self-Supervised 1D-SimCLR Training.
Generates stochastic augmented view pairs (x_A, x_B) on-the-fly for contrastive pre-training.
"""

from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from augmentations_1d import SimCLRAugmentationPipeline


class NILMSimCLRDataset(Dataset):
    """
    Sliding window dataset for 1D-SimCLR contrastive pre-training.
    Returns:
      x_A: (C, L) augmented view A
      x_B: (C, L) augmented view B
      appliance_label: scalar integer or vector (for downstream evaluation/visualization)
    """
    def __init__(self, 
                 X: np.ndarray, 
                 y_power: Optional[np.ndarray] = None,
                 window_len: int = 320, 
                 stride: int = 32,
                 augmentation_pipeline: Optional[SimCLRAugmentationPipeline] = None):
        """
        X shape: (N_samples, num_features)
        y_power shape: (N_samples, num_appliances)
        """
        self.window_len = window_len
        self.stride = stride

        n_samples = len(X)
        self.indices = list(range(0, n_samples - window_len + 1, stride))

        self.X_data = torch.tensor(X, dtype=torch.float32)
        if y_power is not None:
            self.y_power_data = torch.tensor(y_power, dtype=torch.float32)
        else:
            self.y_power_data = None

        self.aug_pipeline = augmentation_pipeline or SimCLRAugmentationPipeline()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start_idx = self.indices[idx]
        end_idx = start_idx + self.window_len

        # Shape: (C, L) where C = num_features, L = window_len
        x_win = self.X_data[start_idx:end_idx].transpose(0, 1)

        # Generate stochastic augmented pair (x_A, x_B)
        x_A, x_B = self.aug_pipeline(x_win)

        # Determine dominant active appliance label for cluster color tagging
        if self.y_power_data is not None:
            y_win = self.y_power_data[start_idx:end_idx]
            mean_powers = torch.mean(y_win, dim=0)
            dominant_app_idx = torch.argmax(mean_powers) if torch.max(mean_powers) >= 5.0 else -1
        else:
            dominant_app_idx = torch.tensor(-1)

        return x_A, x_B, dominant_app_idx
