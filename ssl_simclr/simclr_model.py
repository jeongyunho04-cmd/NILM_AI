#!/usr/bin/env python3
"""
1D-SimCLR Neural Network Architecture for NILM AI.
Consists of:
  1. Feature Encoder f(·): 1D Convolutional UNet-style Encoder backbone extracting h in R^512
  2. Projection Head g(·): 2-Layer MLP projecting representation h to z in R^128 (L2 Normalized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv1D(nn.Module):
    """Double 1D Convolutional Block with BatchNorm & LeakyReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SimCLREncoder1D(nn.Module):
    """
    1D Feature Encoder f(·).
    Converts (B, in_channels, L) waveform tensor into high-dimensional embedding vector h (B, 512).
    Compatible with UNet Contracting Path.
    """
    def __init__(self, in_channels: int = 55, feature_dim: int = 512):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # Encoder (Contracting Path)
        self.enc1 = DoubleConv1D(in_channels, 64)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = DoubleConv1D(64, 128)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = DoubleConv1D(128, 256)
        self.pool3 = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = DoubleConv1D(256, feature_dim)

        # Global Pooling to collapse temporal sequence length L -> 1
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, C, L)
        Output: (B, feature_dim) representation vector h
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        h = self.global_pool(b).squeeze(-1)  # (B, 512)
        return h


class ProjectionHead1D(nn.Module):
    """
    Projection Head g(·).
    Maps 512-dim representation vector h to 128-dim normalized projection space z.
    """
    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, in_dim)
        Output: (B, out_dim) L2-normalized projection vector z
        """
        z_raw = self.net(h)
        z = F.normalize(z_raw, p=2, dim=1)
        return z


class SimCLR1DModel(nn.Module):
    """
    Full 1D-SimCLR Model combining Feature Encoder f(·) and Projection Head g(·).
    """
    def __init__(self, in_channels: int = 55, embedding_dim: int = 512, projection_dim: int = 128):
        super().__init__()
        self.encoder = SimCLREncoder1D(in_channels=in_channels, feature_dim=embedding_dim)
        self.projection_head = ProjectionHead1D(in_dim=embedding_dim, hidden_dim=256, out_dim=projection_dim)

    def forward(self, x: torch.Tensor):
        """
        Returns:
          h: (B, embedding_dim) representation vector
          z: (B, projection_dim) L2-normalized projection vector
        """
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z
