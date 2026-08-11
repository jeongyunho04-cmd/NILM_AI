#!/usr/bin/env python3
"""
1D Waveform Data Augmentation Module for Self-Supervised SimCLR NILM AI.
Applies physical transformations to power/harmonic time series:
  1. VoltagePerturbation (Sag/Swell +/-3%)
  2. AdditiveNoise (Gaussian & Harmonic Jitter SNR ~30dB)
  3. PhaseShiftRoll (Micro time shift +/-3 deg)
  4. RandomTimeCropResize (Sub-window cropping & resampling)
  5. HarmonicJitter (Higher-order current harmonic perturbation)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VoltagePerturbation(nn.Module):
    """
    Simulates grid voltage fluctuations (sag/swell +/-3% to +/-5%).
    Applies small proportional scaling to active power and voltage RMS features.
    """
    def __init__(self, max_scale_pct: float = 0.03, vrms_idx: int = 2, p_agg_idx: int = 0):
        super().__init__()
        self.max_scale_pct = max_scale_pct
        self.vrms_idx = vrms_idx
        self.p_agg_idx = p_agg_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (C, L) or (B, C, L)
        """
        is_batched = (x.ndim == 3)
        if not is_batched:
            x = x.unsqueeze(0)  # (1, C, L)

        B, C, L = x.shape
        scale_factors = 1.0 + torch.randn(B, 1, 1, device=x.device) * (self.max_scale_pct / 2.0)
        scale_factors = torch.clamp(scale_factors, 1.0 - self.max_scale_pct, 1.0 + self.max_scale_pct)

        x_aug = x.clone()
        # Scale active power & voltage RMS
        if self.p_agg_idx < C:
            x_aug[:, self.p_agg_idx:self.p_agg_idx+1, :] = x[:, self.p_agg_idx:self.p_agg_idx+1, :] * scale_factors
        if self.vrms_idx < C:
            x_aug[:, self.vrms_idx:self.vrms_idx+1, :] = x[:, self.vrms_idx:self.vrms_idx+1, :] * scale_factors

        if not is_batched:
            x_aug = x_aug.squeeze(0)
        return x_aug


class AdditiveGaussianNoise(nn.Module):
    """
    Adds zero-mean Gaussian noise to waveform features with specified SNR (dB).
    """
    def __init__(self, snr_db: float = 30.0):
        super().__init__()
        self.snr_db = snr_db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_batched = (x.ndim == 3)
        if not is_batched:
            x = x.unsqueeze(0)

        signal_power = torch.mean(x ** 2, dim=(-2, -1), keepdim=True) + 1e-8
        noise_power = signal_power / (10 ** (self.snr_db / 10.0))
        noise_std = torch.sqrt(noise_power)
        noise = torch.randn_like(x) * noise_std

        x_aug = x + noise
        if not is_batched:
            x_aug = x_aug.squeeze(0)
        return x_aug


class PhaseShiftRoll(nn.Module):
    """
    Simulates micro phase shifts (+/- 2 to 5 degrees ~ 1 to 4 cycles roll).
    """
    def __init__(self, max_shift_cycles: int = 4):
        super().__init__()
        self.max_shift_cycles = max_shift_cycles

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_batched = (x.ndim == 3)
        if not is_batched:
            x = x.unsqueeze(0)

        B, C, L = x.shape
        x_aug = torch.zeros_like(x)
        for i in range(B):
            shift = torch.randint(-self.max_shift_cycles, self.max_shift_cycles + 1, (1,)).item()
            x_aug[i] = torch.roll(x[i], shifts=shift, dims=-1)

        if not is_batched:
            x_aug = x_aug.squeeze(0)
        return x_aug


class RandomTimeCropResize(nn.Module):
    """
    Randomly crops a time sub-window (up to max_crop_pct) and resamples back to target length L.
    """
    def __init__(self, max_crop_pct: float = 0.15):
        super().__init__()
        self.max_crop_pct = max_crop_pct

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_batched = (x.ndim == 3)
        if not is_batched:
            x = x.unsqueeze(0)

        B, C, L = x.shape
        crop_ratio = torch.empty(1).uniform_(1.0 - self.max_crop_pct, 1.0).item()
        crop_len = int(L * crop_ratio)
        if crop_len >= L:
            return x.squeeze(0) if not is_batched else x

        start_idx = torch.randint(0, L - crop_len + 1, (1,)).item()
        cropped = x[:, :, start_idx:start_idx + crop_len]

        # Resample back to original length L using 1D linear interpolation
        resized = F.interpolate(cropped, size=L, mode='linear', align_corners=False)

        if not is_batched:
            resized = resized.squeeze(0)
        return resized


class HarmonicJitter(nn.Module):
    """
    Applies random perturbation to current harmonic features (ih1~ih15).
    """
    def __init__(self, harmonic_start_idx: int = 5, harmonic_count: int = 15, jitter_pct: float = 0.05):
        super().__init__()
        self.start_idx = harmonic_start_idx
        self.end_idx = harmonic_start_idx + harmonic_count
        self.jitter_pct = jitter_pct

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_batched = (x.ndim == 3)
        if not is_batched:
            x = x.unsqueeze(0)

        B, C, L = x.shape
        end = min(self.end_idx, C)
        if self.start_idx < C and end > self.start_idx:
            jitter = 1.0 + torch.randn(B, end - self.start_idx, 1, device=x.device) * self.jitter_pct
            x_aug = x.clone()
            x_aug[:, self.start_idx:end, :] = x[:, self.start_idx:end, :] * jitter
        else:
            x_aug = x.clone()

        if not is_batched:
            x_aug = x_aug.squeeze(0)
        return x_aug


class SimCLRAugmentationPipeline(nn.Module):
    """
    Composite 1D Augmentation Pipeline.
    Given a waveform sequence x, produces two stochastic perturbed views (x_A, x_B).
    """
    def __init__(self, 
                 voltage_jitter_pct: float = 0.03,
                 noise_snr_db: float = 30.0,
                 max_shift_cycles: int = 3,
                 max_crop_pct: float = 0.15,
                 harmonic_jitter_pct: float = 0.05):
        super().__init__()
        self.voltage_aug = VoltagePerturbation(max_scale_pct=voltage_jitter_pct)
        self.noise_aug = AdditiveGaussianNoise(snr_db=noise_snr_db)
        self.phase_aug = PhaseShiftRoll(max_shift_cycles=max_shift_cycles)
        self.crop_aug = RandomTimeCropResize(max_crop_pct=max_crop_pct)
        self.harmonic_aug = HarmonicJitter(jitter_pct=harmonic_jitter_pct)

    def _apply_view(self, x: torch.Tensor) -> torch.Tensor:
        # Randomly choose subset of augmentations per view
        out = x
        if torch.rand(1).item() > 0.3:
            out = self.voltage_aug(out)
        if torch.rand(1).item() > 0.3:
            out = self.phase_aug(out)
        if torch.rand(1).item() > 0.4:
            out = self.crop_aug(out)
        if torch.rand(1).item() > 0.3:
            out = self.harmonic_aug(out)
        if torch.rand(1).item() > 0.2:
            out = self.noise_aug(out)
        return out

    def forward(self, x: torch.Tensor):
        x_A = self._apply_view(x)
        x_B = self._apply_view(x)
        return x_A, x_B
