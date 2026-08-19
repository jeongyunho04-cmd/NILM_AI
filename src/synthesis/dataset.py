"""
PyTorch-Compatible Dataset and Fast Batch Generator for NILM Synthetic Loads
Generates on-the-fly augmented multi-appliance composite windows for neural network training.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .segment_pool import SegmentPool
from .synthesizer import LoadSynthesizer, SyntheticLoadSample


class NILMBatchGenerator:
    """Fast on-the-fly batch generator for NILM AI model training."""

    def __init__(
        self,
        segment_pool: SegmentPool,
        window_size_cycles: int = 600,  # 10 seconds default window
        max_concurrent_appliances: int = 3,
        include_power_channels: bool = True,
        target_mode: str = "seq2point",  # "seq2point" (center target) or "seq2seq" (full window)
    ):
        self.synthesizer = LoadSynthesizer(segment_pool=segment_pool)
        self.window_size = window_size_cycles
        self.max_concurrent = max_concurrent_appliances
        self.include_power_channels = include_power_channels
        self.target_mode = target_mode
        self.appliance_list = sorted(self.synthesizer.known_appliances)
        self.app_to_idx = {app: i for i, app in enumerate(self.appliance_list)}

    def generate_single_sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generates a single synthetic window.

        Returns:
            X: (Channels, Window_Size) float32
                Channels: 15 Real + 15 Imag (+ 3: P, Q, V if include_power_channels=True)
            y_power: (Num_Appliances,) or (Num_Appliances, Window_Size) float32
            y_state: (Num_Appliances,) or (Num_Appliances, Window_Size) int16
            y_on: (Num_Appliances,) or (Num_Appliances, Window_Size) int8
        """
        sample: SyntheticLoadSample = self.synthesizer.synthesize_random_window(
            window_size_cycles=self.window_size,
            max_concurrent_appliances=self.max_concurrent,
        )

        # 1. Format X: (Channels, W)
        # Real harmonics (15, W), Imag harmonics (15, W)
        r_part = sample.harmonics_ri[:, :, 0].T  # (15, W)
        i_part = sample.harmonics_ri[:, :, 1].T  # (15, W)

        if self.include_power_channels:
            # P_total (W), Q_total (VAR), V_bus (V)
            p_chan = sample.power_features[:, 0:1].T  # (1, W)
            q_chan = sample.power_features[:, 1:2].T  # (1, W)
            v_chan = sample.power_features[:, 4:5].T  # (1, W)
            x_channels = np.concatenate([r_part, i_part, p_chan, q_chan, v_chan], axis=0).astype(np.float32)
        else:
            x_channels = np.concatenate([r_part, i_part], axis=0).astype(np.float32)

        # 2. Format Y per Appliance
        n_apps = len(self.appliance_list)
        mid_idx = self.window_size // 2

        if self.target_mode == "seq2point":
            y_power = np.zeros(n_apps, dtype=np.float32)
            y_state = np.zeros(n_apps, dtype=np.int16)
            y_on = np.zeros(n_apps, dtype=np.int8)

            for i, app in enumerate(self.appliance_list):
                y_power[i] = sample.gt_target_power_w[app][mid_idx]
                y_state[i] = sample.gt_state_id[app][mid_idx]
                y_on[i] = sample.gt_is_on[app][mid_idx]

        else:  # seq2seq
            y_power = np.zeros((n_apps, self.window_size), dtype=np.float32)
            y_state = np.zeros((n_apps, self.window_size), dtype=np.int16)
            y_on = np.zeros((n_apps, self.window_size), dtype=np.int8)

            for i, app in enumerate(self.appliance_list):
                y_power[i] = sample.gt_target_power_w[app]
                y_state[i] = sample.gt_state_id[app]
                y_on[i] = sample.gt_is_on[app]

        return x_channels, y_power, y_state, y_on

    def generate_batch(self, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generates a full training batch of size B.

        Returns:
            X: (B, Channels, Window_Size) float32
            Y_power: (B, Num_Appliances) or (B, Num_Appliances, W)
            Y_state: (B, Num_Appliances) or (B, Num_Appliances, W)
            Y_on: (B, Num_Appliances) or (B, Num_Appliances, W)
        """
        x_list, yp_list, ys_list, yo_list = [], [], [], []
        for _ in range(batch_size):
            x, yp, ys, yo = self.generate_single_sample()
            x_list.append(x)
            yp_list.append(yp)
            ys_list.append(ys)
            yo_list.append(yo)

        return (
            np.stack(x_list, axis=0),
            np.stack(yp_list, axis=0),
            np.stack(ys_list, axis=0),
            np.stack(yo_list, axis=0),
        )


# ── PyTorch Dataset Wrapper (if torch is available) ──────────────────────────
try:
    import torch
    from torch.utils.data import Dataset

    class NILMPyTorchDataset(Dataset):
        """PyTorch Dataset wrapper for NILM load synthesis."""

        def __init__(
            self,
            segment_pool: SegmentPool,
            epoch_size: int = 5000,
            window_size_cycles: int = 600,
            target_mode: str = "seq2point",
            include_power_channels: bool = True,
        ):
            self.generator = NILMBatchGenerator(
                segment_pool=segment_pool,
                window_size_cycles=window_size_cycles,
                target_mode=target_mode,
                include_power_channels=include_power_channels,
            )
            self.epoch_size = epoch_size

        def __len__(self) -> int:
            return self.epoch_size

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            x, yp, ys, yo = self.generator.generate_single_sample()
            return (
                torch.from_numpy(x),
                torch.from_numpy(yp),
                torch.from_numpy(ys.astype(np.int64)),
                torch.from_numpy(yo.astype(np.float32)),
            )

except ImportError:
    NILMPyTorchDataset = None
