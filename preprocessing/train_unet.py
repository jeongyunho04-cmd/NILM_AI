#!/usr/bin/env python3
"""
PyTorch 1D-UNet (Sequence-to-Sequence) NILM AI Disaggregation Training Script.
Supports GPU acceleration (NVIDIA RTX 5070 / CUDA) with Mixed Precision FP16.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

# Fix Windows Intel OpenMP library duplication error (OMP: Error #15)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import time

import pickle
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# =====================================================================
# 1. Attention Gate 1D & Multi-Task Attention 1D-UNet Model Architecture
# =====================================================================
class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionBlock1D(nn.Module):
    """
    1D Attention Gate mechanism for UNet Skip Connections.
    Allows the model to dynamically focus (attend) to specific frequency/harmonic features
    and transient peaks during multi-appliance disaggregation.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[-1] != x1.shape[-1]:
            g1 = F.pad(g1, (0, x1.shape[-1] - g1.shape[-1]))
        net = self.relu(g1 + x1)
        psi = self.psi(net)
        return x * psi


class AttentionMultiTaskUNet1D(nn.Module):
    """
    Attention Multi-Task 1D-UNet Architecture for NILM Disaggregation.
    Includes:
      1. 1D Attention Gates on Skip Connections
      2. Multi-Task Learning Heads (Power Regression + State Classification)
    """
    def __init__(self, in_channels=55, out_channels=6):
        super().__init__()

        # Encoder (Contracting Path)
        self.enc1 = DoubleConv1D(in_channels, 64)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = DoubleConv1D(64, 128)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = DoubleConv1D(128, 256)
        self.pool3 = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = DoubleConv1D(256, 512)

        # Attention Gates
        self.attn3 = AttentionBlock1D(F_g=256, F_l=256, F_int=128)
        self.attn2 = AttentionBlock1D(F_g=128, F_l=128, F_int=64)
        self.attn1 = AttentionBlock1D(F_g=64, F_l=64, F_int=32)

        # Decoder (Expanding Path with Attention-Weighted Skip Connections)
        self.up3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv1D(512, 256)

        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv1D(256, 128)

        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv1D(128, 64)

        # Multi-Task Output Heads
        # Head 1: Active Power Regression (W)
        self.power_head = nn.Conv1d(64, out_channels, kernel_size=1)
        # Head 2: Appliance ON/OFF State Classification (Raw Logits for BCEWithLogitsLoss)
        self.state_head = nn.Conv1d(64, out_channels, kernel_size=1)


    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder with Attention Gates
        d3 = self.up3(b)
        if d3.shape[-1] != e3.shape[-1]:
            d3 = F.pad(d3, (0, e3.shape[-1] - d3.shape[-1]))
        e3_attn = self.attn3(g=d3, x=e3)
        d3 = torch.cat([d3, e3_attn], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.shape[-1] != e2.shape[-1]:
            d2 = F.pad(d2, (0, e2.shape[-1] - d2.shape[-1]))
        e2_attn = self.attn2(g=d2, x=e2)
        d2 = torch.cat([d2, e2_attn], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape[-1] != e1.shape[-1]:
            d1 = F.pad(d1, (0, e1.shape[-1] - d1.shape[-1]))
        e1_attn = self.attn1(g=d1, x=e1)
        d1 = torch.cat([d1, e1_attn], dim=1)
        d1 = self.dec1(d1)

        # Multi-Task Outputs
        out_power = F.relu(self.power_head(d1))
        out_state = self.state_head(d1)

        return out_power, out_state


# Alias for backward compatibility
UNet1D = AttentionMultiTaskUNet1D



# =====================================================================
# 2. PyTorch Dataset & Sliding Window Loader
# =====================================================================
class NILMSlidingWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y_power: np.ndarray, y_state: Optional[np.ndarray] = None, window_len: int = 320, stride: int = 32):
        """
        X shape: (N_samples, num_features)
        y_power shape: (N_samples, num_appliances)
        y_state shape: (N_samples, num_appliances)
        """
        self.window_len = window_len
        self.stride = stride

        n_samples = len(X)
        self.indices = list(range(0, n_samples - window_len + 1, stride))

        self.X_data = torch.tensor(X, dtype=torch.float32)
        self.y_power_data = torch.tensor(y_power, dtype=torch.float32)
        
        if y_state is None:
            # Generate binary state label on the fly (1 if power >= 5.0W else 0)
            y_state = (y_power >= 5.0).astype(np.float32)
        self.y_state_data = torch.tensor(y_state, dtype=torch.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.window_len

        x_win = self.X_data[start_idx:end_idx].transpose(0, 1)
        y_p_win = self.y_power_data[start_idx:end_idx].transpose(0, 1)
        y_s_win = self.y_state_data[start_idx:end_idx].transpose(0, 1)

        return x_win, y_p_win, y_s_win


# =====================================================================
# 3. Main Training Function
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Train 1D-UNet for NILM AI Disaggregation.")
    parser.add_argument("-data", "--dataset", type=str, default="./output/synthetic_nilm_12h.csv", help="Synthetic CSV dataset path")
    parser.add_argument("-o", "--output-dir", type=str, default="./checkpoint_unet", help="Checkpoint save directory")
    parser.add_argument("-w", "--window-len", type=int, default=320, help="Sliding window length in cycles (default: 320 ~5.33s)")
    parser.add_argument("-stride", "--stride", type=int, default=32, help="Sliding window stride (default: 32 ~0.53s)")
    parser.add_argument("-b", "--batch-size", type=int, default=256, help="Batch size (optimized for RTX 5070: 256)")
    parser.add_argument("-e", "--epochs", type=int, default=35, help="Number of training epochs (default: 35 for high-precision micro load scaling)")
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.001, help="Adam optimizer learning rate")
    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)

    # CUDA Device Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print(" NILM AI Attention Multi-Task 1D-UNet Training")
    print("==================================================")
    print(f"Device        : {device}")
    if device.type == "cuda":
        print(f"GPU Model     : {torch.cuda.get_device_name(0)}")
        print(f"VRAM Capacity : {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")
    print(f"Dataset Path  : {args.dataset}")
    print(f"Window Length : {args.window_len} cycles ({round(args.window_len/60, 2)}s)")
    print(f"Batch Size    : {args.batch_size}")
    print(f"Epochs        : {args.epochs}")
    print("--------------------------------------------------")

    if not os.path.exists(args.dataset):
        print(f"[ERROR] Dataset file not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    # 1. Load Dataset
    print("[1/4] Loading CSV dataset into memory...")
    t0 = time.time()
    df = pd.read_csv(args.dataset)
    print(f"  - Loaded {len(df):,} cycles ({round(len(df)/60/60, 2)} hours) in {round(time.time()-t0, 2)}s")

    # Separate Input Features (X) and Ground Truth Targets (y)
    ignore_cols = ["global_cycle", "t_s"]
    target_cols = [c for c in df.columns if c.startswith("p_w_") and not c.endswith("_agg") and not c.endswith("_smooth")]
    input_cols = [c for c in df.columns if c not in ignore_cols and c not in target_cols and not c.startswith("state_") and not c.startswith("irms_")]

    print(f"  - Input Features ({len(input_cols)}): {input_cols[:4]}...")
    print(f"  - Target Appliances ({len(target_cols)}): {target_cols}")

    X_raw = df[input_cols].values
    y_power_raw = df[target_cols].values
    
    # Extract state targets if present
    state_cols = [c.replace("p_w_", "state_") for c in target_cols]
    if all(c in df.columns for c in state_cols):
        y_state_raw = (df[state_cols].values > 0).astype(np.float32)
    else:
        y_state_raw = (y_power_raw >= 5.0).astype(np.float32)

    # Fit & Apply Feature Scaler
    print("[2/4] Normalizing input features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    scaler_path = os.path.join(args.output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "input_cols": input_cols, "target_cols": target_cols}, f)

    # Train / Val Split (80% train, 20% validation)
    split_idx = int(len(X_scaled) * 0.8)
    X_train, y_p_train, y_s_train = X_scaled[:split_idx], y_power_raw[:split_idx], y_state_raw[:split_idx]
    X_val, y_p_val, y_s_val = X_scaled[split_idx:], y_power_raw[split_idx:], y_state_raw[split_idx:]

    train_dataset = NILMSlidingWindowDataset(X_train, y_p_train, y_s_train, window_len=args.window_len, stride=args.stride)
    val_dataset = NILMSlidingWindowDataset(X_val, y_p_val, y_s_val, window_len=args.window_len, stride=args.stride)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))

    print(f"  - Train Window Samples : {len(train_dataset):,}")
    print(f"  - Val Window Samples   : {len(val_dataset):,}")

    # 3. Model Initialization
    # Pre-compute appliance weights to boost small loads (10W~50W) like minipc/chargers
    max_powers = np.max(y_power_raw, axis=0)  # Max power per appliance
    # Small load weighting: min power appliances get up to 3.0x weight factor!
    weights_np = np.where(max_powers < 50.0, 3.0, np.where(max_powers < 100.0, 2.0, 1.0))
    app_weights = torch.tensor(weights_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(2)

    print(f"  - Appliance Loss Weighting Factors: {dict(zip([c.replace('p_w_', '') for c in target_cols], weights_np.tolist()))}")

    model = AttentionMultiTaskUNet1D(in_channels=len(input_cols), out_channels=len(target_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion_power_none = nn.SmoothL1Loss(reduction="none") # Huber Loss without reduction for weighting
    criterion_power_log = nn.SmoothL1Loss()                  # Log-scale loss log1p(y) for micro load scale sensitivity
    criterion_state = nn.BCEWithLogitsLoss()                  # Safe for Mixed Precision FP16 autocast!
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 4. Training Loop
    print("[3/4] Starting Attention Multi-Task 1D-UNet Model Training...")
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_model_path = os.path.join(args.output_dir, "best_unet_nilm.pth")

    start_train_time = time.time()
    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]")

        for x_b, y_p_b, y_s_b in pbar:
            x_b, y_p_b, y_s_b = x_b.to(device), y_p_b.to(device), y_s_b.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred_p, pred_s = model(x_b)

                # 1. Weighted Linear Power Loss
                raw_power_err = criterion_power_none(pred_p, y_p_b)
                loss_p_weighted = (raw_power_err * app_weights).mean()

                # 2. Log-Scale Power Loss log(1 + y) for micro load resolution
                pred_p_log = torch.log1p(pred_p)
                y_p_log = torch.log1p(y_p_b)
                loss_p_log = criterion_power_log(pred_p_log, y_p_log)

                loss_p = loss_p_weighted + 2.0 * loss_p_log
                loss_s = criterion_state(pred_s, y_s_b)

                # Combined Multi-Task Joint Loss
                loss = loss_p + 0.5 * loss_s

            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

            running_train_loss += loss.item() * len(x_b)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for x_b, y_p_b, y_s_b in val_loader:
                x_b, y_p_b, y_s_b = x_b.to(device), y_p_b.to(device), y_s_b.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    pred_p, pred_s = model(x_b)
                    raw_power_err = criterion_power_none(pred_p, y_p_b)
                    loss_p_weighted = (raw_power_err * app_weights).mean()

                    pred_p_log = torch.log1p(pred_p)
                    y_p_log = torch.log1p(y_p_b)
                    loss_p_log = criterion_power_log(pred_p_log, y_p_log)

                    loss_p = loss_p_weighted + 2.0 * loss_p_log
                    loss_s = criterion_state(pred_s, y_s_b)
                    loss = loss_p + 0.5 * loss_s

                running_val_loss += loss.item() * len(x_b)


        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)

        print(f"  --> Epoch {epoch:02d} - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        # Save Best Model Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": best_val_loss}, best_model_path)
            print(f"      [Checkpoint Saved] best_unet_nilm.pth (Val Loss: {best_val_loss:.4f})")


    total_time = round(time.time() - start_train_time, 2)
    print(f"\n[4/4] Training Complete! Total Duration: {total_time}s ({round(total_time/60, 2)} min)")

    # Plot Loss Curve if matplotlib is available
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss", linewidth=2)
        plt.plot(val_losses, label="Validation Loss", linewidth=2)
        plt.title("NILM 1D-UNet Training & Validation Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Smooth L1 Loss")
        plt.legend()
        plt.grid(True)
        loss_curve_path = os.path.join(args.output_dir, "loss_curve.png")
        plt.savefig(loss_curve_path)
        plt.close()
        print(f"  - Saved Training Loss Curve to : {loss_curve_path}")

    print(f"  - Saved Best Model Checkpoint to: {best_model_path}")
    print("==================================================")


if __name__ == "__main__":
    main()
