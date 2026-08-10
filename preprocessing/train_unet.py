#!/usr/bin/env python3
"""
PyTorch 1D-UNet (Sequence-to-Sequence) NILM AI Disaggregation Training Script.
Supports GPU acceleration (NVIDIA RTX 5070 / CUDA) with Mixed Precision FP16.
"""

import argparse
import os
import sys
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
# 1. PyTorch 1D-UNet Model Architecture
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


class UNet1D(nn.Module):
    """
    1D U-Net Sequence-to-Sequence Architecture for NILM Disaggregation.
    Preserves transient surges and sharp switching edges via Skip Connections.
    """
    def __init__(self, in_channels=51, out_channels=6):
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

        # Decoder (Expanding Path with Skip Connections)
        self.up3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv1D(512, 256)

        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv1D(256, 128)

        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv1D(128, 64)

        # Final Output Layer (Power prediction >= 0.0W)
        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder
        d3 = self.up3(b)
        # Pad if needed due to odd sequence lengths
        if d3.shape[-1] != e3.shape[-1]:
            d3 = F.pad(d3, (0, e3.shape[-1] - d3.shape[-1]))
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.shape[-1] != e2.shape[-1]:
            d2 = F.pad(d2, (0, e2.shape[-1] - d2.shape[-1]))
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape[-1] != e1.shape[-1]:
            d1 = F.pad(d1, (0, e1.shape[-1] - d1.shape[-1]))
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Final Output Layer (ReLU ensures non-negative active power predictions)
        out = F.relu(self.final_conv(d1))
        return out


# =====================================================================
# 2. PyTorch Dataset & Sliding Window Loader
# =====================================================================
class NILMSlidingWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, window_len: int = 320, stride: int = 32):
        """
        X shape: (N_samples, num_features)
        y shape: (N_samples, num_appliances)
        """
        self.window_len = window_len
        self.stride = stride

        # Pre-compute valid sliding window start indices
        n_samples = len(X)
        self.indices = list(range(0, n_samples - window_len + 1, stride))

        # Convert arrays to Float Tensors
        self.X_data = torch.tensor(X, dtype=torch.float32)
        self.y_data = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.window_len

        # Shape: (features, window_len) for 1D Conv
        x_win = self.X_data[start_idx:end_idx].transpose(0, 1)
        # Shape: (appliances, window_len)
        y_win = self.y_data[start_idx:end_idx].transpose(0, 1)

        return x_win, y_win


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
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.001, help="Adam optimizer learning rate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # CUDA Device Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print(" NILM AI 1D-UNet Disaggregation Training")
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
    y_raw = df[target_cols].values

    # Fit & Apply Feature Scaler
    print("[2/4] Normalizing input features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    scaler_path = os.path.join(args.output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "input_cols": input_cols, "target_cols": target_cols}, f)

    # Train / Val Split (80% train, 20% validation)
    split_idx = int(len(X_scaled) * 0.8)
    X_train, y_train = X_scaled[:split_idx], y_raw[:split_idx]
    X_val, y_val = X_scaled[split_idx:], y_raw[split_idx:]

    train_dataset = NILMSlidingWindowDataset(X_train, y_train, window_len=args.window_len, stride=args.stride)
    val_dataset = NILMSlidingWindowDataset(X_val, y_val, window_len=args.window_len, stride=args.stride)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))

    print(f"  - Train Window Samples : {len(train_dataset):,}")
    print(f"  - Val Window Samples   : {len(val_dataset):,}")

    # 3. Model Initialization
    model = UNet1D(in_channels=len(input_cols), out_channels=len(target_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.SmoothL1Loss()  # Huber Loss (Smooth L1 Loss) robust against transient spikes
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 4. Training Loop
    print("[3/4] Starting 1D-UNet Model Training...")
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_model_path = os.path.join(args.output_dir, "best_unet_nilm.pth")

    start_train_time = time.time()
    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]")

        for x_b, y_b in pbar:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model(x_b)
                loss = criterion(pred, y_b)

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
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    pred = model(x_b)
                    loss = criterion(pred, y_b)
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
