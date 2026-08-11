#!/usr/bin/env python3
"""
PyTorch Self-Supervised 1D-SimCLR Model Training Script.
Optimized for NVIDIA RTX 5070 (CUDA AMP FP16).
"""

import argparse
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import yaml

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from simclr_model import SimCLR1DModel
from loss_infonce import InfoNCELoss
from dataset_simclr import NILMSimCLRDataset
from augmentations_1d import SimCLRAugmentationPipeline

# Fix Windows Intel OpenMP library duplication error (OMP: Error #15)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    parser = argparse.ArgumentParser(description="Train 1D-SimCLR for NILM AI (Self-Supervised Contrastive Learning).")
    parser.add_argument("-c", "--config", type=str, default="./config_simclr.yaml", help="Path to YAML config file")
    parser.add_argument("-e", "--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("-b", "--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("-lr", "--learning-rate", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    epochs = args.epochs or config["training"]["epochs"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    lr = args.learning_rate or config["training"]["learning_rate"]
    output_dir = config["paths"]["output_dir"]
    dataset_path = config["paths"]["dataset"]
    scaler_path = config["paths"]["scaler"]

    os.makedirs(output_dir, exist_ok=True)

    # CUDA Device Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print(" 1D-SimCLR Self-Supervised Pre-Training")
    print("==================================================")
    print(f"Device        : {device}")
    if device.type == "cuda":
        print(f"GPU Model     : {torch.cuda.get_device_name(0)}")
        print(f"VRAM Capacity : {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")
    print(f"Dataset Path  : {dataset_path}")
    print(f"Batch Size    : {batch_size}")
    print(f"Epochs        : {epochs}")
    print(f"Temperature   : {config['simclr']['temperature']}")
    print("--------------------------------------------------")

    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset file not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Load Dataset
    print("[1/4] Loading CSV dataset into memory...")
    t0 = time.time()
    df = pd.read_csv(dataset_path).fillna(0.0)
    print(f"  - Loaded {len(df):,} cycles ({round(len(df)/60/60, 2)} hours) in {round(time.time()-t0, 2)}s")

    ignore_cols = ["global_cycle", "t_s"]
    target_cols = [c for c in df.columns if c.startswith("p_w_") and not c.endswith("_agg") and not c.endswith("_smooth")]
    input_cols = [c for c in df.columns if c not in ignore_cols and c not in target_cols and not c.startswith("state_") and not c.startswith("irms_")]

    X_raw = np.nan_to_num(df[input_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y_power_raw = np.nan_to_num(df[target_cols].values, nan=0.0, posinf=0.0, neginf=0.0)

    # Load or fit feature scaler
    if os.path.exists(scaler_path):
        print(f"[2/4] Loading existing scaler from {scaler_path}...")
        with open(scaler_path, "rb") as f:
            scaler_data = pickle.load(f)
            scaler = scaler_data["scaler"]
            feature_boost = scaler_data.get("feature_boost_weights", np.ones(len(input_cols), dtype=np.float32))
            X_scaled = scaler.transform(X_raw) * feature_boost
    else:
        print("[2/4] Fitting new StandardScaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        feature_boost = np.ones(len(input_cols), dtype=np.float32)
        for idx, col in enumerate(input_cols):
            if 'ih' in col or 'q_var' in col or 'phase' in col or 'power_factor' in col:
                feature_boost[idx] = 2.5
        X_scaled = X_scaled * feature_boost

    # Augmentation Pipeline setup
    aug_config = config.get("augmentations", {})
    aug_pipeline = SimCLRAugmentationPipeline(
        voltage_jitter_pct=aug_config.get("voltage_jitter_pct", 0.03),
        noise_snr_db=aug_config.get("noise_snr_db", 30.0),
        max_shift_cycles=3,
        max_crop_pct=aug_config.get("max_crop_pct", 0.15),
        harmonic_jitter_pct=aug_config.get("harmonic_jitter_pct", 0.05)
    )

    dataset = NILMSimCLRDataset(
        X_scaled, 
        y_power_raw, 
        window_len=config["simclr"]["window_len"], 
        stride=config["simclr"]["stride"],
        augmentation_pipeline=aug_pipeline
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )

    print(f"  - Sliding Window Samples : {len(dataset):,}")
    print(f"  - Total Batches / Epoch  : {len(dataloader):,}")

    # Initialize Model, Loss, Optimizer
    model = SimCLR1DModel(
        in_channels=len(input_cols),
        embedding_dim=config["simclr"]["embedding_dim"],
        projection_dim=config["simclr"]["projection_dim"]
    ).to(device)

    criterion = InfoNCELoss(temperature=config["simclr"]["temperature"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["training"].get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and config["training"].get("use_amp", True)))

    # Training Loop
    print("[3/4] Starting 1D-SimCLR Self-Supervised Pre-Training...")
    loss_history = []
    best_loss = float("inf")
    best_model_path = os.path.join(output_dir, "best_simclr_encoder.pth")

    start_train_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{epochs:02d} [SimCLR]")

        for x_A_b, x_B_b, _ in pbar:
            x_A_b, x_B_b = x_A_b.to(device), x_B_b.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and config["training"].get("use_amp", True))):
                _, z_A = model(x_A_b)
                _, z_B = model(x_B_b)
                loss = criterion(z_A, z_B)

            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

            running_loss += loss.item() * len(x_A_b)
            pbar.set_postfix({"InfoNCE Loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / (len(dataloader) * batch_size)
        loss_history.append(epoch_loss)
        scheduler.step()

        print(f"  --> Epoch {epoch:02d} - InfoNCE Loss: {epoch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
                "input_cols": input_cols,
                "target_cols": target_cols
            }, best_model_path)
            print(f"      [Checkpoint Saved] best_simclr_encoder.pth (Loss: {best_loss:.4f})")

    total_time = round(time.time() - start_train_time, 2)
    print(f"\n[4/4] 1D-SimCLR Pre-Training Complete! Total Duration: {total_time}s ({round(total_time/60, 2)} min)")

    # Save Loss Curve
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, label="InfoNCE Loss", color="purple", linewidth=2)
        plt.title("1D-SimCLR Self-Supervised Contrastive Learning Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("InfoNCE (NT-Xent) Loss")
        plt.legend()
        plt.grid(True)
        loss_curve_path = os.path.join(output_dir, "simclr_loss_curve.png")
        plt.savefig(loss_curve_path)
        plt.close()
        print(f"  - Saved Training Loss Curve to: {loss_curve_path}")

    print(f"  - Saved Best SimCLR Encoder Checkpoint to: {best_model_path}")
    print("==================================================")


if __name__ == "__main__":
    main()
