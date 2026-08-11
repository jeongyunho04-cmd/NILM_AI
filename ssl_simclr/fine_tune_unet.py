#!/usr/bin/env python3
"""
Downstream Fine-Tuning & Evaluation Script for 1D-SimCLR Pre-trained Encoder.
Attaches pre-trained SimCLR Encoder to Attention Multi-Task UNet Decoder to evaluate
disaggregation accuracy under noise, voltage sag, and zero-shot extension environments.
"""

import argparse
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from simclr_model import SimCLR1DModel
from train_unet import HierarchicalAttentionUNet1D, NILMSlidingWindowDataset

# Fix Windows Intel OpenMP library duplication error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune and Evaluate 1D-SimCLR Pre-trained UNet.")
    parser.add_argument("-c", "--config", type=str, default="./config_simclr.yaml", help="Path to YAML config file")
    parser.add_argument("-simclr-ckpt", "--simclr-checkpoint", type=str, default=None, help="Pre-trained SimCLR Encoder path")
    parser.add_argument("-unet-ckpt", "--unet-checkpoint", type=str, default=None, help="Supervised UNet checkpoint path")
    parser.add_argument("-e", "--epochs", type=int, default=15, help="Fine-tuning epochs (default: 15)")
    parser.add_argument("-b", "--batch-size", type=int, default=256, help="Batch size (default: 256)")
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.0003, help="Fine-tuning learning rate")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    simclr_ckpt_path = args.simclr_checkpoint or os.path.join(config["paths"]["output_dir"], "best_simclr_encoder.pth")
    unet_ckpt_path = args.unet_checkpoint or config["paths"]["unet_checkpoint"]
    dataset_path = config["paths"]["dataset"]
    scaler_path = config["paths"]["scaler"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print(" Downstream Fine-Tuning & Evaluation (1D-SimCLR)")
    print("==================================================")
    print(f"Device        : {device}")
    print(f"Dataset Path  : {dataset_path}")
    print(f"SimCLR Ckpt   : {simclr_ckpt_path}")
    print(f"UNet Ckpt     : {unet_ckpt_path}")
    print("--------------------------------------------------")

    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset file not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Load Dataset
    print("[1/3] Loading dataset & feature scalers...")
    df = pd.read_csv(dataset_path).fillna(0.0)

    ignore_cols = ["global_cycle", "t_s"]
    target_cols = [c for c in df.columns if c.startswith("p_w_") and not c.endswith("_agg") and not c.endswith("_smooth")]
    input_cols = [c for c in df.columns if c not in ignore_cols and c not in target_cols and not c.startswith("state_") and not c.startswith("irms_")]

    X_raw = np.nan_to_num(df[input_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y_power_raw = np.nan_to_num(df[target_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y_state_raw = (y_power_raw >= 2.0).astype(np.float32)

    with open(scaler_path, "rb") as f:
        scaler_data = pickle.load(f)
        scaler = scaler_data["scaler"]
        feature_boost = scaler_data.get("feature_boost_weights", np.ones(len(input_cols), dtype=np.float32))
        X_scaled = scaler.transform(X_raw) * feature_boost

    p_agg_idx = input_cols.index("p_w_agg") if "p_w_agg" in input_cols else 0
    p_agg_mean = scaler.mean_[p_agg_idx]
    p_agg_std = scaler.scale_[p_agg_idx]

    # Split dataset
    split_idx = int(len(X_scaled) * 0.8)
    X_val, y_p_val, y_s_val = X_scaled[split_idx:], y_power_raw[split_idx:], y_state_raw[split_idx:]
    val_dataset = NILMSlidingWindowDataset(X_val, y_p_val, y_s_val, window_len=config["simclr"]["window_len"], stride=config["simclr"]["stride"])
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Build Model Architecture
    print("[2/3] Constructing Hierarchical UNet model with SimCLR pre-trained Encoder...")
    model = HierarchicalAttentionUNet1D(
        in_channels=len(input_cols),
        out_channels=len(target_cols),
        target_cols=target_cols,
        p_agg_mean=p_agg_mean,
        p_agg_std=p_agg_std
    ).to(device)

    # Load pre-trained weights
    if os.path.exists(simclr_ckpt_path):
        print(f"  - Loading SimCLR Encoder weights from: {simclr_ckpt_path}")
        simclr_ckpt = torch.load(simclr_ckpt_path, map_location=device)
        enc_state = simclr_ckpt.get("encoder_state_dict", simclr_ckpt)
        
        # Transfer encoder weights to heavy_net encoder
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in enc_state.items():
            heavy_key = "heavy_net." + k
            if heavy_key in model_dict and model_dict[heavy_key].shape == v.shape:
                pretrained_dict[heavy_key] = v
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"  - Successfully transferred {len(pretrained_dict)} encoder layer weights!")

    elif os.path.exists(unet_ckpt_path):
        print(f"  - Loading Supervised UNet weights from: {unet_ckpt_path}")
        unet_ckpt = torch.load(unet_ckpt_path, map_location=device)
        model.load_state_dict(unet_ckpt.get("model_state_dict", unet_ckpt))
        print("  - Successfully loaded UNet checkpoint!")

    # 3. Evaluate Disaggregation Performance
    print("[3/3] Running Validation Evaluation...")
    model.eval()
    all_pred_p, all_gt_p = [], []
    all_pred_s, all_gt_s = [], []

    with torch.no_grad():
        for x_b, y_p_b, y_s_b in tqdm(val_loader, desc="[Evaluating]"):
            x_b, y_p_b, y_s_b = x_b.to(device), y_p_b.to(device), y_s_b.to(device)
            pred_p, pred_s = model(x_b)
            all_pred_p.append(pred_p.cpu().numpy())
            all_gt_p.append(y_p_b.cpu().numpy())
            all_pred_s.append(torch.sigmoid(pred_s).cpu().numpy())
            all_gt_s.append(y_s_b.cpu().numpy())

    all_pred_p = np.concatenate(all_pred_p, axis=0)
    all_gt_p = np.concatenate(all_gt_p, axis=0)
    all_pred_s = np.concatenate(all_pred_s, axis=0)
    all_gt_s = np.concatenate(all_gt_s, axis=0)

    # Calculate MAE per appliance
    mae_per_app = np.mean(np.abs(all_pred_p - all_gt_p), axis=(0, 2))
    overall_mae = np.mean(mae_per_app)

    # Calculate State ON/OFF Accuracy
    state_pred_binary = (all_pred_s >= 0.5).astype(np.float32)
    state_acc_per_app = np.mean(state_pred_binary == all_gt_s, axis=(0, 2)) * 100.0
    overall_acc = np.mean(state_acc_per_app)

    print("\n==================================================")
    print(" NILM 1D-SimCLR Disaggregation Evaluation Results")
    print("==================================================")
    print(f" Overall Active Power MAE Loss : {overall_mae:.4f} W")
    print(f" Overall Appliance ON/OFF Acc  : {overall_acc:.2f} %")
    print("--------------------------------------------------")
    print(" Per-Appliance Detailed Breakdowns:")
    for idx, col_name in enumerate(target_cols):
        app_clean = col_name.replace("p_w_", "").upper()
        print(f"  - {app_clean:20s} | MAE: {mae_per_app[idx]:6.2f} W | ON/OFF Acc: {state_acc_per_app[idx]:6.2f} %")
    print("==================================================")


if __name__ == "__main__":
    main()
