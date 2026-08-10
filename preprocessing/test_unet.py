#!/usr/bin/env python3
"""
PyTorch 1D-UNet NILM AI Disaggregation Evaluation & Testing Script.
Loads trained checkpoint (best_unet_nilm.pth) and scaler.pkl, calculates evaluation metrics (MAE, RMSE, R2, F1-Score),
and plots real vs predicted disaggregation waveforms.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score

# Fix Windows Intel OpenMP library duplication error (OMP: Error #15)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from train_unet import UNet1D, NILMSlidingWindowDataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained 1D-UNet for NILM AI Disaggregation.")
    parser.add_argument("-data", "--dataset", type=str, default="./output/synthetic_nilm_12h.csv", help="Test CSV dataset path")
    parser.add_argument("-ckpt", "--checkpoint-dir", type=str, default="./checkpoint_unet", help="Checkpoint directory containing best_unet_nilm.pth")
    parser.add_argument("-w", "--window-len", type=int, default=320, help="Sliding window length in cycles (default: 320)")
    parser.add_argument("-b", "--batch-size", type=int, default=256, help="Batch size for evaluation")
    parser.add_argument("-plots", "--plot-cycles", type=int, default=3600, help="Number of cycles to visualize in plot (~60 seconds)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print(" NILM AI 1D-UNet Disaggregation Test & Evaluation")
    print("==================================================")
    print(f"Device        : {device}")
    if device.type == "cuda":
        print(f"GPU Model     : {torch.cuda.get_device_name(0)}")
    print(f"Test Dataset  : {args.dataset}")
    print(f"Checkpoint    : {args.checkpoint_dir}")
    print("--------------------------------------------------")

    scaler_path = os.path.join(args.checkpoint_dir, "scaler.pkl")
    model_path = os.path.join(args.checkpoint_dir, "best_unet_nilm.pth")

    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        print(f"[ERROR] Trained model files not found in {args.checkpoint_dir}", file=sys.stderr)
        print("Please train the model first using: python train_unet.py", file=sys.stderr)
        sys.exit(1)

    # 1. Load Scaler & Feature metadata
    print("[1/4] Loading trained scaler & model checkpoint...")
    with open(scaler_path, "rb") as f:
        meta = pickle.load(f)

    scaler = meta["scaler"]
    input_cols = meta["input_cols"]
    target_cols = meta["target_cols"]

    # Load Model
    model = UNet1D(in_channels=len(input_cols), out_channels=len(target_cols)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  - Loaded model checkpoint from Epoch {checkpoint.get('epoch', '?')} (Val Loss: {checkpoint.get('val_loss', 0.0):.4f})")

    # 2. Load & Preprocess Test Data
    print("[2/4] Loading test dataset & normalizing...")
    df = pd.read_csv(args.dataset)
    # Use validation slice (last 20% of data) for evaluation
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    X_test_raw = test_df[input_cols].values
    y_test_raw = test_df[target_cols].values

    X_test_scaled = scaler.transform(X_test_raw)
    test_dataset = NILMSlidingWindowDataset(X_test_scaled, y_test_raw, window_len=args.window_len, stride=args.window_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Perform Disaggregation Inference
    print("[3/4] Running Attention Multi-Task 1D-UNet Model Inference...")
    all_preds, all_targets = [], []

    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            x_b = batch[0].to(device)
            y_b = batch[1]
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                res = model(x_b)
                pred_power = res[0] if isinstance(res, (tuple, list)) else res
            all_preds.append(pred_power.cpu().numpy())
            all_targets.append(y_b.numpy())


    infer_time = round(time.time() - t0, 2)
    print(f"  - Disaggregated {len(test_dataset)*args.window_len:,} cycles in {infer_time}s ({round(len(test_dataset)*args.window_len/infer_time, 0):,} cycles/sec)")

    # Reconstruct continuous predictions: (N_windows, num_app, window_len) -> (N_total_cycles, num_app)
    preds_arr = np.concatenate(all_preds, axis=0).transpose(0, 2, 1).reshape(-1, len(target_cols))
    targets_arr = np.concatenate(all_targets, axis=0).transpose(0, 2, 1).reshape(-1, len(target_cols))

    # 4. Calculate Evaluation Metrics
    print("\n==================================================")
    print(" 📊 NILM Disaggregation Performance Metrics")
    print("==================================================")
    print(f"{'Appliance Category':<28} | {'MAE (W)':<8} | {'RMSE (W)':<8} | {'R² Score':<9} | {'F1-Score':<8}")
    print("-" * 75)

    app_names = [c.replace("p_w_", "") for c in target_cols]
    mae_list, r2_list, f1_list = [], [], []

    for idx, app_name in enumerate(app_names):
        y_true = targets_arr[:, idx]
        y_pred = preds_arr[:, idx]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # F1-Score for ON/OFF state detection (Threshold: 5W)
        true_on = y_true >= 5.0
        pred_on = y_pred >= 5.0
        f1 = f1_score(true_on, pred_on, zero_division=1.0)

        mae_list.append(mae)
        r2_list.append(r2)
        f1_list.append(f1)

        print(f"{app_name:<28} | {mae:8.2f} | {rmse:8.2f} | {r2*100:8.2f}% | {f1*100:7.2f}%")

    print("-" * 75)
    print(f"{'AVERAGE PERFORMANCE':<28} | {np.mean(mae_list):8.2f} | {'-':<8} | {np.mean(r2_list)*100:8.2f}% | {np.mean(f1_list)*100:7.2f}%")
    print("==================================================")

    # 5. Generate Visual Disaggregation Result Plot
    if HAS_MATPLOTLIB:
        plot_len = min(args.plot_cycles, len(targets_arr))
        time_axis = np.arange(plot_len) * 0.016667  # seconds

        fig, axes = plt.subplots(len(app_names) + 1, 1, figsize=(14, 2 * (len(app_names) + 1)), sharex=True)

        # Plot 0: Total Aggregate Active Power Input
        p_agg_test = test_df["p_w_agg"].values[:plot_len]
        axes[0].plot(time_axis, p_agg_test, color="black", label="Aggregate Input (P_agg)", linewidth=1.2)
        axes[0].set_ylabel("Power (W)")
        axes[0].set_title("NILM Disaggregation Test Results: Total Aggregate Power & Appliance Predictions")
        axes[0].legend(loc="upper right")
        axes[0].grid(True)

        # Plot 1~N: Appliance Ground Truth vs Prediction
        for idx, app_name in enumerate(app_names):
            ax = axes[idx + 1]
            ax.plot(time_axis, targets_arr[:plot_len, idx], color="blue", label="Ground Truth (Real)", linewidth=1.5, alpha=0.7)
            ax.plot(time_axis, preds_arr[:plot_len, idx], color="red", linestyle="--", label="1D-UNet Prediction", linewidth=1.5, alpha=0.8)
            ax.set_ylabel(f"{app_name}\n(W)")
            ax.legend(loc="upper right")
            ax.grid(True)

        axes[-1].set_xlabel("Time (seconds)")
        plt.tight_layout()

        plot_save_path = os.path.join(args.checkpoint_dir, "disaggregation_result.png")
        plt.savefig(plot_save_path, dpi=200)
        plt.close()
        print(f"\nSaved Disaggregation Result Plot to: {plot_save_path}")


if __name__ == "__main__":
    main()
