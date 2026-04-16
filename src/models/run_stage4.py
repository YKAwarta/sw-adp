"""
SW-ADP Stage 4: Master Runner
Runs LSTM-AE and TCN-AE under both conditions, compiles combined ablation table.

Usage:
    cd <repo_root>/src/models
    python run_stage4.py

Expected runtime with RTX 4060Ti: ~10-15 minutes total.
Outputs saved to experiments/phase1/stage4_deep_learning/
"""

import json
import os
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local imports
from data_utils import load_data
from lstm_ae import LSTMAutoencoder
from scipy import stats
from tcn_ae import TCNAutoencoder
from train_eval import run_ablation_for_model

# ── Configuration ──────────────────────────────────────────────────────
SEEDS = [42, 123, 456]  # 3 seeds (adequate for poster; add 789, 1024 for paper)
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
PATIENCE = 10

# Paths (relative to repo root, or adjusted if running from src/models/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TENSOR_PATH = os.path.join(REPO_ROOT, "data", "processed", "track_b_tensors.npz")
TRACK_A_SW_PATH = os.path.join(REPO_ROOT, "data", "processed", "track_a_telem_sw.csv")
OUTPUT_DIR = os.path.join(REPO_ROOT, "experiments", "phase1", "stage4_deep_learning")
IF_RESULTS_PATH = os.path.join(
    REPO_ROOT, "experiments", "phase1", "stage3_baselines", "if_ablation_table.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def summarize_results(results_list, label):
    """Compute mean +/- std across seeds."""
    summary = {}
    metrics = list(results_list[0].keys())
    print(f"\n{label}:")
    for m in metrics:
        vals = [r[m] for r in results_list]
        mn, sd = np.mean(vals), np.std(vals)
        summary[f"{m}_mean"] = round(mn, 4)
        summary[f"{m}_std"] = round(sd, 4)
        print(f"  {m}: {mn:.4f} +/- {sd:.4f}")
    return summary


def significance_test(results_A, results_B):
    """Paired t-test between conditions."""
    deltas = {}
    metrics = list(results_A[0].keys())
    print("\nSignificance (B - A):")
    for m in metrics:
        va = [r[m] for r in results_A]
        vb = [r[m] for r in results_B]
        t_stat, p_val = stats.ttest_rel(va, vb)
        d = np.mean(vb) - np.mean(va)
        sig = (
            "***"
            if p_val < 0.001
            else "**"
            if p_val < 0.01
            else "*"
            if p_val < 0.05
            else "ns"
        )
        deltas[m] = {"delta": round(d, 4), "p": round(p_val, 4), "sig": sig}
        print(f"  {m}: {d:+.4f}  p={p_val:.4f} {sig}")
    return deltas


def plot_training_curves(histories, model_name, output_dir):
    """Plot train/val loss curves for both conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (cond, hist_list) in zip(axes, histories.items()):
        for i, h in enumerate(hist_list):
            alpha = 1.0 if i == 0 else 0.3
            ax.plot(
                h["train_loss"],
                color="blue",
                alpha=alpha,
                label="Train" if i == 0 else None,
            )
            ax.plot(
                h["val_loss"], color="red", alpha=alpha, label="Val" if i == 0 else None
            )
        ax.set_title(f"{model_name} - {cond}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Masked MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(
        output_dir, f"{model_name.lower().replace('-', '_')}_training_curves.png"
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved to {path}")


def run_model(model_class, model_name, data, device):
    """Run a single model class through full ablation."""
    input_dim_A = data["X_telem"].shape[-1]  # 1
    input_dim_B = data["X_telem_sw"].shape[-1]  # 7

    if model_class == LSTMAutoencoder:
        kwargs_A = {
            "input_dim": input_dim_A,
            "hidden_dim": 64,
            "n_layers": 2,
            "dropout": 0.1,
        }
        kwargs_B = {
            "input_dim": input_dim_B,
            "hidden_dim": 64,
            "n_layers": 2,
            "dropout": 0.1,
        }
    elif model_class == TCNAutoencoder:
        kwargs_A = {
            "input_dim": input_dim_A,
            "channels": [16, 32, 64],
            "kernel_size": 3,
            "dropout": 0.1,
        }
        kwargs_B = {
            "input_dim": input_dim_B,
            "channels": [16, 32, 64],
            "kernel_size": 3,
            "dropout": 0.1,
        }
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    results, histories = run_ablation_for_model(
        model_class=model_class,
        model_kwargs_A=kwargs_A,
        model_kwargs_B=kwargs_B,
        data=data,
        seeds=SEEDS,
        device=device,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        patience=PATIENCE,
        model_name=model_name,
    )

    # Summarize
    sum_A = summarize_results(results["Telem_Only"], f"{model_name} Telem Only")
    sum_B = summarize_results(results["Telem_SW"], f"{model_name} Telem + SW")
    deltas = significance_test(results["Telem_Only"], results["Telem_SW"])

    # Plot
    plot_training_curves(histories, model_name, OUTPUT_DIR)

    # Save per-model results
    model_results = {
        "model": model_name,
        "architecture": str(kwargs_A),
        "condition_A": {
            "label": f"{model_name} Telem Only",
            "per_seed": results["Telem_Only"],
            "summary": sum_A,
        },
        "condition_B": {
            "label": f"{model_name} Telem + SW",
            "per_seed": results["Telem_SW"],
            "summary": sum_B,
        },
        "deltas": deltas,
        "seeds": SEEDS,
        "hyperparams": {
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
            "patience": PATIENCE,
        },
    }
    fname = model_name.lower().replace("-", "_")
    with open(os.path.join(OUTPUT_DIR, f"{fname}_results.json"), "w") as f:
        json.dump(model_results, f, indent=2, default=str)

    return sum_A, sum_B, deltas


def build_combined_ablation_table(model_summaries, output_dir):
    """
    Build the full ablation table: 4 models x 2 conditions x 6 metrics.
    Includes IF results from Stage 3.
    """
    rows = []
    for model_name, sum_A, sum_B in model_summaries:
        rows.append({"Model": model_name, "Condition": "Telem_Only", **sum_A})
        rows.append({"Model": model_name, "Condition": "Telem_SW", **sum_B})

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "combined_ablation_table.csv")
    df.to_csv(path, index=False)
    print(f"\nCombined ablation table saved to {path}")

    # Pretty print
    print("\n" + "=" * 80)
    print("FULL ABLATION TABLE")
    print("=" * 80)
    metrics = ["accuracy", "precision", "recall", "f1", "au_pr", "roc_auc"]
    header = f"{'Model':<20} {'Condition':<12}"
    for m in metrics:
        header += f" {m:>12}"
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        line = f"{row['Model']:<20} {row['Condition']:<12}"
        for m in metrics:
            mean = row.get(f"{m}_mean", 0)
            std = row.get(f"{m}_std", 0)
            line += f" {mean:>5.3f}+{std:.3f}"
        print(line)

    return df


def main():
    print("=" * 60)
    print("SW-ADP STAGE 4: Deep Learning Models")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Check for tensor file
    if not os.path.exists(TENSOR_PATH):
        print(f"\nERROR: Cannot find {TENSOR_PATH}")
        print("Make sure track_b_tensors.npz is in data/processed/")
        print(
            "If running from a different directory, adjust TENSOR_PATH at top of script."
        )
        sys.exit(1)

    # Load data
    print(f"\nLoading data from {TENSOR_PATH}")
    data = load_data(TENSOR_PATH)
    print(f"  Segments: {len(data['y'])} ({(data['y'] == 1).sum()} anomalous)")
    print(f"  Train: {data['train_mask'].sum()}, Test: {(~data['train_mask']).sum()}")
    print(
        f"  Telem shape: {data['X_telem'].shape}, Telem+SW shape: {data['X_telem_sw'].shape}"
    )

    t_total = time.time()
    model_summaries = []

    # ── Load IF results from Stage 3 ──────────────────────────────────
    if os.path.exists(IF_RESULTS_PATH):
        print(f"\nLoading IF results from {IF_RESULTS_PATH}")
        if_df = pd.read_csv(IF_RESULTS_PATH)
        if_A = (
            if_df[
                if_df["Condition"].str.contains(
                    "Telem Only|telem_only|A", case=False, na=False
                )
            ]
            .iloc[0]
            .to_dict()
        )
        if_B = (
            if_df[
                if_df["Condition"].str.contains(
                    "Telem.*SW|telem_sw|B", case=False, na=False
                )
            ]
            .iloc[0]
            .to_dict()
        )
        # Normalize to our format
        sum_if_A, sum_if_B = {}, {}
        for m in ["accuracy", "precision", "recall", "f1", "au_pr", "roc_auc"]:
            for suffix in ["_mean", "_std"]:
                key = f"{m}{suffix}"
                if key in if_A:
                    sum_if_A[key] = round(float(if_A[key]), 4)
                    sum_if_B[key] = round(float(if_B[key]), 4)
        model_summaries.append(("IF", sum_if_A, sum_if_B))
        print("  IF results loaded successfully.")
    else:
        print(f"\nWARNING: IF results not found at {IF_RESULTS_PATH}")
        print(
            "  IF will be missing from combined table. Run Stage 3 first or adjust path."
        )

    # ── Load MP results if available ──────────────────────────────────
    mp_results_path = os.path.join(OUTPUT_DIR, "mp_results.json")
    if os.path.exists(mp_results_path):
        print(f"\nLoading MP results from {mp_results_path}")
        with open(mp_results_path) as f:
            mp_data = json.load(f)
        # Handle both formats: top-level 'A'/'B' or nested 'condition_A'/'condition_B'
        if "A" in mp_data:
            model_summaries.append(("MP+IF", mp_data["A"], mp_data["B"]))
        else:
            model_summaries.append(
                (
                    "MP+IF",
                    mp_data["condition_A"]["summary"],
                    mp_data["condition_B"]["summary"],
                )
            )
        print("  MP results loaded successfully.")
    else:
        print("\nWARNING: MP results not found. Run run_mp_baseline.py first.")

    # ── LSTM Autoencoder ──────────────────────────────────────────────
    print("\n" + "#" * 60)
    print("# LSTM AUTOENCODER (DL Baseline)")
    print("#" * 60)
    t0 = time.time()
    sum_A, sum_B, deltas = run_model(LSTMAutoencoder, "LSTM-AE", data, device)
    print(f"\nLSTM-AE total time: {time.time() - t0:.1f}s")
    model_summaries.append(("LSTM-AE", sum_A, sum_B))

    # ── TCN Autoencoder ───────────────────────────────────────────────
    print("\n" + "#" * 60)
    print("# TCN AUTOENCODER (Core Model)")
    print("#" * 60)
    t0 = time.time()
    sum_A, sum_B, deltas = run_model(TCNAutoencoder, "TCN-AE", data, device)
    print(f"\nTCN-AE total time: {time.time() - t0:.1f}s")
    model_summaries.append(("TCN-AE", sum_A, sum_B))

    # ── Combined ablation table ───────────────────────────────────────
    df = build_combined_ablation_table(model_summaries, OUTPUT_DIR)

    total_time = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"STAGE 4 COMPLETE in {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"All results saved to {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
