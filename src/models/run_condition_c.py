"""
SW-ADP Stage 4: Condition C Runner
Conditional fusion: encode all 7 channels (telem + SW), reconstruct telemetry only.

Usage:
    cd <repo_root>/src/models
    python run_condition_c.py

Loads existing results from Conditions A and B, runs Condition C,
produces updated combined ablation table.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score, roc_auc_score)

from data_utils import load_data, SegmentDataset
from torch.utils.data import DataLoader
from conditional_lstm_ae import ConditionalLSTMAutoencoder
from conditional_tcn_ae import ConditionalTCNAutoencoder

# ── Configuration ──────────────────────────────────────────────────────
SEEDS = [42, 123, 456]
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
PATIENCE = 10

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TENSOR_PATH = os.path.join(REPO_ROOT, 'data', 'processed', 'track_b_tensors.npz')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'experiments', 'stage4_deep_learning')
IF_RESULTS_PATH = os.path.join(REPO_ROOT, 'experiments', 'stage3_baselines', 'if_ablation_table.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Conditional loss: MSE on telemetry channel only ───────────────────

def conditional_masked_mse_loss(reconstructed, full_input, mask):
    """
    MSE between reconstructed telemetry and actual telemetry channel,
    masked for padding.

    Args:
        reconstructed: [batch, seq_len, 1] (model output -- telem only)
        full_input: [batch, seq_len, 7] (full input with all channels)
        mask: [batch, seq_len] (1=real, 0=pad)
    Returns:
        scalar loss
    """
    # Target is just the first channel (telemetry)
    target = full_input[:, :, 0:1]  # [batch, seq_len, 1]

    mask_expanded = mask.unsqueeze(-1)  # [batch, seq_len, 1]
    sq_error = (reconstructed - target) ** 2
    masked_error = sq_error * mask_expanded

    real_counts = mask.sum(dim=1)  # [batch]
    per_sample_mse = masked_error.sum(dim=(1, 2)) / (real_counts + 1e-8)
    return per_sample_mse.mean()


def conditional_per_segment_mse(reconstructed, full_input, mask):
    """Per-segment reconstruction error for anomaly scoring."""
    target = full_input[:, :, 0:1]
    mask_expanded = mask.unsqueeze(-1)
    sq_error = (reconstructed - target) ** 2
    masked_error = sq_error * mask_expanded
    real_counts = mask.sum(dim=1)
    return masked_error.sum(dim=(1, 2)) / (real_counts + 1e-8)


# ── Training loop for conditional models ──────────────────────────────

def train_conditional(model, train_loader, val_loader, device,
                      epochs=100, lr=1e-3, patience=10, verbose=True):
    """Train conditional AE: input is full 7ch, loss on telem channel only."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            X_batch, mask_batch = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            reconstructed = model(X_batch)  # [batch, seq_len, 1]
            loss = conditional_masked_mse_loss(reconstructed, X_batch, mask_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                X_batch, mask_batch = batch[0].to(device), batch[1].to(device)
                reconstructed = model(X_batch)
                loss = conditional_masked_mse_loss(reconstructed, X_batch, mask_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{epochs}: train={train_loss:.6f}  val={val_loss:.6f}  lr={lr_now:.1e}")

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1} (best val={best_val_loss:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    return model, history


def compute_conditional_scores(model, dataloader, device):
    """Compute per-segment reconstruction error (telem channel only)."""
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                X_batch, mask_batch, y_batch = batch
                all_labels.append(y_batch.numpy())
            else:
                X_batch, mask_batch = batch
            X_batch = X_batch.to(device)
            mask_batch = mask_batch.to(device)
            reconstructed = model(X_batch)
            errors = conditional_per_segment_mse(reconstructed, X_batch, mask_batch)
            all_scores.append(errors.cpu().numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels) if all_labels else None


def evaluate_scores(scores, y_true, contamination=0.197):
    threshold = np.percentile(scores, (1 - contamination) * 100)
    y_pred = (scores >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'au_pr': average_precision_score(y_true, scores),
        'roc_auc': roc_auc_score(y_true, scores),
    }


def build_dataloaders(X, mask, y, train_mask, batch_size=32, val_frac=0.2, seed=42):
    """Build train(normal only)/val/test dataloaders."""
    rng = np.random.RandomState(seed)
    normal_train_idx = np.where((y == 0) & train_mask)[0]
    rng.shuffle(normal_train_idx)
    n_val = int(len(normal_train_idx) * val_frac)
    val_idx = normal_train_idx[:n_val]
    train_idx = normal_train_idx[n_val:]
    test_idx = np.where(~train_mask)[0]

    train_ds = SegmentDataset(X[train_idx], mask[train_idx])
    val_ds = SegmentDataset(X[val_idx], mask[val_idx])
    test_ds = SegmentDataset(X[test_idx], mask[test_idx], y[test_idx])

    print(f"  Train (normal): {len(train_idx)}  Val (normal): {len(val_idx)}  Test: {len(test_idx)} ({(y[test_idx]==1).sum()} anom)")
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False),
            y[test_idx])


def run_condition_c(model_class, model_name, data, device, seeds):
    """Run Condition C for one model class."""
    X_sw = data['X_telem_sw']  # [2098, 128, 7] -- full input
    input_dim = X_sw.shape[-1]  # 7
    output_dim = 1               # reconstruct telem only

    if model_class == ConditionalLSTMAutoencoder:
        kwargs = {'input_dim': input_dim, 'output_dim': output_dim,
                  'hidden_dim': 64, 'n_layers': 2, 'dropout': 0.1}
    elif model_class == ConditionalTCNAutoencoder:
        kwargs = {'input_dim': input_dim, 'output_dim': output_dim,
                  'channels': [16, 32, 64], 'kernel_size': 3, 'dropout': 0.1}
    else:
        raise ValueError(f"Unknown: {model_class}")

    print(f"\n{'='*60}")
    print(f"{model_name} -- Condition C: Conditional Fusion")
    print(f"  Encoder input: {input_dim}ch (telem + SW)")
    print(f"  Decoder output: {output_dim}ch (telem only)")
    print(f"{'='*60}")

    all_results = []
    all_histories = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader, val_loader, test_loader, test_y = build_dataloaders(
            X_sw, data['mask'], data['y'], data['train_mask'],
            batch_size=BATCH_SIZE, val_frac=0.2, seed=seed
        )

        model = model_class(**kwargs)
        n_params, size_kb = model.get_model_size()
        if seed == seeds[0]:
            print(f"  Model: {n_params:,} params ({size_kb:.1f} KB)")

        t0 = time.time()
        model, history = train_conditional(
            model, train_loader, val_loader, device,
            epochs=EPOCHS, lr=LR, patience=PATIENCE, verbose=(seed == seeds[0])
        )
        print(f"  Training: {len(history['train_loss'])} epochs in {time.time()-t0:.1f}s")
        all_histories.append(history)

        scores, labels = compute_conditional_scores(model, test_loader, device)
        contamination = data['y'][data['train_mask']].mean()
        metrics = evaluate_scores(scores, labels, contamination)
        all_results.append(metrics)
        print(f"  F1={metrics['f1']:.4f}  AU-PR={metrics['au_pr']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")

    return all_results, all_histories


def summarize(results, label):
    print(f"\n{label}:")
    summary = {}
    for m in results[0]:
        vals = [r[m] for r in results]
        mn, sd = np.mean(vals), np.std(vals)
        summary[f'{m}_mean'] = round(mn, 4)
        summary[f'{m}_std'] = round(sd, 4)
        print(f"  {m}: {mn:.4f} +/- {sd:.4f}")
    return summary


def significance_test(res_a, res_c, label):
    """Test Condition C vs Condition A (telem only)."""
    print(f"\n{label}:")
    deltas = {}
    for m in res_a[0]:
        va = [r[m] for r in res_a]
        vc = [r[m] for r in res_c]
        t, p = stats.ttest_rel(va, vc)
        d = np.mean(vc) - np.mean(va)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        deltas[m] = {'delta': round(d, 4), 'p': round(p, 4), 'sig': sig}
        print(f"  {m}: C-A = {d:+.4f}  p={p:.4f} {sig}")
    return deltas


def plot_training_curves(histories, model_name, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i, h in enumerate(histories):
        alpha = 1.0 if i == 0 else 0.3
        ax.plot(h['train_loss'], color='blue', alpha=alpha,
                label='Train' if i == 0 else None)
        ax.plot(h['val_loss'], color='red', alpha=alpha,
                label='Val' if i == 0 else None)
    ax.set_title(f'{model_name} - Conditional Fusion (Condition C)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Masked MSE Loss (telem only)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = model_name.lower().replace('-', '_')
    path = os.path.join(output_dir, f'{fname}_condition_c_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Curves saved to {path}")


def main():
    print("=" * 60)
    print("SW-ADP STAGE 4: Condition C -- Conditional Fusion")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(TENSOR_PATH):
        print(f"\nERROR: {TENSOR_PATH} not found")
        sys.exit(1)

    data = load_data(TENSOR_PATH)
    print(f"  Segments: {len(data['y'])} ({(data['y']==1).sum()} anomalous)")
    print(f"  X_telem_sw shape: {data['X_telem_sw'].shape}")

    t_total = time.time()

    # ── LSTM-AE Condition C ───────────────────────────────────────────
    lstm_results, lstm_histories = run_condition_c(
        ConditionalLSTMAutoencoder, 'LSTM-AE', data, device, SEEDS)
    lstm_sum = summarize(lstm_results, "LSTM-AE Condition C (Conditional Fusion)")
    plot_training_curves(lstm_histories, 'LSTM-AE', OUTPUT_DIR)

    # ── TCN-AE Condition C ────────────────────────────────────────────
    tcn_results, tcn_histories = run_condition_c(
        ConditionalTCNAutoencoder, 'TCN-AE', data, device, SEEDS)
    tcn_sum = summarize(tcn_results, "TCN-AE Condition C (Conditional Fusion)")
    plot_training_curves(tcn_histories, 'TCN-AE', OUTPUT_DIR)

    # ── Load Condition A results for significance testing ─────────────
    lstm_a_path = os.path.join(OUTPUT_DIR, 'lstm_ae_results.json')
    tcn_a_path = os.path.join(OUTPUT_DIR, 'tcn_ae_results.json')

    if os.path.exists(lstm_a_path) and os.path.exists(tcn_a_path):
        with open(lstm_a_path) as f:
            lstm_prev = json.load(f)
        with open(tcn_a_path) as f:
            tcn_prev = json.load(f)

        lstm_a_results = lstm_prev['condition_A']['per_seed']
        tcn_a_results = tcn_prev['condition_A']['per_seed']

        print("\n" + "=" * 60)
        print("SIGNIFICANCE TESTS")
        print("=" * 60)
        lstm_deltas_ca = significance_test(lstm_a_results, lstm_results, "LSTM-AE: C vs A")
        tcn_deltas_ca = significance_test(tcn_a_results, tcn_results, "TCN-AE: C vs A")

        # Also test C vs B
        lstm_b_results = lstm_prev['condition_B']['per_seed']
        tcn_b_results = tcn_prev['condition_B']['per_seed']
        lstm_deltas_cb = significance_test(lstm_b_results, lstm_results, "LSTM-AE: C vs B")
        tcn_deltas_cb = significance_test(tcn_b_results, tcn_results, "TCN-AE: C vs B")
    else:
        print("\nWARNING: Could not load A/B results for significance tests.")
        print("  Run run_stage4.py first, then run this script.")
        lstm_deltas_ca = tcn_deltas_ca = {}
        lstm_deltas_cb = tcn_deltas_cb = {}

    # ── Save Condition C results ──────────────────────────────────────
    for model_name, results, histories, sum_c, deltas_ca, deltas_cb in [
        ('LSTM-AE', lstm_results, lstm_histories, lstm_sum, lstm_deltas_ca, lstm_deltas_cb),
        ('TCN-AE', tcn_results, tcn_histories, tcn_sum, tcn_deltas_ca, tcn_deltas_cb),
    ]:
        fname = model_name.lower().replace('-', '_')
        result_dict = {
            'model': model_name,
            'condition': 'C_Conditional_Fusion',
            'description': 'Encode telem+SW (7ch), reconstruct telem only (1ch)',
            'per_seed': results,
            'summary': sum_c,
            'deltas_vs_A': deltas_ca,
            'deltas_vs_B': deltas_cb,
            'seeds': SEEDS,
            'hyperparams': {'batch_size': BATCH_SIZE, 'lr': LR,
                            'epochs': EPOCHS, 'patience': PATIENCE},
        }
        path = os.path.join(OUTPUT_DIR, f'{fname}_condition_c_results.json')
        with open(path, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)

    # ── Build updated combined ablation table ─────────────────────────
    print("\n" + "=" * 60)
    print("BUILDING UPDATED ABLATION TABLE")
    print("=" * 60)

    rows = []

    # Load existing combined table if available
    existing_path = os.path.join(OUTPUT_DIR, 'combined_ablation_table.csv')
    if os.path.exists(existing_path):
        existing = pd.read_csv(existing_path)
        for _, row in existing.iterrows():
            rows.append(row.to_dict())

    # Add Condition C rows
    rows.append({'Model': 'LSTM-AE', 'Condition': 'Conditional_SW', **lstm_sum})
    rows.append({'Model': 'TCN-AE', 'Condition': 'Conditional_SW', **tcn_sum})

    df = pd.DataFrame(rows)

    # Save updated table
    path = os.path.join(OUTPUT_DIR, 'combined_ablation_table_v2.csv')
    df.to_csv(path, index=False)

    # Pretty print
    print(f"\n{'Model':<12} {'Condition':<18} {'F1':>8} {'AU-PR':>8} {'ROC-AUC':>8} {'Acc':>8}")
    print("-" * 70)
    for _, row in df.iterrows():
        f1 = row.get('f1_mean', 0)
        aupr = row.get('au_pr_mean', 0)
        roc = row.get('roc_auc_mean', 0)
        acc = row.get('accuracy_mean', 0)
        print(f"{row['Model']:<12} {row['Condition']:<18} {f1:>8.4f} {aupr:>8.4f} {roc:>8.4f} {acc:>8.4f}")

    total = time.time() - t_total
    print(f"\nCondition C complete in {total:.0f}s ({total/60:.1f} min)")
    print(f"Results saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
