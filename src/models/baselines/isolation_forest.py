"""
Stage 3 - Isolation Forest Baseline (Both Conditions)
======================================================
Runs Isolation Forest on Track A segment-level features.
  Condition A: 18 handcrafted telemetry features only
  Condition B: 18 telemetry + 17 space weather summary statistics

Evaluation: Accuracy, Precision, Recall, F1, AU-PR, ROC-AUC
Runs 5 seeds, reports mean +/- std.
Per-channel breakdown included.
Statistical significance via paired t-test on F1 scores across seeds.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix
)
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]
N_ESTIMATORS = 200  # more trees = more stable scores

META_COLS = ['segment', 'anomaly', 'train', 'channel', 'sampling']
TELEM_FEATURES = [
    'duration', 'len', 'mean', 'var', 'std', 'kurtosis', 'skew',
    'n_peaks', 'smooth10_n_peaks', 'smooth20_n_peaks',
    'diff_peaks', 'diff2_peaks', 'diff_var', 'diff2_var',
    'gaps_squared', 'len_weighted', 'var_div_duration', 'var_div_len'
]
SW_FEATURES = [
    'sw_kp_mean', 'sw_kp_max', 'sw_kp_std',
    'sw_dst_mean', 'sw_dst_min', 'sw_dst_std',
    'sw_f10_7_mean', 'sw_f10_7_std',
    'sw_sw_speed_mean', 'sw_sw_speed_max', 'sw_sw_speed_std',
    'sw_sw_proton_density_mean', 'sw_sw_proton_density_max', 'sw_sw_proton_density_std',
    'sw_imf_bz_gsm_mean', 'sw_imf_bz_gsm_min', 'sw_imf_bz_gsm_std'
]

DATA_DIR = '/mnt/project'
OUTPUT_DIR = '/home/claude/experiments/stage3_baselines'


def load_data():
    """Load both Track A CSVs and return train/test splits."""
    df_to = pd.read_csv(os.path.join(DATA_DIR, 'track_a_telem_only.csv'))
    df_sw = pd.read_csv(os.path.join(DATA_DIR, 'track_a_telem_sw.csv'))

    return df_to, df_sw


def get_splits(df, feature_cols):
    """Split into train/test using the pre-defined split column."""
    train = df[df['train'] == 1]
    test = df[df['train'] == 0]

    X_train = train[feature_cols].values
    y_train = train['anomaly'].values
    X_test = test[feature_cols].values
    y_test = test['anomaly'].values
    channels_test = test['channel'].values

    return X_train, y_train, X_test, y_test, channels_test


def evaluate(y_true, y_pred, anomaly_scores):
    """Compute all metrics. anomaly_scores should be higher = more anomalous."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    # AU-PR and ROC-AUC need continuous scores
    # sklearn IF decision_function: lower = more anomalous, so negate
    if len(np.unique(y_true)) > 1:
        metrics['au_pr'] = average_precision_score(y_true, anomaly_scores)
        metrics['roc_auc'] = roc_auc_score(y_true, anomaly_scores)
    else:
        metrics['au_pr'] = float('nan')
        metrics['roc_auc'] = float('nan')

    return metrics


def run_if_single(X_train, y_train, X_test, y_test, contamination, seed):
    """Run a single Isolation Forest experiment."""
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Fit IF
    clf = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1
    )
    clf.fit(X_train_s)

    # Predict: IF returns -1 for anomaly, 1 for normal
    y_pred_raw = clf.predict(X_test_s)
    y_pred = (y_pred_raw == -1).astype(int)  # convert to 0/1

    # Anomaly scores: negate decision_function so higher = more anomalous
    scores = -clf.decision_function(X_test_s)

    return y_pred, scores


def run_condition(df, feature_cols, condition_name):
    """Run IF under one condition across all seeds."""
    X_train, y_train, X_test, y_test, channels_test = get_splits(df, feature_cols)

    # Set contamination to training anomaly rate
    contamination = y_train.mean()
    print(f"\n{'='*60}")
    print(f"Condition: {condition_name}")
    print(f"Features: {len(feature_cols)}")
    print(f"Train: {len(X_train)} (anomaly rate: {contamination:.3f})")
    print(f"Test: {len(X_test)} (anomaly rate: {y_test.mean():.3f})")
    print(f"Contamination parameter: {contamination:.3f}")
    print(f"{'='*60}")

    # ── Pooled results across seeds ──
    all_metrics = []
    all_preds = []
    all_scores = []

    for seed in SEEDS:
        y_pred, scores = run_if_single(X_train, y_train, X_test, y_test, contamination, seed)
        m = evaluate(y_test, y_pred, scores)
        all_metrics.append(m)
        all_preds.append(y_pred)
        all_scores.append(scores)
        print(f"  Seed {seed:>4d}: Acc={m['accuracy']:.3f}  P={m['precision']:.3f}  "
              f"R={m['recall']:.3f}  F1={m['f1']:.3f}  AU-PR={m['au_pr']:.3f}  "
              f"ROC={m['roc_auc']:.3f}")

    # Summary stats
    summary = {}
    for metric_name in all_metrics[0].keys():
        vals = [m[metric_name] for m in all_metrics]
        summary[metric_name] = {
            'mean': np.nanmean(vals),
            'std': np.nanstd(vals),
            'values': vals
        }

    print(f"\n  Summary (mean +/- std):")
    for k, v in summary.items():
        print(f"    {k:>10s}: {v['mean']:.4f} +/- {v['std']:.4f}")

    # ── Per-channel breakdown (using median-seed run) ──
    # Use the seed with median F1
    f1_vals = [m['f1'] for m in all_metrics]
    median_idx = np.argsort(f1_vals)[len(f1_vals) // 2]
    y_pred_median = all_preds[median_idx]
    scores_median = all_scores[median_idx]

    channel_results = {}
    unique_channels = sorted(np.unique(channels_test))
    print(f"\n  Per-channel breakdown (median seed = {SEEDS[median_idx]}):")
    print(f"  {'Channel':<12s} {'N':>4s} {'Anom':>4s} {'Acc':>6s} {'P':>6s} {'R':>6s} {'F1':>6s} {'AU-PR':>6s}")

    for ch in unique_channels:
        mask = channels_test == ch
        y_ch = y_test[mask]
        pred_ch = y_pred_median[mask]
        scores_ch = scores_median[mask]

        ch_metrics = evaluate(y_ch, pred_ch, scores_ch)
        channel_results[ch] = ch_metrics
        n_anom = y_ch.sum()
        print(f"  {ch:<12s} {mask.sum():>4d} {n_anom:>4d} {ch_metrics['accuracy']:>6.3f} "
              f"{ch_metrics['precision']:>6.3f} {ch_metrics['recall']:>6.3f} "
              f"{ch_metrics['f1']:>6.3f} {ch_metrics['au_pr']:>6.3f}")

    return summary, channel_results, all_metrics, all_preds, all_scores, y_test, channels_test


def main():
    print("Stage 3: Isolation Forest Baseline")
    print("=" * 60)

    df_to, df_sw = load_data()

    # Condition A: Telemetry only
    summary_a, ch_a, metrics_a, preds_a, scores_a, y_test, ch_test = run_condition(
        df_to, TELEM_FEATURES, "Telemetry Only (18 features)"
    )

    # Condition B: Telemetry + Space Weather
    summary_b, ch_b, metrics_b, preds_b, scores_b, _, _ = run_condition(
        df_sw, TELEM_FEATURES + SW_FEATURES, "Telemetry + SW (35 features)"
    )

    # ── Statistical Significance ──
    print(f"\n{'='*60}")
    print("STATISTICAL COMPARISON: Condition A vs. Condition B")
    print(f"{'='*60}")

    for metric_name in ['f1', 'au_pr', 'roc_auc', 'accuracy', 'precision', 'recall']:
        vals_a = summary_a[metric_name]['values']
        vals_b = summary_b[metric_name]['values']
        diff = np.array(vals_b) - np.array(vals_a)
        mean_diff = np.mean(diff)

        if np.std(diff) > 0:
            t_stat, p_val = stats.ttest_rel(vals_b, vals_a)
        else:
            t_stat, p_val = 0.0, 1.0

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        direction = "+" if mean_diff > 0 else "-" if mean_diff < 0 else "="

        print(f"  {metric_name:>10s}: A={summary_a[metric_name]['mean']:.4f}  "
              f"B={summary_b[metric_name]['mean']:.4f}  "
              f"diff={direction}{abs(mean_diff):.4f}  "
              f"p={p_val:.4f} {sig}")

    # ── Per-channel delta table ──
    print(f"\n{'='*60}")
    print("PER-CHANNEL F1 DELTA (B - A)")
    print(f"{'='*60}")
    print(f"  {'Channel':<12s} {'F1_A':>6s} {'F1_B':>6s} {'Delta':>8s}")
    for ch in sorted(ch_a.keys()):
        f1_a = ch_a[ch]['f1']
        f1_b = ch_b[ch]['f1']
        delta = f1_b - f1_a
        marker = " <--" if abs(delta) > 0.05 else ""
        print(f"  {ch:<12s} {f1_a:>6.3f} {f1_b:>6.3f} {delta:>+8.3f}{marker}")

    # ── Save results ──
    results = {
        'model': 'IsolationForest',
        'n_estimators': N_ESTIMATORS,
        'seeds': SEEDS,
        'conditions': {
            'telem_only': {
                'n_features': len(TELEM_FEATURES),
                'summary': {k: {'mean': v['mean'], 'std': v['std']} for k, v in summary_a.items()},
                'per_seed': [m for m in metrics_a],
                'per_channel': {k: v for k, v in ch_a.items()}
            },
            'telem_sw': {
                'n_features': len(TELEM_FEATURES) + len(SW_FEATURES),
                'summary': {k: {'mean': v['mean'], 'std': v['std']} for k, v in summary_b.items()},
                'per_seed': [m for m in metrics_b],
                'per_channel': {k: v for k, v in ch_b.items()}
            }
        }
    }

    # Add significance tests
    sig_tests = {}
    for metric_name in ['f1', 'au_pr', 'roc_auc', 'accuracy', 'precision', 'recall']:
        vals_a = summary_a[metric_name]['values']
        vals_b = summary_b[metric_name]['values']
        diff = np.array(vals_b) - np.array(vals_a)
        if np.std(diff) > 0:
            t_stat, p_val = stats.ttest_rel(vals_b, vals_a)
        else:
            t_stat, p_val = 0.0, 1.0
        sig_tests[metric_name] = {
            'mean_diff': float(np.mean(diff)),
            't_stat': float(t_stat),
            'p_value': float(p_val),
            'significant_005': bool(p_val < 0.05)
        }
    results['significance_tests'] = sig_tests

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'if_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # ── Save ablation table as CSV ──
    rows = []
    for cond_name, summary in [('Telem Only', summary_a), ('Telem + SW', summary_b)]:
        row = {'Condition': cond_name}
        for metric_name, vals in summary.items():
            row[f'{metric_name}_mean'] = vals['mean']
            row[f'{metric_name}_std'] = vals['std']
        rows.append(row)
    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(os.path.join(OUTPUT_DIR, 'if_ablation_table.csv'), index=False)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("  - if_results.json (full results)")
    print("  - if_ablation_table.csv (ablation summary)")


if __name__ == '__main__':
    main()
