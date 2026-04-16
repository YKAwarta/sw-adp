"""
SW-ADP Stage 4: Matrix Profile Baseline
Computes 1D matrix profile per segment, extracts features, classifies with IF.
Two conditions: MP features only vs MP features + SW summary stats.

Run: python run_mp_baseline.py
"""
import numpy as np
import stumpy
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score, roc_auc_score)
from scipy import stats
import json, time, warnings, os
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────
TENSOR_PATH = os.path.join('data', 'processed', 'track_b_tensors.npz')
TRACK_A_SW_PATH = os.path.join('data', 'processed', 'track_a_telem_sw.csv')
OUTPUT_DIR = os.path.join('experiments', 'phase1', 'stage4_deep_learning')
SEEDS = [42, 123, 456, 789, 1024]

# ── Try to find files (handle running from repo root or src/models/) ──
for prefix in ['', '../../', '../../../']:
    if os.path.exists(prefix + TENSOR_PATH):
        TENSOR_PATH = prefix + TENSOR_PATH
        TRACK_A_SW_PATH = prefix + TRACK_A_SW_PATH
        OUTPUT_DIR = prefix + OUTPUT_DIR
        break

os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_mp_features(signal, real_len):
    """Extract 6 features from the matrix profile of a 1D signal."""
    sig = signal[:real_len].copy()
    if real_len < 6 or np.std(sig) < 1e-10:
        return np.full(6, np.nan)
    m = max(3, min(10, real_len // 4))
    if real_len <= m + 1:
        m = max(3, real_len // 3)
    if real_len <= m:
        return np.full(6, np.nan)
    try:
        mp = stumpy.stump(sig, m)
        distances = mp[:, 0].astype(float)
        finite = np.isfinite(distances)
        if finite.sum() == 0:
            return np.full(6, np.nan)
        distances[~finite] = np.max(distances[finite])
        return np.array([
            np.max(distances),
            np.min(distances),
            np.mean(distances),
            np.std(distances),
            np.argmax(distances) / len(distances),
            np.mean(distances > np.mean(distances))
        ])
    except Exception:
        return np.full(6, np.nan)


def evaluate_if(X_tr, X_te, y_te, seeds, contam):
    """Run IF across multiple seeds, return list of metric dicts."""
    results = []
    for seed in seeds:
        clf = IsolationForest(n_estimators=200, contamination=contam,
                              random_state=seed, n_jobs=-1)
        clf.fit(X_tr)
        scores = -clf.decision_function(X_te)
        preds = (clf.predict(X_te) == -1).astype(int)
        results.append({
            'accuracy': accuracy_score(y_te, preds),
            'precision': precision_score(y_te, preds, zero_division=0),
            'recall': recall_score(y_te, preds, zero_division=0),
            'f1': f1_score(y_te, preds, zero_division=0),
            'au_pr': average_precision_score(y_te, scores),
            'roc_auc': roc_auc_score(y_te, scores),
        })
    return results


def main():
    print("=" * 60)
    print("MATRIX PROFILE BASELINE")
    print("=" * 60)

    # Load data
    data = np.load(TENSOR_PATH, allow_pickle=True)
    X_telem = data['X_telem'].astype(np.float64)  # stumpy needs float64
    y = data['y']
    mask = data['mask']
    train_mask = data['train_mask'].astype(bool)

    track_a_sw = pd.read_csv(TRACK_A_SW_PATH)
    sw_cols = [c for c in track_a_sw.columns if c.startswith('sw_')]
    sw_features = track_a_sw[sw_cols].values

    # Compute MP features
    print("\nComputing Matrix Profile features...")
    t0 = time.time()
    n = len(y)
    mp_features = np.zeros((n, 6))
    n_failed = 0
    for i in range(n):
        real_len = int(mask[i].sum())
        mp_features[i] = compute_mp_features(X_telem[i, :, 0], real_len)
        if np.any(np.isnan(mp_features[i])):
            n_failed += 1
    mp_features = np.nan_to_num(mp_features, nan=0.0)
    print(f"Done in {time.time()-t0:.1f}s ({n_failed} segments failed/constant)")

    # Setup conditions
    test_idx = ~train_mask
    feat_A = mp_features
    feat_B = np.hstack([mp_features, sw_features])
    contamination = y[train_mask].mean()
    print(f"Features: A={feat_A.shape[1]}, B={feat_B.shape[1]}, contamination={contamination:.3f}")

    # Evaluate
    print("\nCondition A: MP features only...")
    res_A = evaluate_if(feat_A[train_mask], feat_A[test_idx], y[test_idx], SEEDS, contamination)
    print("Condition B: MP features + SW...")
    res_B = evaluate_if(feat_B[train_mask], feat_B[test_idx], y[test_idx], SEEDS, contamination)

    # Summarize
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

    sum_A = summarize(res_A, "Condition A: MP Only")
    sum_B = summarize(res_B, "Condition B: MP + SW")

    # Significance
    print("\nSignificance (B - A):")
    deltas = {}
    for m in res_A[0]:
        va, vb = [r[m] for r in res_A], [r[m] for r in res_B]
        t, p = stats.ttest_rel(va, vb)
        d = np.mean(vb) - np.mean(va)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        deltas[m] = {'delta': round(d, 4), 'p': round(p, 4), 'sig': sig}
        print(f"  {m}: {d:+.4f} p={p:.4f} {sig}")

    # Save
    full_results = {
        'model': 'Matrix Profile + Isolation Forest',
        'mp_features': ['max_dist', 'min_dist', 'mean_dist', 'std_dist',
                        'discord_pos', 'above_mean_ratio'],
        'n_segments_failed': n_failed,
        'condition_A': {'label': 'MP Telem Only', 'n_features': 6,
                        'per_seed': res_A, 'summary': sum_A},
        'condition_B': {'label': 'MP Telem + SW', 'n_features': 23,
                        'per_seed': res_B, 'summary': sum_B},
        'deltas': deltas,
        'seeds': SEEDS,
    }
    with open(os.path.join(OUTPUT_DIR, 'mp_results.json'), 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    rows = [{'Condition': 'MP_Telem_Only', **sum_A},
            {'Condition': 'MP_Telem_SW', **sum_B}]
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, 'mp_ablation_table.csv'), index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
