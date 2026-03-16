# Stage 3 Baseline Results: Isolation Forest

**Date:** March 6, 2026
**Model:** Isolation Forest (n_estimators=200, contamination=0.197)
**Seeds:** 42, 123, 456, 789, 1024 (5 runs, mean +/- std reported)

---

## Ablation Table

| Condition | Accuracy | Precision | Recall | F1 | AU-PR | ROC-AUC |
|---|---|---|---|---|---|---|
| Telem Only (18 feat) | 0.699 +/- 0.008 | 0.297 +/- 0.015 | 0.316 +/- 0.016 | 0.306 +/- 0.015 | 0.331 +/- 0.015 | 0.629 +/- 0.009 |
| Telem + SW (35 feat) | 0.696 +/- 0.006 | 0.263 +/- 0.011 | 0.247 +/- 0.013 | 0.255 +/- 0.011 | 0.247 +/- 0.003 | 0.510 +/- 0.006 |
| **Delta (B - A)** | **-0.003 (ns)** | **-0.034 (**)** | **-0.069 (***)** | **-0.052 (***)** | **-0.083 (***)** | **-0.119 (***)** |

Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant (paired t-test)

**Result: Adding SW features significantly HURTS Isolation Forest performance across all ranking metrics.**

---

## Per-Channel F1 (median seed)

| Channel | Type | Test N | Anomalies | F1 (Telem) | F1 (Telem+SW) | Delta |
|---|---|---|---|---|---|---|
| CADC0872 | Mag-X | 132 | 32 | 0.216 | 0.400 | +0.184 |
| CADC0873 | Mag-Y | 153 | 31 | 0.171 | 0.218 | +0.047 |
| CADC0874 | Mag-Z | 52 | 23 | 0.550 | 0.324 | -0.226 |
| CADC0884 | Photo-1 | 36 | 0 | 0.000 | 0.000 | 0.000 |
| CADC0888 | Photo-2 | 64 | 12 | 0.410 | 0.077 | -0.333 |
| CADC0892 | Photo-3 | 53 | 7 | 0.194 | 0.235 | +0.042 |
| CADC0894 | Photo-4 | 33 | 5 | 0.435 | 0.296 | -0.138 |

No consistent pattern by sensor type. SW features help Mag-X but hurt Mag-Z. Help Photo-3 slightly but devastate Photo-2.

---

## Interpretation

1. **IF performance is far below the OPSSAT-AD RF benchmark (~95.7% accuracy).** This is expected: IF is unsupervised (uses contamination rate but not labels), while RF is fully supervised. IF at ~70% accuracy with F1~0.31 is a reasonable unsupervised baseline on this data.

2. **SW features hurt IF because of the curse of dimensionality.** IF randomly selects features and split points. Going from 18 to 35 features means each tree has a lower probability of selecting an informative telemetry feature for its splits. The 17 SW features (summary stats over the segment's time window) are weakly correlated with the binary anomaly label at the segment level, so they dilute the signal.

3. **This is a meaningful negative result for the ablation.** It tells us that naive feature concatenation does not help a non-temporal, non-learned model. The hypothesis is that models which can learn conditional relationships (autoencoders, or even supervised classifiers) will benefit from SW context where IF cannot.

---

## Matrix Profile Status

MP cannot be run on Track A data. MP operates on raw time-series waveforms to find subsequence discords, not on pre-extracted feature vectors. Running MP requires access to the raw segment waveforms from `segments.csv` or the `track_b_tensors.npz` file, neither of which is available in the current project files.

**Options:**
- Upload `track_b_tensors.npz` to project files, or
- Upload `segments.csv` (raw OPSSAT-AD data), or
- Defer MP to Stage 4 when working with Track B data directly

---

## Files

- `isolation_forest.py` -- full reproducible script
- `if_results.json` -- complete results (per-seed, per-channel, significance tests)
- `if_ablation_table.csv` -- summary table
