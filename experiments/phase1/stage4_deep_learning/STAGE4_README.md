# SW-ADP Stage 4: Deep Learning Models

## What's Here

```
src/models/
  data_utils.py        - Data loading, PyTorch datasets, masked MSE loss
  lstm_ae.py           - LSTM Autoencoder model (DL baseline)
  tcn_ae.py            - TCN Autoencoder model (core model)
  train_eval.py        - Shared training loop, evaluation, ablation runner
  run_mp_baseline.py   - Matrix Profile baseline (standalone, already has results)
  run_stage4.py        - Master runner: LSTM-AE + TCN-AE + combined ablation table
```

## Prerequisites

Your `sw-adp` conda environment should already have everything. Verify:

```bash
conda activate sw-adp
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import stumpy; print('stumpy OK')"
python -c "from sklearn.metrics import f1_score; print('sklearn OK')"
```

Expected: PyTorch 2.x with CUDA=True.

## File Layout Expected

The scripts expect this structure from repo root:

```
sw-adp/
  data/processed/
    track_b_tensors.npz        <- your Track B tensors
    track_a_telem_sw.csv       <- Track A with SW features (for MP baseline)
  experiments/
    stage3_baselines/
      if_ablation_table.csv    <- IF results from Stage 3
    stage4_deep_learning/      <- output directory (created automatically)
  src/models/
    data_utils.py
    lstm_ae.py
    tcn_ae.py
    train_eval.py
    run_mp_baseline.py
    run_stage4.py
```

If your processed data is in a different location, edit the paths at the top of
run_stage4.py (TENSOR_PATH, TRACK_A_SW_PATH, IF_RESULTS_PATH).

## Step-by-Step

### Step 1: Copy files into your repo

Copy all 6 .py files into `src/models/` in your sw-adp repo.

### Step 2: Run Matrix Profile baseline (optional, ~60s)

MP results are already confirmed (see below). If you want to regenerate:

```bash
cd src/models
python run_mp_baseline.py
```

This saves results to `experiments/stage4_deep_learning/mp_results.json`.

### Step 3: Run LSTM-AE and TCN-AE (the main event, ~10-15 min on GPU)

```bash
cd src/models
python run_stage4.py
```

This will:
1. Load IF results from Stage 3
2. Load MP results (if available)
3. Train LSTM-AE under both conditions (3 seeds each)
4. Train TCN-AE under both conditions (3 seeds each)
5. Generate training curve plots
6. Print the full combined ablation table
7. Save everything to `experiments/stage4_deep_learning/`

### Step 4: Check outputs

After running, you should have:

```
experiments/stage4_deep_learning/
  mp_results.json              - MP full results
  mp_ablation_table.csv        - MP summary
  lstm_ae_results.json         - LSTM-AE full results
  tcn_ae_results.json          - TCN-AE full results
  lstm_ae_training_curves.png  - Loss curves
  tcn_ae_training_curves.png   - Loss curves
  combined_ablation_table.csv  - THE BIG TABLE: 4 models x 2 conditions
```

## Confirmed Results: Matrix Profile

Already run and verified. These numbers are locked in:

| Condition | Accuracy | Precision | Recall | F1 | AU-PR | ROC-AUC |
|---|---|---|---|---|---|---|
| MP Telem Only (6 feat) | 0.756 +/- 0.007 | 0.428 +/- 0.017 | 0.471 +/- 0.040 | 0.448 +/- 0.027 | 0.414 +/- 0.009 | 0.746 +/- 0.007 |
| MP Telem + SW (23 feat) | 0.672 +/- 0.003 | 0.252 +/- 0.006 | 0.285 +/- 0.007 | 0.268 +/- 0.006 | 0.237 +/- 0.007 | 0.510 +/- 0.011 |
| Delta (B - A) | -0.085*** | -0.175*** | -0.186** | -0.180*** | -0.177*** | -0.235*** |

Same pattern as IF: SW features hurt non-learned models. Curse of dimensionality.

## Confirmed Results: Isolation Forest (from Stage 3)

| Condition | Accuracy | Precision | Recall | F1 | AU-PR | ROC-AUC |
|---|---|---|---|---|---|---|
| IF Telem Only (18 feat) | 0.699 +/- 0.008 | 0.297 +/- 0.015 | 0.316 +/- 0.016 | 0.306 +/- 0.015 | 0.331 +/- 0.015 | 0.629 +/- 0.009 |
| IF Telem + SW (35 feat) | 0.696 +/- 0.006 | 0.263 +/- 0.011 | 0.247 +/- 0.013 | 0.255 +/- 0.011 | 0.247 +/- 0.003 | 0.510 +/- 0.006 |

## Architecture Details

### LSTM Autoencoder (DL Baseline)
- Encoder: 2-layer LSTM (hidden_dim=64)
- Bottleneck: last hidden state repeated across sequence
- Decoder: 2-layer LSTM + linear projection
- Training: normal segments only, masked MSE loss

### TCN Autoencoder (Core Model)
- Encoder: 3 TCN blocks with dilations [1, 2, 4], channels [16, 32, 64]
- Kernel size: 3 (receptive field ~29 timesteps)
- BatchNorm + residual connections in each block
- Decoder: mirrored TCN blocks [64, 32, 16] + 1x1 conv output
- Training: same as LSTM-AE

Both models reconstruct the input; reconstruction error = anomaly score.
Threshold set at the (1-contamination) percentile of test scores.

## Troubleshooting

**CUDA out of memory**: Reduce BATCH_SIZE to 16 in run_stage4.py.

**Training too slow**: The default 100 epochs with patience=10 means it'll stop
early. On RTX 4060Ti expect ~2-3 min per model per condition.

**Path errors**: Edit the constants at the top of run_stage4.py. The script tries
to find the repo root by going up 2 directories from its own location.
