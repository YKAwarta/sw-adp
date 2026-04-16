# Space Weather-Aware Anomaly Detection & Prognostics for CubeSats (SW-ADP)

**Status:** Phase 1 Complete — 1st Place, College of Engineering Division, Alfaisal University Research & Design Expo (April 2, 2026). Phase 2 in progress.

## Overview

Traditional CubeSat anomaly detectors analyze only onboard telemetry. When a geomagnetic storm strikes, the detector sees the symptom but not the cause. SW-ADP investigates whether fusing space weather indices (Kp, Dst, F10.7, solar wind speed, proton density, IMF Bz) with CubeSat telemetry improves anomaly detection vs. telemetry-only baselines.

**Hypothesis:** Space weather fusion improves anomaly detection across all model classes tested.

**Phase 1 result:** The hypothesis is *not supported* on ESA's OPSSAT-AD benchmark. The project's contribution is three findings that emerged from the systematic ablation:

1. Naive fusion (concatenate SW features, reconstruct everything) degrades every model class — p < 0.001 across IF, MP+IF, LSTM-AE, and TCN-AE.
2. Conditional/asymmetric fusion (encode all channels, reconstruct telemetry only) recovers most of the lost performance, proving fusion *architecture* matters more than whether you fuse.
3. TCN-AE telemetry-only reaches F1 = 0.604 — the best unsupervised result reported on OPSSAT-AD to date.

## Phase 1 Models

| Component | Method | Role |
|-----------|--------|------|
| Baseline | Isolation Forest | Statistical point-in-time benchmark |
| Baseline | Matrix Profile (STUMPY) + IF | Temporal pattern benchmark |
| Deep Learning | LSTM Autoencoder (PyTorch) | Established DL comparison |
| **Core** | **TCN Autoencoder (PyTorch)** | **Primary contribution — edge-deployable** |

## Phase 1 Results

Systematic ablation: 4 models × 3 fusion conditions (telemetry-only, naive SW concat, conditional SW fusion) on OPSSAT-AD.

| Model | Telem-Only F1 | Naive Fusion F1 | Conditional F1 |
|-------|---------------|------------------|-----------------|
| Isolation Forest | 0.306 | 0.255 (−16.7%) | — |
| MP + IF | 0.448 | 0.268 (−40.2%) | — |
| LSTM-AE | 0.488 | 0.216 (−55.7%) | 0.429 |
| **TCN-AE** | **0.604** | 0.222 (−63.2%) | 0.376 |

**TCN-AE telem-only:** F1 = 0.604, AU-PR = 0.677, ROC-AUC = 0.753.

Phase 1 technical artifact: `reports/SW-ADP_Phase1_Paper.pdf`. Winning poster: `reports/competition/SW-ADP Poster.pdf` (submission code `UGE0054AWARTANI2026`). Tagged at `v1.0-phase1`.

## Datasets

- **OPSSAT-AD** (ESA / Zenodo) — primary benchmark. 2,098 labeled segments across 7 channels, Jan–Jun 2022 UTC, ~20% anomaly rate. Confirmed (via exhaustive search) as the only public satellite telemetry AD dataset with real timestamps, expert anomaly annotations, and open access.
- **NOAA OMNI2** — hourly Kp, Dst, F10.7, solar wind speed, proton density, IMF Bz. 100% alignment with OPSSAT telemetry via forward-fill (discrete indices) + linear interpolation (continuous measurements).

## Roadmap

**Phase 1 — Proof of Concept** (March 2026) ✅
Systematic 4-model × 3-condition ablation on OPSSAT-AD. 1st-place win, internal research paper drafted as Phase 2 foundation.

**Phase 2 — Conference Paper** (April – November 2026)
Adds semi-synthetic validation (real OPSSAT waveforms + injected anomalies with known SW-correlated vs SW-independent ground truth), conformal prediction calibration, and ≥10-seed statistical rigor. **Target:** IEEE Aerospace Conference 2027 (AeroConf), November 2026 deadline.

**Phase 3 — Journal Paper** (2027)
Real satellite telemetry validation (pending NARSS / ESOC / OPS-SAT VOLT outreach), TinyML edge deployment on ESP32-S3, full PRISMA literature review. **Target:** Acta Astronautica or IEEE Transactions on Aerospace and Electronic Systems (TAES).

## Repository Structure

```
sw-adp/
├── src/
│   ├── pipeline/          # Data ingestion, cleaning, alignment, EDA
│   ├── models/            # IF, MP, LSTM-AE, TCN-AE, conditional variants
│   ├── synthetic/         # Phase 2: semi-synthetic anomaly injection
│   └── calibration/       # Phase 2: conformal prediction
├── experiments/
│   ├── phase1/            # Frozen Phase 1 results (stage3_baselines, stage4_deep_learning, egyptsat_exploration)
│   └── phase2/            # Synthetic / conformal / 10-seed OPSSAT rerun
├── figures/               # Publication-quality EDA and results plots
├── reports/               # Phase 1 paper, literature scan, abstract, winning poster
├── scripts/               # Data download & verification
├── supervisor/            # Supervisor feedback & communications
├── notebooks/             # Jupyter notebooks (placeholder)
└── data/                  # Raw & processed data (git-ignored)
```

## Environment Setup

```bash
conda env create -f environment.yml
conda activate sw-adp
```

## Key Commands

```bash
# Download datasets (from repo root)
python scripts/download_opssat.py
python scripts/download_noaa.py

# Full data pipeline (clean, align, build tracks, EDA)
python src/pipeline/run_pipeline.py [--seq-len 128]

# Phase 1 model runs (from src/models/)
cd src/models
python run_stage4.py          # LSTM-AE + TCN-AE ablation (Conditions A & B)
python run_condition_c.py     # Conditional fusion (Condition C)
python run_mp_baseline.py     # Matrix Profile + IF baseline
```

## Stack

Python 3.11 | PyTorch (CUDA 12.1) | scikit-learn | STUMPY | pandas | numpy | matplotlib

## Owner & Supervisor

Yousef K. Awartani (owner) · Dr. Anis Koubaa (supervisor, confirmed March 4, 2026) · Alfaisal University, Riyadh.
