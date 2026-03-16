# Space Weather-Aware Anomaly Detection & Prognostics for CubeSats (SW-ADP)

**Status:** Phase 1 Complete — Phase 2 in progress

## Overview

Traditional CubeSat anomaly detectors analyze only onboard telemetry. When a geomagnetic storm strikes during Solar Cycle 25's current peak, the detector sees the symptom but not the cause. This leads to late detection, high false-alarm rates, and alerts operators cannot verify.

This project fuses satellite telemetry with space weather indices from NOAA (Kp, Dst, F10.7, solar wind) and applies calibrated anomaly detection to the combined signal. The goal is earlier warnings with statistically guaranteed confidence scores.

**Core hypothesis:** Space weather context improves anomaly detection accuracy vs. telemetry-only baselines, across multiple model classes.

## Phase 1 Models

| Component | Method | Role |
|-----------|--------|------|
| Baseline | Isolation Forest | Statistical point-in-time benchmark |
| Baseline | Matrix Profile (STUMPY) | Temporal pattern benchmark |
| Deep Learning | LSTM Autoencoder (PyTorch) | Established DL comparison |
| **Core** | **TCN Autoencoder (PyTorch)** | **Primary contribution — edge-deployable** |

## Phase 1 Results

Systematic ablation across 4 models x 3 conditions (telemetry-only, telemetry+SW naive concat, conditional SW fusion) on OPSSAT-AD:

| Model | Condition | F1 | ROC-AUC |
|-------|-----------|-----|---------|
| TCN-AE | Telemetry-only | **0.604** | **0.753** |
| LSTM-AE | Telemetry-only | 0.488 | 0.719 |
| MP+IF | Telemetry-only | 0.448 | 0.746 |
| IF | Telemetry-only | 0.306 | 0.629 |
| TCN-AE | Conditional SW | 0.376 | 0.613 |
| LSTM-AE | Conditional SW | 0.429 | 0.670 |

**Key finding:** Naive SW feature concatenation degrades all models (curse of dimensionality). Conditional fusion partially recovers deep learning performance but does not yet surpass telemetry-only baselines — motivating Phase 2 work on attention-based fusion and feature selection.

## Datasets

- **OPSSAT-AD** (ESA/Zenodo) — primary benchmark, 9-channel labeled anomalies with real UTC timestamps
- **NOAA SWPC** — Kp, Dst, F10.7, solar wind speed, proton density, IMF Bz
- **ESA Anomaly Dataset** — planned for Phase 2 cross-mission generalization

## Roadmap

**Phase 1 - Proof of Concept** (March 2026) ✅
Data pipeline, EDA, 4-model ablation study. Target: Alfaisal Student Poster Competition (April 2, 2026)

**Phase 2 - MVP** (April–May 2026)
Full pipeline + Streamlit dashboard + conformal calibration. Target: Boeing Engineering Design Expo (May 2026)

**Phase 3 - Publication** (post-Boeing)
Full paper with Transformer ceiling comparison, SMAP/MSL architecture benchmark. Target: IAC student track or IEEE Aerospace workshop.

## Repository Structure

```
sw-adp/
├── src/
│   ├── pipeline/          # Data ingestion, cleaning, alignment, EDA
│   └── models/            # IF, MP, LSTM-AE, TCN-AE, conditional variants
├── experiments/
│   ├── stage3_baselines/  # Isolation Forest results
│   ├── stage4_deep_learning/  # DL + MP results, ablation tables
│   └── egyptsat_exploration/  # EgyptSat-1 exploratory analysis
├── figures/               # Publication-quality EDA plots
├── reports/               # Paper, literature scan, competition guidelines
├── scripts/               # Data download & verification
├── supervisor/            # Supervisor feedback & communications
├── notebooks/             # Jupyter notebooks (placeholder)
└── data/                  # Raw & processed data (git-ignored)
```

## Stack

Python 3.11 | PyTorch | scikit-learn | STUMPY | pandas | numpy | matplotlib
