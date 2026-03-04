# Space Weather-Aware Anomaly Detection & Prognostics for CubeSats (SW-ADP)

**Status:** Active - Phase 1 (Proof of Concept)

## Overview

Traditional CubeSat anomaly detectors analyze only onboard telemetry. When a geomagnetic storm strikes during Solar Cycle 25's current peak, the detector sees the symptom but not the cause. This is caused by late detection, high false-alarm rates, and alerts operators cannot verify.

This project fuses CubeSat & satellite telemetry with space weather indices from NOAA (Kp, Dst, F10.7, solar wind) and applies calibrated anomaly detection to the combined signal. The goal is earlier warnings with statistically guaranteed confidence scores.

**Core research hypothesis:** Space weather context improves anomaly detection accuracy and lead time vs. telemetry-only baselines, across all model classes tested (isolation forest, matrix profile, LSTM-AE).

## Approach

| Component | Method | Role |
|-----------|--------|------|
| Baselines | Isolation Forest, Matrix Profile (STUMPY) | Benchmarks |
| Model | LSTM Autoencoder (PyTorch) | Core contribution |
| Calibration | Online Conformal Prediction | Guaranteed confidence |
| Explainability | SHAP, Integrated Gradients | Operator-facing |
| TinyML (demo goal) | TFLite-Micro on ESP32-S3 | On-board feasibility |

## Datasets

- **OPSSAT-AD** (ESA/Zenodo) - primary benchmark, 9-channel labeled anomalies
- **ESA Anomaly Dataset** - multi-mission labeled anomalies, for cross-mission generalization
- **NOAA SWPC** - Kp, Dst, F10.7 indices, open access

## Roadmap

**Phase 1 - Proof of Concept** (March 2026)
Baseline + LSTM-AE + ablation. Target: Alfaisal Student Poster Competition (April 2, 2026)

**Phase 2 - MVP** (April-May 2026)
Full pipeline + dashboard. Target: Boeing Engineering Design Expo (May 2026)

**Phase 3 - Publication** (post-Boeing)
Full paper. Target: IAC student track or IEEE Aerospace workshop.

## Stack
Python 3.11 | PyTorch | scikit-learn | STUMPY | pandas | numpy | matplotlib | Streamlit
