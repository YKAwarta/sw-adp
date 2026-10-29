# Space Weather-Aware Anomaly Detection for CubeSats (SW-ADP)

The first study, a controlled ablation on ESA's OPSSAT-AD benchmark, is complete. It won 1st place in the College of Engineering division of Alfaisal University's Research and Design Expo (April 2026). A revised manuscript is in preparation, and a follow-up study is planned for the International Astronautical Congress (IAC 2027).

## Overview

SW-ADP tests whether adding space weather indices (Kp, Dst, F10.7, solar wind speed, proton density and IMF Bz) to CubeSat telemetry improves unsupervised anomaly detection compared with telemetry alone.

The hypothesis, stated before any experiments, was that space weather fusion would improve detection for every model class tested. The results on OPSSAT-AD do not support it.

1. Naive concatenation of the space weather features lowered F1 for all four detectors. All four drops remain significant after Holm correction, and ROC-AUC fell to between 0.495 and 0.510, close to chance.
2. A conditional design, in which the encoder sees all seven channels and the decoder reconstructs only telemetry, recovered part of the loss. LSTM autoencoder F1 rose from 0.216 to 0.429 (Holm-adjusted p = 0.044) and TCN autoencoder F1 from 0.222 to 0.376 (adjusted p = 0.22, not significant). Neither reached its telemetry-only baseline.
3. The proposed cause of the loss is dilution. In the autoencoders, six of the seven channels in the anomaly score are smooth index channels that are never labeled anomalous, so well-reconstructed indices can mask high error on the telemetry channel. In Isolation Forest, the 17 added index statistics make up about half of the features, so more random splits are wasted. Neither mechanism has been isolated experimentally yet.
4. OPSSAT-AD cannot settle whether fusion can help. Its anomalies are labeled by signal shape, OPS-SAT was an experimental platform that expected frequent software faults, and the recording period (January to June 2022) saw only moderate geomagnetic activity, with Kp peaking at 6.7.

The telemetry-only TCN autoencoder (F1 0.604, AU-PR 0.677, ROC-AUC 0.753) was the strongest detector in this study. Among the unsupervised models compared in the revised manuscript, it is second on AU-PR to MO-GAAL (AU-PR 0.779, Ruszczak et al., 2025).

## Models and conditions

| Detector | Conditions tested |
|----------|-------------------|
| Isolation Forest | A, B |
| Matrix Profile (STUMPY) + Isolation Forest | A, B |
| LSTM autoencoder (PyTorch) | A, B, C |
| TCN autoencoder (PyTorch) | A, B, C |

Condition A uses telemetry only. Condition B concatenates the space weather indices with telemetry. Condition C is the conditional design, tested on the autoencoders only.

## Results (F1)

| Model | A: telemetry only | B: naive fusion | C: conditional |
|-------|-------------------|-----------------|----------------|
| Isolation Forest | 0.306 | 0.255 | not tested |
| MP + IF | 0.448 | 0.268 | not tested |
| LSTM autoencoder | 0.488 | 0.216 | 0.429 |
| TCN autoencoder | 0.604 | 0.222 | 0.376 |

Values are means over seeds: five for Isolation Forest and MP + IF, three for the autoencoders. Conditions are compared with paired t-tests, Holm-corrected over eight tests. Three seeds give little statistical power, so the autoencoders will be rerun with more seeds.

## Reports

- `reports/competition/SW-ADP Poster.pdf`: the winning poster (April 2026).
- The original internal report (March 2026) and competition abstract (April 2026) are kept at tag `v1.0-phase1` for the record. The revised manuscript supersedes them. The revision corrects the statistics (Holm correction instead of uncorrected p-values), credits the space weather data to NASA OMNI2 and CelesTrak rather than NOAA, treats the failure mechanisms as proposed rather than identified, and withdraws the claim that the TCN autoencoder is the best unsupervised result on OPSSAT-AD.

## Datasets

- **OPSSAT-AD** (ESA, on Zenodo, described by Ruszczak et al., 2025): telemetry from the OPS-SAT 3U CubeSat, recorded between 4 January and 2 June 2022 UTC. After two sparse channels are excluded, 2,098 labeled segments from seven channels remain, of which 434 (20.7%) are anomalous. It was chosen because it combines real UTC timestamps, expert anomaly labels and open access, which alignment with space weather data requires.
- **Space weather indices** from NASA OMNI2 and CelesTrak, aligned to each telemetry segment by forward-filling the discrete indices and linearly interpolating the continuous measurements.

## Planned follow-up study (IAC 2027)

The follow-up study planned for the International Astronautical Congress 2027 will:

- test directly whether the labeled anomalies depend on the space weather state,
- run control experiments on the proposed dilution mechanisms, including random-feature controls and rescoring the autoencoders on telemetry error alone,
- inject semi-synthetic anomalies into real OPSSAT waveforms, with their occurrence tied to space weather, to test whether the conditional design can use context when the labels do depend on it,
- rerun the autoencoders with more seeds.

Later work, after IAC 2027: storm-period and real satellite telemetry, and on-device deployment on an ESP32-S3 microcontroller.

## Repository structure

`experiments/phase1` holds the completed ablation. `experiments/phase2` holds the follow-up study.

```
sw-adp/
├── src/
│   ├── pipeline/          # Data ingestion, cleaning, alignment, EDA
│   ├── models/            # IF, MP, LSTM-AE, TCN-AE, conditional variants
│   ├── synthetic/         # Follow-up study: semi-synthetic anomaly injection
│   └── calibration/       # Follow-up study: threshold calibration
├── experiments/
│   ├── phase1/            # Frozen results of the completed ablation
│   └── phase2/            # Follow-up study experiments
├── figures/               # EDA and results plots
├── reports/               # Literature scan, abstract, winning poster
├── scripts/               # Data download & verification
├── notebooks/             # Jupyter notebooks (placeholder)
└── data/                  # Raw & processed data (git-ignored)
```

## Environment setup

```bash
conda env create -f environment.yml
conda activate sw-adp
```

## Key commands

```bash
# Download datasets (from repo root)
python scripts/download_opssat.py
python scripts/download_noaa.py

# Full data pipeline (clean, align, build tracks, EDA)
python src/pipeline/run_pipeline.py [--seq-len 128]

# Model runs for the completed ablation (from src/models/)
cd src/models
python run_stage4.py          # LSTM-AE + TCN-AE ablation (Conditions A & B)
python run_condition_c.py     # Conditional fusion (Condition C)
python run_mp_baseline.py     # Matrix Profile + IF baseline
```

## Stack

Python 3.11 | PyTorch (CUDA 12.1) | scikit-learn | STUMPY | pandas | numpy | matplotlib

## Owner and supervisor

Yousef K. Awartani (owner) · Dr. Anis Koubaa (supervisor) · Alfaisal University, Riyadh.
