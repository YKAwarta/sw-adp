# Space Weather-Aware Anomaly Detection for CubeSats (SW-ADP)

**Project Summary**

**Student:** Yousef K. Awartani (Software Engineering, Year 3, AI & Big Data Track)
**GitHub Repository:** [YKAwarta/sw-adp](https://github.com/YKAwarta/sw-adp.git)

---

## 1. The Problem

Current anomaly detection systems for CubeSats rely solely on onboard telemetry (e.g., voltage, temperature, magnetic fields). Because they are "blind" to external environmental factors like geomagnetic storms, they suffer from:

- **Late detections:** Anomalies are only caught after they occur.
- **High false-alarm rates:** Operators lose trust in alerts.
- **Lack of context:** Operators see the symptom but cannot explain the cause.

With "Solar Cycle 25" bringing increased geomagnetic storm frequency, radiation, and drag on satellites, the need for context-aware detectors is critical.

## 2. The Proposed Solution

The project aims to **fuse onboard CubeSat telemetry with external space weather indices** to create a context-aware anomaly detector. This approach seeks to provide earlier warnings, reduce false alarms, and give operators calibrated confidence scores they can trust.

## 3. Technical Approach

### Datasets

- **OPSSAT-AD (ESA):** The primary benchmark containing labeled anomalies across 9 telemetry channels.
- **NOAA SWPC:** Timestamped, open-access space weather indices (Kp, Dst, F10.7, solar wind).
- **ESA Anomaly Dataset:** Used for cross-mission generalization testing.

### Machine Learning Models

- **Baselines:** Isolation Forest and Matrix Profile (STUMPY).
- **Deep Learning:** LSTM Autoencoder for temporal pattern modeling.
- **Calibration:** Online conformal prediction on reconstruction error to calibrate confidence scores.

### Core Experiment (Ablation Study)

The core proof-of-concept will compare model performance using **telemetry-only** vs. **telemetry + space weather features**. The goal is to prove that adding external context significantly improves anomaly detection.

## 4. Success Metrics & Deliverables

### Target Metrics

- **+10%** improvement in NAB Score over the telemetry-only baseline.
- **30+ seconds** of lead time before anomaly onset.
- **≥90%** calibrated alert coverage with reduced false alerts.

### Deliverables

- A reproducible, open-source Python pipeline.
- An operator dashboard (including anomaly overlays, predictions, and confidence stats).
- A documented research paper and poster.
- **Stretch goal:** A working demo quantized for a TinyML model runnable on an ESP32-S3 microcontroller for on-board inference.

## 5. Timeline & Deadlines

- **March 4-17:** Complete data pipeline, baseline models, ablation study, and preliminary results.
- **March 19:** Deadline to submit the abstract/preliminary results for the poster competition.
- **April 2:** 17th Annual Student Research Poster Competition.
- **April - May:** Finalize LSTM autoencoder, dashboard, and conformal calibration.
- **May 2026:** Live demonstration at the Boeing Engineering Design Expo.

---

_Note: Yousef requested a GitHub username or email address to add you as a collaborator to the private repository._
