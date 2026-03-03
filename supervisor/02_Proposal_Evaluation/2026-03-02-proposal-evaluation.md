# Proposal Evaluation: Space Weather-Aware Anomaly Detection for CubeSats (SW-ADP)

**Student Name:** Yousef K. Awartani  
**Date Evaluated:** March 2, 2026

---

## 1. Overall Review & Strengths

This proposal is exceptionally strong for an undergraduate Capstone and Student Challenge.

**Strengths:**

- **Clear Problem Statement:** Identifying that current onboard anomaly detectors are "blind" to external space weather is a fantastic, novel hook.
- **Rigorous Methodology:** The proposed ablation study (Telemetry-Only vs. Telemetry + Space Weather) is the exactly right scientific way to prove the hypothesis.
- **Deliverables:** Aiming for a live TinyML demonstration on an ESP32-S3 microchip is a massive selling point for a design expo, putting the work firmly into the "edge AI" category.

## 2. Areas for Improvement (How to Win the Challenge)

To take this from a "good" proposal to a "winning" proposal, the following areas need to be tightened:

- **Data Alignment Strategy (Crucial):** Space weather indices from NOAA (like Kp or F10.7) are often updated every hour or every 3 hours. Satellite telemetry (OPSSAT) is recorded multiple times per second (Hz). The proposal must briefly address _how_ it will merge and upsample/downsample these drastically different timeframes.
- **Feature Justification:** The student should explicitly state _which_ of the 9 OPSSAT telemetry channels are most likely to show space weather effects (e.g., magnetometers reacting to geomagnetic storms, or power systems reacting to radiation).
- **Edge/TinyML Feasibility:** The jump from an LSTM Autoencoder to an ESP32-S3 microcontroller is extremely difficult, as RNNs (LSTMs) are notoriously hard to run on TinyML architecture compared to standard Convolutional layers. They should clarify if the TinyML model will be a smaller, quantized version or a different algorithm entirely (like a Random Forest or simple 1D-CNN).

## 3. Suitability of Proposed Datasets

**Are the proposed datasets suitable? Yes, highly suitable.**

- **OPSSAT-AD** is perfectly chosen because it is real, annotated data from an actual CubeSat (OPS-SAT).
- **NOAA SWPC** provides the globally accepted standard for space weather indices.
- **ESA Anomaly Dataset** is a great addition for cross-mission testing.

**Other Datasets in this Context:**
If the student needs alternative or supplementary datasets, they should consider:

- **NASA SMAP (Soil Moisture Active Passive) & MSL (Mars Curiosity) Telemetry Datasets:** These are the most famous benchmark datasets for spacecraft anomaly detection and are publicly available (e.g., on Kaggle). Testing the model on SMAP data would add immense credibility.
- **SMD (Server Machine Dataset):** While not space-related, it is standard for multivariate time-series anomaly detection and acts as a good control dataset.

## 4. AI Algorithms to Test

The proposal currently lists Isolation Forest, STUMPY, and an LSTM Autoencoder. To be competitive and cutting-edge, consider adding or swapping in these algorithms:

- **TCN (Temporal Convolutional Networks):** TCNs are generally faster to train and often outperform LSTMs on time-series anomaly detection. Crucially, 1D-CNNs (which make up a TCN) are _much_ easier to port to TinyML (ESP32) than LSTMs.
- **Transformer-based Models (e.g., Anomaly Transformer):** Transformers are currently state-of-the-art for multivariate time-series data.
- **Graph Neural Networks (GNNs):** For modeling the relationships _between_ different sensors (e.g., how the battery temperature relates to the solar panel voltage).

## 5. Winning Objectives for the Student Challenge

To frame this as a guaranteed winner for the Boeing Engineering Design Expo and Poster Competition, the objectives should be pitched as follows:

1.  **The "Context-Aware" Hook:** Shift the narrative from "finding bugs" to "giving satellites environmental awareness."
2.  **Explainability:** Operators don't trust black boxes. The objective should highlight the use of **SHAP values** or **Conformal Prediction** (already mentioned by the student) to tell operators _why_ the AI flagged an anomaly (e.g., "Alert: Voltage drop predicted with 92% confidence due to incoming Kp-7 Solar Storm").
3.  **The Hardware Demo:** The objective of deploying an offline, localized AI to an ESP32 microcontroller perfectly addresses an industry need: Satellites cannot always communicate with Earth to access large cloud models, so the AI _must_ run on cheap, low-power chips in orbit.

---

**Summary Verdict:** Ensure the student focuses heavily on the "Time-series alignment" of the datasets and considers swapping the LSTM for a TCN to ensure the final hardware demo is successful. This has the makings of a first-place capstone project.
