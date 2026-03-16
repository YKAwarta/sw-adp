# Response to Supervisor Feedback -- March 4, 2026

**Student:** Yousef Khalid Awartani
**Supervisor:** Dr. Anis Koubaa
**In response to:** supervisor/ folder pushed March 2, 2026 (Proposal Evaluation, Dataset Analysis, Methodology Design)

---

Dr. Koubaa,

Thank you for the thorough review and the structured feedback. I have reviewed all three documents carefully and address each point below with a decision, rationale, and the phase in which it will be implemented.

---

## 1. Data Alignment Strategy

**Your feedback:** Flagged as "Crucial." NOAA space weather indices (Kp: 3-hourly, Dst: hourly, F10.7: daily) operate at drastically different temporal resolutions than OPSSAT telemetry (Hz-frequency). The merge and resampling approach must be explicitly defined.

**Response:** Agreed. This will be addressed directly in the data pipeline. The alignment strategy will be physically justified rather than arbitrary:

- **Kp index (3-hourly):** Forward-fill. Kp represents a fixed planetary disturbance value for its entire 3-hour window. Forward-filling reflects the actual measurement semantics -- the value does not change between readings.
- **F10.7 solar flux (daily):** Forward-fill. Same rationale -- the daily value characterizes that full day's solar activity.
- **Dst index (hourly):** Forward-fill. The value represents the hour's average ring current strength.
- **Solar wind speed and density (~1-min cadence from ACE/DSCOVR):** Linear interpolation. Unlike the discrete indices above, solar wind parameters are continuous physical quantities with high temporal resolution, making interpolation between readings physically defensible.

**Rule:** Forward-fill for discrete indices that represent fixed time windows; linear interpolation for continuous physical measurements. This will be documented in the pipeline code, in the poster methods section, and in the paper.

**Phase:** Phase 1. Implemented during data pipeline construction (March 6-9).

---

## 2. Feature Justification

**Your feedback:** Explicitly state which of the 9 OPSSAT telemetry channels are most likely to show space weather effects and why. Magnetometers and power systems were noted as primary candidates.

**Response:** Agreed. The 9 OPSSAT-AD channels are 3 magnetometers and 6 photodiodes. The physical justification for space weather sensitivity:

- **Magnetometers (3 channels) -- Primary sensitivity.** Geomagnetic storms (tracked by Kp and Dst indices) directly perturb the local magnetic field at LEO altitude. This is a direct physical coupling: elevated Kp/Dst values correspond to increased geomagnetic disturbance, which the magnetometers measure. These are the channels where we most strongly hypothesize that space weather features will improve anomaly detection.

- **Photodiodes (6 channels) -- Secondary sensitivity.** These are sun sensors used for attitude determination, measuring light intensity. Solar flares increase UV/EUV flux, and F10.7 correlates with overall solar radiative output. The coupling is real but less direct than the magnetometer case.

**Action:** This hypothesis will be stated explicitly before any experiments are run. During EDA, we will compute correlation analysis between each OPSSAT channel and the space weather indices during known anomaly windows to confirm (or challenge) this expectation. The correlation results will produce a channel-SW heatmap figure for the poster.

**Phase:** Phase 1. Hypothesis stated in pipeline documentation; confirmed via EDA (March 9-10).

---

## 3. TCN as Core Model

**Your feedback:** LSTMs are difficult to compress onto an ESP32-S3 for edge inference. TCNs (1D Dilated Causal Convolutions) are faster to train, easier to parallelize, and significantly more TinyML-friendly. Recommended as an alternative to the LSTM Autoencoder.

**Response:** Adopting TCN as the core model for this project.

The updated model lineup for Phase 1 is:

| Model | Role | Rationale |
|---|---|---|
| Isolation Forest | Point-in-time baseline | Fast, simple, sets the statistical floor |
| Matrix Profile (STUMPY) | Temporal baseline | Captures sequential patterns without learning |
| LSTM Autoencoder | Deep learning baseline | Established method in spacecraft AD literature (Hundman et al., KDD 2018). Included for completeness and comparison |
| **TCN Autoencoder** | **Core contribution** | Captures temporal dependencies via 1D dilated causal convolutions. Faster training, parallelizable, and directly portable to ESP32-S3 via TFLite-Micro |

The ablation study now runs all four models under both conditions (telemetry-only vs. telemetry + space weather). This strengthens the science: if the space weather fusion hypothesis holds across all four model classes, the result is more robust than proving it on a single architecture.

The LSTM-AE is retained not as the primary model, but as a meaningful comparison point. It ensures reviewers and judges who expect LSTM-based approaches (the current standard in spacecraft AD) see that we are aware of the established work and can directly benchmark against it.

TCN carries forward from Phase 1 PoC directly into Phase 2 edge deployment, providing a clean narrative: "Proved in research, deployed on hardware."

**Phase:** Phase 1. TCN is core; LSTM-AE is a DL comparison.

---

## 4. Transformer-Based Models

**Your feedback:** Transformers (e.g., Anomaly Transformer, TranAD) use self-attention to capture long-range dependencies and currently achieve state-of-the-art accuracy on benchmarks like SMAP/MSL.

**Response:** Acknowledged as the current accuracy leaders in the literature. However, I do not believe transformers are suitable for Phases 1-2 of this project for a specific reason: our application focuses on-board, edge-deployable inference on resource-constrained hardware (ESP32-S3, anywhere betwee 512KB-8MB RAM). Transformer architectures are too computationally heavy and memory-intensive for this deployment and hardware target.

Transformers are definitely a Phase 3 consideration. If the project proceeds to publication, Transformer results could serve as an accuracy ceiling comparison, demonstrating where our edge-deployable TCN stands relative to the best unconstrained models. This would strengthen the paper's contribution by framing the tradeoff between accuracy and deployability.

**Phase:** Phase 3 (publication), as an accuracy ceiling benchmark. Not actively in scope for Phase 1-2.

---

## 5. Graph Neural Networks (GNNs)

**Your feedback:** GNNs model the structural, physical relationships between satellite components (e.g., battery temperature connected to battery voltage, Kp index affecting magnetometers). This enables root-cause explainability rather than generic anomaly scores.

**Response:** This is a very compelling suggestion, particularly for operator-facing diagnostics, which is a big focus of the project. However, GNNs require defining an explicit graph topology over the sensor network and add significant implementation complexity. For Phase 1-2, explainability is covered by:

- **SHAP values** for the baseline models (Isolation Forest, Matrix Profile)
- **Integrated Gradients** for the deep learning models (TCN, LSTM-AE)

These provide feature-level attribution (which inputs contributed most to each anomaly flag) without requiring a predefined graph structure.

GNNs are noted as a strong Phase 3 candidate. If the project reaches the publication stage, a GNN-based explainability comparison would add significant value, particularly for demonstrating how physical sensor relationships interact with space weather perturbations.

**Phase:** Phase 3 (publication). Phase 1-2 explainability currently handled by SHAP and Integrated Gradients.

---

## 6. NASA SMAP/MSL Dataset

**Your feedback:** Testing the model on SMAP/MSL data would add significant credibility, as it is the most widely used benchmark in spacecraft anomaly detection literature.

**Response:** Researched in detail. SMAP/MSL is indeed the dominant benchmark, with extensive published results to compare against (Hundman et al. 2018, OmniAnomaly 2019, TranAD, Anomaly Transformer, and many others).

However, there is a critical constraint for our specific project: the SMAP/MSL dataset has been **fully anonymized** with respect to time. All timestamps have been removed, and telemetry values are pre-scaled to (-1, 1). Channel IDs are also anonymized (only the first letter indicates type: P = power, R = radiation, etc.).

This means we cannot align SMAP/MSL telemetry with NOAA space weather indices, which is the core experiment of this project. Without real timestamps, there is no way to test the space weather fusion hypothesis on this data.

SMAP/MSL is still valuable for a different purpose, however: demonstrating that our model architectures (TCN, LSTM-AE) perform competitively against published benchmarks on a standard dataset, independent of the space weather fusion. This is a credibility argument for the paper.

**Decision:** Deferred to Phase 3. In the publication, SMAP/MSL results will be included as an architecture benchmark (TCN vs. published LSTM/Transformer results on the same dataset), separate from the SW fusion ablation which requires timestamped data.

**Phase:** Phase 3 (publication). Not suitable for the core SW fusion experiment in phases 1 and 2 due to timestamp anonymization.

---

## 7. SMD (Server Machine Dataset)

**Your feedback:** Suggested as a non-space control dataset for multivariate time-series anomaly detection.

**Response:** SMD (from the OmniAnomaly KDD 2019 paper) is useful for demonstrating that our anomaly detection pipeline generalizes beyond the space domain.

For Phase 1-2, our focus is on proving the space weather fusion hypothesis using space-specific data (OPSSAT-AD as primary, ESA Anomaly Dataset for Phase 2 cross-mission generalization). Adding a non-space dataset does not strengthen the core argument in phase 1, but will be considered in phase 3 if it reaches the publication stage.

For Phase 3, SMD could serve as a domain-transfer test: "Does the TCN architecture trained on spacecraft telemetry transfer to non-space multivariate time series?" This is a useful generalization claim for a publication.

**Decision:** Deferred to Phase 3 as a domain-transfer generalization test.

**Phase:** Phase 3 (publication).

---

## Summary of Decisions

| # | Feedback Point | Decision | Phase |
|---|---|---|---|
| 1 | Data alignment strategy | Forward-fill for discrete indices, linear interpolation for continuous measurements. Physically justified. | Phase 1 |
| 2 | Feature justification | Magnetometers primary, photodiodes secondary. Confirm with EDA before experiments. | Phase 1 |
| 3 | TCN suggestion | Adopted as core model. LSTM-AE becomes DL baseline comparison. | Phase 1 |
| 4 | Transformers | Noted for Phase 3 accuracy ceiling comparison. Too heavy for edge deployment target. | Phase 3 |
| 5 | GNNs | Noted for Phase 3 explainability. SHAP/Integrated Gradients cover Phase 1-2. | Phase 3 |
| 6 | SMAP/MSL | Timestamps anonymized; cannot test SW fusion during Phase 1 and 2. Architecture benchmark for paper during Phase 3. | Phase 3 |
| 7 | SMD | Domain-transfer generalization test for paper during Phase 3. | Phase 3 |

---

## Updated Phase 1 Timeline (reflecting TCN addition)

| Milestone | Target Date |
|---|---|
| Dev environment setup (conda + CUDA) | March 4-5 |
| Dataset download (OPSSAT-AD + NOAA) | March 5-6 |
| Data pipeline (ingest, align, window) | March 6-9 |
| EDA + feature justification confirmation | March 9-10 |
| Baselines: IF + MP (both conditions) | March 10-12 |
| LSTM-AE (both conditions) | March 12-14 |
| TCN-AE (both conditions) | March 14-16 |
| Ablation table + publication-quality figures | March 16-18 |
| Abstract submitted (200 words) | March 19 |
| Poster design + iteration | March 19-28 |
| Poster submitted | March 29 |
| In-person judging | April 2 |

I welcome any feedback or corrections on these decisions. If you would prefer TCN to fully replace LSTM-AE in Phase 1 rather than running both, I am happy to adjust. My reasoning for including both is that LSTM-AE is the established baseline in the spacecraft AD literature and strengthens the ablation, but I defer to your judgment on whether that added comparison justifies the additional compute.

Best regards,
Yousef Khalid Awartani
