# Stage 2 Complete: Data Pipeline and Exploratory Data Analysis

**Student:** Yousef K. Awartani
**Date:** March 6, 2026
**Stage:** 2 of 10 (Phase 1 -- Proof of Concept)

---

## Summary

The data pipeline and exploratory data analysis are complete. All data has been ingested, cleaned, aligned, and prepared for model training in two parallel tracks. The pipeline runs end-to-end in approximately 11 seconds and produces all necessary outputs for the next stage (baseline model experiments). Several important corrections and findings emerged from this stage.

---

## Your Feedback -- Actions Taken

### 1. Data Alignment Strategy (flagged as "Crucial")

**Implemented.** OPSSAT telemetry timestamps were matched to hourly NOAA space weather values using a physically justified approach:

- Kp (3-hourly), F10.7 (daily), Dst (hourly): forward-fill, reflecting that these are discrete values representing fixed time windows
- Solar wind speed, proton density, IMF Bz: linear interpolation, reflecting that these are continuous physical quantities

100% of the 2,098 usable segments were successfully aligned. No telemetry timestamps fell outside the NOAA data range.

### 2. Feature Justification (EDA Results)

**Completed -- with an unexpected result.** The initial hypothesis stated that magnetometers would show the strongest space weather sensitivity due to direct geomagnetic coupling. EDA showed the opposite.

Photodiode channels show substantially stronger correlation with space weather indices (mean |r| = 0.349) than magnetometer channels (mean |r| = 0.108). The standout channel is CADC0894, which correlates at r = 0.68 with F10.7 solar flux, r = 0.64 with solar wind speed, and r = 0.61 with Kp index.

The physical explanation: photodiodes directly measure incoming light flux, which varies with solar activity (tracked by F10.7). This coupling is more direct than the magnetometer-geomagnetic coupling at LEO altitude during moderate storm conditions.

This reversal is documented in the figures (see the correlation heatmap below) and in the living document. The original hypothesis has been corrected based on evidence.

---

## Critical Data Corrections

### OPSSAT Telemetry Period

The OPSSAT-AD telemetry covers **January 4 to June 2, 2022 (UTC)**, not 2024. The May 2024 G5 geomagnetic superstorm (Dst = -406 nT, Kp = 9.0) does not overlap with the telemetry data.

Space weather during the actual OPSSAT period was moderate: Kp max of 6.7, Dst min of -91 nT, 75 hours with Kp >= 5, 60 hours with Dst < -50 nT. This is sufficient for the ablation study. The framing on the poster will be "moderate geomagnetic activity during Solar Cycle 25 ascending phase."

### Data Structure

The OPSSAT-AD dataset is **segment-based and univariate**, not a continuous multivariate time series. It consists of 2,123 separate single-channel segments of varying length (median 70 timesteps, range 8 to 1,040). This required a redesigned pipeline with two parallel tracks, detailed below.

### Channel Exclusions

CADC0886 (11 segments) and CADC0890 (14 segments) were dropped from analysis due to insufficient data. Seven channels remain with adequate segment counts for meaningful analysis.

---

## Pipeline Design: Dual-Track Approach

Given the segment-based data structure, a dual-track pipeline was designed to match each model class with its natural input paradigm:

**Track A (Segment-Level Classification):** Uses the 18 handcrafted statistical features from the OPSSAT-AD benchmark. For the ablation, 17 space weather summary statistics are appended (mean, max, and standard deviation of each SW index over the segment's time window). This track feeds Isolation Forest and Matrix Profile. It is directly comparable to the 30 algorithms benchmarked in the OPSSAT-AD paper.

**Track B (Time-Series Reconstruction):** Uses the raw univariate waveform per segment, padded or truncated to a fixed length of 128 timesteps. Per-timestep space weather values are attached. This track feeds the LSTM Autoencoder and TCN Autoencoder. The autoencoders learn to reconstruct normal waveforms; high reconstruction error flags anomalies. This approach is novel and was not attempted in the OPSSAT-AD benchmark.

Both tracks run the same ablation: telemetry-only vs. telemetry + space weather features.

### Pipeline Outputs

| Output | Description | Ready |
|---|---|---|
| track_a_telem_only.csv | 2,098 segments, 18 features + metadata | Yes |
| track_a_telem_sw.csv | 2,098 segments, 18 features + 17 SW stats + metadata | Yes |
| track_b_tensors.npz | Tensors: X_telem [2098, 128, 1], X_telem_sw [2098, 128, 7] + masks | Yes |
| 4 EDA figures | Correlation heatmap, anomaly rates, segment lengths, SW conditions | Yes |

---

## Key EDA Figures

Four publication-quality figures have been generated and committed to the `figures/` directory in the repository:

1. **Telemetry Channel vs. Space Weather Correlation Heatmap** -- Shows Spearman correlations across all segments, normal-only, and anomaly-only. The key finding (photodiodes > magnetometers) is immediately visible.

2. **Anomaly Rate per Telemetry Channel** -- Shows anomaly prevalence ranges from 0% (CADC0884) to 35.6% (CADC0874/Mag-Z). CADC0884's zero anomaly rate means it contributes only to the normal class and is useful as a reconstruction baseline for autoencoders.

3. **Segment Length Distribution** -- Shows the heavy right skew in segment lengths. 59.8% of segments are shorter than the seq_len = 128 cutoff. Anomaly segments tend to be longer (median 184.5 timesteps) than typical normal segments (median 56 for the a3 category). A mask array is used during training so models only compute error on real timesteps, not padding.

4. **Space Weather Conditions During OPSSAT Period** -- Four-panel time series of Kp, Dst, solar wind speed, and F10.7 during Jan-Jun 2022. Confirms moderate storm activity with clear variability patterns.

---

## Updated Timeline

| Milestone | Target Date | Status |
|---|---|---|
| Environment + data download (Stage 1) | March 4 | DONE |
| Data pipeline + EDA (Stage 2) | March 5 | DONE |
| Baselines: IF + MP, both conditions (Stage 3) | March 6-8 | NEXT |
| LSTM-AE, both conditions | March 8-10 | Upcoming |
| TCN-AE, both conditions | March 10-12 | Upcoming |
| Publication-quality figures | March 12-14 | Upcoming |
| **Abstract submitted** | **March 19** | Upcoming |
| Poster design + iteration | March 19-28 | Upcoming |
| **Poster submitted** | **March 29** | Upcoming |
| **In-person judging** | **April 2** | Upcoming |

We are currently ahead of the original schedule (pipeline was planned through March 9, EDA through March 10).

---

## Next Steps

Running Isolation Forest on Track A, both conditions. This produces the first numbers for the ablation table and validates whether the experimental setup is working before investing time in the deep learning models.

I welcome any feedback on the pipeline design, the dual-track approach, or the EDA findings.

Best regards,
Yousef K. Awartani
