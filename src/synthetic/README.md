# src/synthetic — Semi-Synthetic Anomaly Injection (Phase 2)

Implementation of the semi-synthetic dataset construction described in Phase 2 decisions D-17 and D-18. Starts from real OPSSAT-AD waveforms (preserving natural signal noise) and injects anomalies with known provenance:

- **SW-correlated anomalies** — perturbation magnitude scaled by a space weather index (Kp, Dst, F10.7). Establishes ground-truth cases where conditional fusion should have a discriminative advantage.
- **SW-independent anomalies** — random perturbations uncorrelated with any SW index. Control cases that both architectures should catch.

Experiment outputs land under `experiments/phase2/synthetic/`. The injection rate mirrors OPSSAT-AD's ~20% natural anomaly prevalence unless sensitivity-swept (see open question Q-12).
