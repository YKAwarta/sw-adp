# src/calibration — Conformal Prediction (Phase 2)

Implementation of conformal prediction calibration on top of the Phase 1 anomaly scorers (decision D-02, deferred from Phase 1). Provides distribution-free, finite-sample coverage guarantees at a nominal level (target ≥ 90%).

Scope:
- Split conformal prediction on anomaly scores from each Phase 1 model (IF, MP+IF, LSTM-AE, TCN-AE) × fusion condition
- Mondrian variant stratified by telemetry channel (7 channels after dropping CADC0886/0890)
- Empirical-coverage reporting against target level on a held-out calibration slice

Calibration outputs land under `experiments/phase2/conformal/`.
