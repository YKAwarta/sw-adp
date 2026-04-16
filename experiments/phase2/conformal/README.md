# Phase 2 — Conformal Prediction Calibration

Calibration runs for split / Mondrian conformal prediction on top of the Phase 1 anomaly scorers (decision D-02, deferred from Phase 1).

Tracks empirical coverage against the nominal level (target ≥ 90%) for each model × fusion condition. Outputs include calibration curves, nonconformity score distributions, and prediction-set sizes.

See `src/calibration/` for the implementation.
