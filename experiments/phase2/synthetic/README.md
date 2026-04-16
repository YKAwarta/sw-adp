# Phase 2 — Semi-Synthetic Anomaly Injection

Results from experiments on the semi-synthetic dataset: real OPSSAT-AD waveforms with programmatically injected anomalies of known provenance (decisions D-17, D-18).

Two anomaly classes evaluated:

- **SW-correlated** — anomaly magnitude scaled by a space weather index (Kp, Dst, F10.7). Conditional fusion should catch these more readily than telemetry-only.
- **SW-independent** — random perturbations with no SW coupling. Both architectures should catch these with similar performance.

Organized by model × fusion condition × anomaly class. JSON result files follow the Phase 1 naming convention (see `../../phase1/stage4_deep_learning/`).
