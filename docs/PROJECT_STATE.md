# PROJECT STATE

## Current Phase
Phase 1 — Synthetic Benchmark & Unimodal Baseline (Step 1)

## Current Step
1.16 — Execute Master Gate Evaluation Sweep (Models 1–14 for Gate 1 & Gate 2)

## Completed Steps
- [x] 0.1 Initialize repo skeleton
- [x] 0.2 Write config & reproducibility utils
- [x] 0.3 Verify python environment dependencies
- [x] 1.1 Implement and verify JDB-S data generator (egsf/data/jdb_s.py)
- [x] 1.2 Implement and verify U-Mod (Model 1) unimodal baseline (egsf/models/u_mod.py)
- [x] 1.3 Implement and verify BF (Model 2) Base Fusion baseline (egsf/models/bf.py)
- [x] 1.4 Implement and verify BF-Oracle (Model 3) oracle baseline (egsf/models/bf_oracle.py)
- [x] 1.5 Implement and verify BAL-U (Model 4) unimodal budget-aware baseline (egsf/models/bal_u.py)
- [x] 1.6 Implement and verify BAL-G (Model 5) OGM-GE group-gated baseline (egsf/models/bal_g.py)
- [x] 1.7 Implement and verify BAL-Q (Model 6) quantile-gated baseline (egsf/models/bal_q.py)
- [x] 1.8 Implement and verify BAL-A (Model 7) anchor-gated baseline (egsf/models/bal_a.py)
- [x] 1.9 Implement and verify D0 Reliance Estimator (Model 8) (egsf/reliance/d0_reliance.py)
- [x] 1.10 Implement and verify D1 Explanation Generator (Model 9) (egsf/explanations/d1_explanation.py)
- [x] 1.11 Implement and verify D2 Budget Module (Model 10) (egsf/budget/d2_budget.py)
- [x] 1.12 Implement and verify D3 Selective Gate (Model 11) (egsf/gate/d3_gate.py)
- [x] 1.13 Implement and verify D4 Uncertainty Estimator (Model 12) (egsf/uncertainty/d4_uncertainty.py)
- [x] 1.14 Implement and verify E4 Conformal Risk Control (Model 13) (egsf/calibration/e4_crc.py)
- [x] 1.15 Implement and verify C1 EGSF-Core (Model 14) (egsf/models/c1_egsf_core.py)

## Current Experiment
Step 1.15 Self-test (C1 EGSF-Core)

## Current Model
Model 14 (C1 EGSF-Core)

## Current Gate
Gate 1 & Gate 2 (Evaluation Pending)

## Known Failures
None

## Next Required Action
1.16 — Execute Master Gate Evaluation Sweep in egsf/experiments/run_gates.py --dev

## Repository State
- Fresh init, skeleton directories created
- No experiments run

## Important Configuration
- Python: 3.11.9
- PyTorch: 2.14.0+cpu
- NumPy: 2.2.6
- scikit-learn: 1.7.1
- SciPy: 1.17.1

## Seeds
Not yet assigned (will use: 0, 1, 2, 3, 4 for 5-seed runs)

## Datasets
- JDB-S (synthetic): NOT YET BUILT
- JDB-R (real): NOT YET BUILT

## Unresolved Questions
- GPU availability (currently CPU-only PyTorch)
- W&B / MLflow preference
