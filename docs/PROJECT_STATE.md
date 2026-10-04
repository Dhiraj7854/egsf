# PROJECT STATE

## Current Phase
Phase 1 — Synthetic Benchmark & Unimodal Baseline (Step 1)

## Current Step
1.4 — Reconcile Model Ladder & Validate BF-Oracle (Model 3)

## Completed Steps
- [x] 0.1 Initialize repo skeleton
- [x] 0.2 Write config & reproducibility utils
- [x] 0.3 Verify python environment dependencies
- [x] 1.1 Implement and verify JDB-S data generator (egsf/data/jdb_s.py)
- [x] 1.2 Implement and verify U-Mod (Model 1) unimodal baseline (egsf/models/u_mod.py)
- [x] 1.3 Implement and verify BF (Model 2) Base Fusion baseline (egsf/models/bf.py)
- [x] 1.4 Implement and verify BF-Oracle (Model 3) oracle baseline (egsf/models/bf_oracle.py)

## Current Experiment
Step 1.4 Self-test (BF-Oracle)

## Current Model
Model 3 (BF-Oracle)

## Current Gate
Gate 1 (Pre-requisite: JDB-S, U-Mod, BF, and BF-Oracle validated)

## Known Failures
None

## Next Required Action
1.5 — Implement BAL-U (Model 4: Unimodal Budget-Aware Baseline) in egsf/models/bal_u.py

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
