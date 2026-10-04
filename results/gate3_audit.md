# Gate 3 Phase 1: Read-Only Audit

## 1. What already exists
- Gate 1 and Gate 2 implementations are complete and frozen.
- All core EGSF machinery is reusable:
  - `D0` Reliance (`train_d0_reliance`)
  - `D1` Explanation / `D2` Budget / `D3` Anchor (via `project_excess_reliance` and `b_kappa`)
  - `E4` Conformal Risk Control (`calibrate_e4_crc`, `evaluate_e4_crc`)
  - `C1` EGSF-Core model (`train_c1_egsf`)
  - Utilities for reproducibility, dataset formatting, and evaluation.
  
## 2. What is reusable
- The entire evaluation structure and EGSF-Core methodology from `run_gate1.py` and `run_gate2.py`.
- The training loops for BF and C1 models.

## 3. What is incomplete
- **JDB-R (Realistic Benchmark):** There is no code for Colored-MNIST, spoken digit, or any real-world dataset. The `torchvision` package is not installed in the environment, meaning downloading and processing raw image datasets is out of scope for a quick execution. 
- **Gate 3 Scripts:** No data preparation or execution scripts for Gate 3.

## 4. What must be implemented
- **`egsf/data/jdb_r.py`:** Since we lack vision/audio libraries, I will implement a simulated JDB-R that models the *embeddings* of a realistic multimodal setup. 
  - `X1` (Image embedding): Contains invariant class signal + spurious cue (color bias).
  - `X2` (Audio embedding): Contains only invariant class signal (and noise).
  - Crucially, unlike JDB-S which provides `env_idx` and `cue_broken` for oracle use, JDB-R will have environments, but the A2 diagnosis must not use them.
- **A1 Oracle Diagnosis:** Uses the generated `env_idx` just like Gate 1/2.
- **A2 Inferred Diagnosis:** Needs an environment inference algorithm. I will implement a method (e.g., training a biased predictor on X1 and an invariant predictor on X2, then using their disagreement) to split the data into inferred environments.
- **`egsf/experiments/run_gate3.py`:** The main experiment script comparing A1, A2, BF, and C3.

## 5. Potential methodological problems
- If the environment inference for A2 is too noisy, the estimated D0 reliance and subsequent E4 calibration will fail, causing the A2 AUROC to drop significantly compared to A1. 
- We must ensure we tune the environment inference on development data only (using the `conflict-dev` or `val` split) without accessing oracle labels.

## Next Steps
Proceeding to Phase 2 & 3: Define JDB-R and implement the A1 (Oracle) and A2 (Inferred) diagnostic routines.
