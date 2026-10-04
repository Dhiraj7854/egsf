# Gate 3 Phase 2 & 3: Minimum Implementation Plan & Initial Results

## Minimum Implementation Status
1. **JDB-R (`egsf/data/jdb_r.py`)**: I have successfully created a minimal viable semi-synthetic JDB-R benchmark. It simulates high-dimensional embeddings for:
   - Modality 1 (Image): Class information + color bias (shortcut).
   - Modality 2 (Audio): Invariant class information + noise.
2. **A1 (Oracle) vs A2 (Inferred) (`egsf/experiments/run_gate3.py`)**: I have implemented the minimal experimental loop.
   - **A1** uses oracle `env_idx` provided in the JDB-R metadata.
   - **A2** infers `env_idx` by training a biased predictor on Modality 1 (Image) and assigning `env_idx = 1` when the predictor fails on the combined train/dev set.

## Initial Smoke Test Results (Dev Experiment)
The experimental loop runs successfully without errors.
- **A1 AUROC:** ~0.19
- **A2 AUROC:** ~0.19
- **Match:** A2 perfectly matches A1 in identifying the conflict environments!

## Bug Audit & Required Fixes
- **AUROC Inversion:** The AUROC is ~0.19, which means the diagnostic score is anti-correlated with the ground truth (random guessing is 0.50). This is because the simple proxy I used for `D0` reliance (max softmax probability) remains high on conflict sets (the biased model confidently predicts the wrong class). I need to refine the `D0` reliance estimation to use conditional mutual information or the true EGSF machinery rather than a simplified proxy.
- **Missing Baselines:** As per the Gate 3 requirements, I need to include `BF` and `BAL-Q` (quality-only) baselines to satisfy **Criterion 2 (A2 AUROC > baselines)**.

## Plan for Next Steps
1. **Refine D0 / JDB-R Integration:** Import and use the actual `D0` reliance machinery from the frozen Gate 1/2 codebase instead of the fast proxy. This will require adapting `jdb_r.py` to seamlessly fit into the `train_d0_reliance` inputs.
2. **Implement Baselines:** Add the `BF` (Base Fusion) and `BAL-Q` baselines to the evaluation script.
3. **Run Dev Experiment:** Ensure the refined setup achieves AUROC > 0.50 and A2 remains competitive with A1.
4. **Full Certification:** Once the dev experiment passes, scale up to the full 5 seeds and report the final metrics (AUROC, AUPRC, Conflict Accuracy, ID Accuracy) and verify all negative controls.
