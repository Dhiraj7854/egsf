# EGSF v8.0 — Explanation-Guided Selective Fusion

## 1. Research Question
**Can explanation-guided diagnosis identify harmful multimodal shortcut reliance and selectively correct it while controlling false corrections and preserving in-distribution behavior?**

## 2. Core EGSF Mechanism
The implemented pipeline proceeds sequentially:
1. **Reliance:** Estimate individual modality reliance via D0.
2. **Invariant information budget:** Compute the capacity of the non-shortcut modality (D2).
3. **Excess reliance:** Compare reliance against the budget (D1).
4. **Trust/spuriousness/uncertainty:** Translate excess reliance into uncertainty/distrust (D4).
5. **Calibrated diagnosis:** Use Conformal Risk Control (E4 CRC) to guarantee bounded false correction rate.
6. **Selective correction:** Only intervene (fuse or drop modality) when the gate explicitly triggers (D3/C1/C3).

## 3. Model/Gate Hierarchy
The project successfully implemented the core diagnostic and selective correction ladder:
* **Gate 0** — BF Shortcut Sensitivity (Establishes the problem space)
* **Gate 1** — D1 Excess Reliance Diagnosis / C1 (Demonstrates diagnostic capacity)
* **Gate 2** — C1 EGSF-Core Certification (Proves statistical safety via CRC)
* **Gate 3** — C3 Real Data Transfer (Transfers diagnosis to semi-synthetic benchmark)

## 4. Gate 0 Results
**Purpose:** Establish baseline modality dominance and shortcut sensitivity.
* **BF ID behavior:** test_id accuracy = 0.896 (R2, ρ=0.9).
* **BF conflict behavior:** test_conflict accuracy collapses to 0.3040.
* **Shortcut-collapse magnitude:** Drop of 59.2 percentage points.
* **Status:** PASS.

## 5. Gate 1 Results
**Purpose:** Demonstrate that EGSF ER diagnosis (D1) correctly identifies harmful reliance and improves over standard baselines.
* **Regime 2 (R2):** C1 EGSF-Core beats Base Fusion (BF) by +0.2480 accuracy (0.5520 vs 0.3040).
* **Regime 5 (R5):** C1 EGSF-Core beats Base Fusion by +0.3207 accuracy (0.7567 vs 0.4360).
* **Regressions:** No regression on R1 or R3 (Accuracy remains 1.0000).
* **Status:** PASS.

## 6. Gate 2 Results
**Purpose:** Prove statistical safety using Conformal Risk Control (CRC).
* **CRC Formulation:** E4 CRC applied to bound empirical False Correction Rate (FCR).
* **Alpha:** α = 0.10.
* **Calibration Split:** Independent calibration subset.
* **FCR Results:** Overall FCR = 0.0024. Regime bounds strictly satisfied (R1=0.0000, R2=0.0018, R3=0.0000, R4=0.0021, R5=0.0084).
* **Conflict Improvement:** Confirmed positive improvement over BF.
* **ID Behavior:** Mean ID regression strictly non-negative.
* **Seeds:** [0, 1, 2, 3, 4] (1000 CRC splits per seed).
* **Safety Interpretation:** The certification demonstrates controlled false-correction risk under the specified protocol. *Note: CRC bounds the risk of false intervention; it does not guarantee high absolute conflict recovery.*
* **Status:** PASS.

## 7. Gate 3 Results
**Purpose:** Evaluate EGSF C3 transfer to a more complex, inferred-anchor domain using the JDB-R benchmark.
* **JDB-R:** Semi-synthetic multimodal benchmark.
* **Modality X1:** Invariant/audio-like modality (K=4 prototypes, σ=5.0).
* **Modality X2:** Shortcut/image-like modality (pure color, ρ=0.95 with label in train, randomized in conflict).
* **Splits:** train (10k), val_id (2k), cal_crc (2k), conflict_dev (2k), test_id (2k), test_conflict (2k).
* **Seeds:** [0, 1, 2, 3, 4].

**Key Results:**
* **Mean A1 (Oracle) AUROC:** 0.8480
* **Mean A2 (Inferred) AUROC:** 0.8279 (A2-A1 diff = -0.0201)
* **BF conflict accuracy:** 0.3062
* **BAL-Q conflict accuracy:** 0.2952
* **C3 conflict accuracy:** 0.3116 (Improvement over BF = +0.54pp)
* **ID regression:** -0.06pp (Strictly within ≤ 5pp allowed limit)
* **Negative-control AUROC:** ≈ 0.499 (Degrades properly).
* **Status:** PASS.

*Note: The A2 evaluation used disagreement with the true label (y) for diagnostic evaluation only. The certified C3 decision path did not use the test label (y).*

## 8. Leakage Audits
The entire final inference path was audited for leakage. Verified facts:
* No oracle cue labels in A2/C3 decision path.
* No environment labels used.
* No cue-broken labels used.
* No true reliance labels used.
* No test-time fitting performed.
* No test threshold tuning performed.
* Independent calibration used strictly for threshold discovery.
* Gate 1–3 frozen-state integrity strictly preserved.

## 9. Reproducibility
The following commands were used to generate the certified results:
* **Gate 1 & 2 Execution:** `python egsf/experiments/run_gates.py` (and data generation via `python egsf/data/jdb_s.py` / `prepare_gate2_data.py`).
* **Gate 3 Execution:** `python egsf/experiments/run_gate3.py`
* **Python Environment:** See `requirements.txt`.
* **Seeds:** [0, 1, 2, 3, 4]
* **Result Locations:** `results/gate_evaluation.json`, `results/gate2_results.json`, `results/gate3_results.md`.

## 10. Scientific Interpretation
**Supported Claim:**
EGSF demonstrates that explanation-guided diagnosis can identify and selectively address harmful modality reliance in controlled and semi-synthetic multimodal settings, while maintaining low false-correction rates and near-zero ID regression under the certified protocols.

**Rejected Claims (What this does NOT claim):**
* EGSF is not a complete solution to multimodal shortcut learning.
* It does not guarantee universal robustness.
* It does not assert real-world clinical validity.
* It does not demonstrate strong conflict recovery (observed conflict improvement in Gate 3 is only +0.54pp).
* It does not claim deployment readiness.
* It does not claim superiority across unspecified datasets.

## 11. Limitations
* **Synthetic Origin:** JDB-S is entirely synthetic. JDB-R is semi-synthetic/simulated, not a fully natural real-world dataset.
* **Modest Conflict Recovery:** The actual conflict improvement observed in Gate 3 is very small (+0.54pp), indicating that diagnosis is easier than structural recovery.
* **Shortcut Nature:** The benchmark forces specific shortcut types and does not establish performance on naturally occurring multimodal shortcuts in the wild.
* **No Deployment Certification:** No C2/C4 deployment certification exists within the core gates.
* **No Real-World Study:** No real-world deployment study exists.
* **Further Validation:** Substantial further validation is required before practical deployment claims can be made.

## 12. Optional Extensions
Models 15–21 (including C2 EGSF-PID and C4 EGSF-Deploy) were intentionally not included in the certified core because the repository does not define authoritative scientific criteria for them. Avoiding invented, unpreregistered criteria preserves scientific integrity and keeps the certified claim strictly bounded to the rigorously defined Gates 0–3.

## 13. Final Certification Table

| Gate | Purpose | Result | Status |
|------|---------|--------|--------|
| **Gate 0** | Shortcut sensitivity | BF ID=0.896 → Conf=0.3040 (-59.2pp) | **PASS** |
| **Gate 1** | Excess reliance diagnosis | C1 > BF by +0.2480 (R2) & +0.3207 (R5) | **PASS** |
| **Gate 2** | Selective correction + CRC | FCR ≤ 0.10 strictly bounded over 1000 splits | **PASS** |
| **Gate 3** | Semi-synthetic transfer | A2≈A1 (-0.0201), C3 > BF (+0.54pp) on conflict | **PASS** |
