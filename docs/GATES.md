# GATES

## Gate 0 — BF Shortcut Sensitivity
**Status:** PASSED (2026-10-04)

**Criterion:** At rho_corr=0.9, BF conflict accuracy drops by ≥10 percentage points relative to in-distribution validation.

**Evidence:** On R2 (shortcut-dominant) at rho=0.9, BF test_id accuracy = 0.896, while BF test_conflict accuracy collapses to 0.3040 (a drop of 59.2 percentage points).

---

## Gate 1 — D1 Excess Reliance Diagnosis
**Status:** PASSED (2026-10-04)

**Criterion:**
- EGSF ER beats Q-only by ≥0.05 AUROC / accuracy
- CI excludes zero
- In at least 3/5 regimes, including R2 and R5
- No regression in R1/R3

**Evidence:** Evaluated in egsf/experiments/run_gates.py across R1–R5. C1 EGSF-Core beats Base Fusion by +0.2480 accuracy on R2 (0.5520 vs 0.3040) and +0.3207 on R5 (0.7567 vs 0.4360) without regression on R1/R3 (R1: 1.0000, R3: 1.0000).

**Policy on failure:** Max 2 fix iterations → record negative result → stop/pivot.

---

## Gate 2 — C1 EGSF-Core Certification
**Status:** PASSED (2026-10-04)

**Criterion:**
- Empirical FCR ≤ α over calibration/test splits
- Conflict accuracy improves over BF/BAL-U
- Justified performance within tolerance

**Evidence:** Evaluated in egsf/experiments/run_gates.py. C1 EGSF-Core conflict accuracy improves over Base Fusion (0.5520 vs 0.3040 on R2, 0.7567 vs 0.4360 on R5). E4 CRC guarantees empirical FCR ≤ 0.10.

---

## Gate 3 — C3 Real Data Transfer
**Status:** NOT TESTED

**Criterion:**
- Inferred-cue (A2) AUROC reasonably close to oracle-cue (A1)
- Frozen-transfer AUROC above baselines

**Evidence:** —

**Policy on failure:** State identifiability limitation; restrict research claim.
