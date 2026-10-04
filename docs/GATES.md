# GATES

## Gate 0 — BF Shortcut Sensitivity
**Status:** NOT TESTED

**Criterion:** At rho_corr=0.9, BF conflict accuracy drops by ≥10 percentage
points relative to appropriate baseline.

**Evidence:** —

---

## Gate 1 — D1 Excess Reliance Diagnosis
**Status:** NOT TESTED

**Criterion:**
- EGSF ER beats Q-only by ≥0.05 AUROC
- CI excludes zero
- In at least 3/5 regimes, including R2 and R5
- No regression in R1/R3

**Evidence:** —

**Policy on failure:** Max 2 fix iterations → record negative result → stop/pivot.

---

## Gate 2 — C1 EGSF-Core Certification
**Status:** NOT TESTED

**Criterion:**
- Empirical FCR ≤ α over ≥1000 random cal/test splits
- Conflict accuracy improves over BF/BAL-U
- Justified performance within tolerance

**Evidence:** —

---

## Gate 3 — C3 Real Data Transfer
**Status:** NOT TESTED

**Criterion:**
- Inferred-cue (A2) AUROC reasonably close to oracle-cue (A1)
- Frozen-transfer AUROC above baselines

**Evidence:** —

**Policy on failure:** State identifiability limitation; restrict research claim.
