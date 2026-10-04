# Gate 3 Final Certification Report — EGSF v8.0

## Dataset: JDB-R

- **Type:** Simulated semi-synthetic multimodal benchmark
- **Modality 0 (X1):** Audio / Spoken-Digit (invariant; K=4 class prototypes, σ_noise=5.0)
- **Modality 1 (X2):** Image / Colored-MNIST (shortcut; pure color-only, no invariant shape component)
- **Shortcut mechanism:** Color cue `c` correlates with class `y` at ρ=0.95 in training. In conflict, `c` is randomized (independent of `y`)
- **Splits:** train (10k), val_id (2k), cal_crc (2k), conflict_dev (2k), test_id (2k), test_conflict (2k)
- **Environments:** env_idx=0 (ID, cue_broken=False), env_idx=1 (Conflict, cue_broken=True)
- **Seeds:** [0, 1, 2, 3, 4]

---

## A1 — Oracle Diagnosis

**Method:** Uses oracle cue label `c` (planted). Anchor = `c != y` (shortcut mismatch).

| Seed | A1 AUROC | A1 AUPRC |
|------|----------|----------|
| 0    | 0.8490   | 0.8269   |
| 1    | 0.8468   | 0.8246   |
| 2    | 0.8455   | 0.8231   |
| 3    | 0.8560   | 0.8333   |
| 4    | 0.8430   | 0.8206   |
| **Mean** | **0.8480** | **0.8260** |

---

## A2 — Inferred Diagnosis

**Method:** Trains a biased unimodal predictor on `X2` (image modality) from training data only. At evaluation time, computes `pred_X2(x) != y` as the inferred shortcut-mismatch anchor. No oracle `c`, `env_idx`, or `cue_broken` labels used.

| Seed | A2 AUROC | A2 AUPRC | A2 − A1 |
|------|----------|----------|---------|
| 0    | 0.7510   | 0.6881   | −0.0980 |
| 1    | 0.8468   | 0.8246   | 0.0000  |
| 2    | 0.8430   | 0.8206   | −0.0025 |
| 3    | 0.8560   | 0.8333   | 0.0000  |
| 4    | 0.8430   | 0.8206   | 0.0000  |
| **Mean** | **0.8279** | **0.7975** | **−0.0201** |

Preregistered tolerance: A2 ≥ A1 − 0.10 → **PASS** (margin: +0.0799)

> Note: Seed 0 is an outlier (−0.098). Seeds 1–4 show A2 ≈ A1. Mean gap is −0.020 — well within tolerance.

---

## Baselines & C3

| Seed | BF Conf | BAL-Q Conf | C3 Conf | BF ID | C3 ID | Regress |
|------|---------|------------|---------|-------|-------|---------|
| 0    | 0.2645  | 0.2615     | 0.2690  | 0.9445 | 0.9445 | 0.00pp |
| 1    | 0.3130  | 0.2980     | 0.3135  | 0.9510 | 0.9520 | 0.00pp |
| 2    | 0.2855  | 0.2745     | 0.2925  | 0.9570 | 0.9565 | 0.05pp |
| 3    | 0.3360  | 0.3290     | 0.3495  | 0.9545 | 0.9565 | −0.20pp |
| 4    | 0.3320  | 0.3130     | 0.3335  | 0.9530 | 0.9535 | −0.05pp |
| **Mean** | **0.3062** | **0.2952** | **0.3116** | **0.9520** | **0.9526** | **−0.06pp** |

- **BF shortcut collapse confirmed:** ID=0.9520 → Conflict=0.3062 (−64.6pp)
- **D0 X2 reliance (ID): 0.951** — BF almost entirely reliant on shortcut modality
- **Conflict Improvement over BF:** +0.54pp → **PASS**
- **C3 > BAL-Q on conflict** in 4/5 seeds (mean: +1.64pp over BAL-Q)
- **ID Regression:** −0.06pp → **PASS** (C3 slightly helps ID)

---

## Negative Controls

| Seed | Shuffled A2 AUROC |
|------|------------------|
| 0    | 0.5100           |
| 1    | 0.4977           |
| 2    | 0.4950           |
| 3    | 0.4995           |
| 4    | 0.4965           |
| **Mean** | **≈ 0.499** |

Shuffled evidence degrades to chance in all seeds. No residual signal survives shuffling. ✓

---

## Leakage Audit

| Check | Result |
|-------|--------|
| A2 uses `oracle_meta["c"]` | ❌ NOT used |
| A2 uses `oracle_meta["env_idx"]` | ❌ NOT used |
| A2 uses `oracle_meta["cue_broken"]` | ❌ NOT used |
| A2 uses true modality-reliance labels | ❌ NOT used |
| C3 test-time decisions use test `y` | ❌ NOT used |
| Budget policy fitted to test data | ❌ NOT done |
| Threshold tuned on test set | ❌ NOT done |
| `pred_X2 != y` used in C3 decision path | ❌ Only in evaluation AUROC |

**Leakage audit: CLEAN ✓**

---

## Reproducibility

- Seeds: [0, 1, 2, 3, 4]
- JDB-R: `egsf/data/jdb_r.py` (K=4, d=8, σ_X1=5.0, σ_X2=0.3, ρ=0.95)
- Runner: `egsf/experiments/run_gate3.py`
- Budget policy: `[1.0, 0.05] * 1.1` (kappa=0.1, fixed architectural prior)
- Results: `results/gate3_audit.md`, `results/gate3_plan.md`
- Gate 1 and Gate 2 source files unchanged

---

## Final Criteria Check

| Criterion | Value | Threshold | Status |
|-----------|-------|-----------|--------|
| A2 AUROC ≥ A1 − 0.10 | 0.8279 ≥ 0.7480 | margin: +0.0799 | **PASS** |
| Conflict Benefit > 0 | +0.54pp | > 0 | **PASS** |
| ID Regression ≤ 5pp | −0.06pp | ≤ 5pp | **PASS** |
| Negative controls degrade | ~0.50 | < real AUROC | **PASS** |
| No oracle leakage | Confirmed | Zero leakage | **PASS** |

---

## GATE 3 FINAL CERTIFICATION: PASS
