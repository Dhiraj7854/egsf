"""D1 Excess Reliance ER (Step 6, Gate 1). Owner: B. ER_m = [rho_m - B_m]+.

Scores compared: A-only (rho), Q-only (quality replaces budget), non-invariant (train env only), full ER.
AUROC/AUPRC across R1-R5 x rho_corr x 5 paired seeds, paired-bootstrap CI.
Gate 1: ER beats Q-only by >=0.05 AUROC in >=3/5 regimes incl R2,R5, no regression R1,R3.
"""
def excess(rho: float, B: float) -> float:
    return max(0.0, rho - B)
