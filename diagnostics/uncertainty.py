"""D4 Uncertainty (Step 9). Owner: B.

K=5 replicates varying reliance method, budget bootstrap, cal seed.
U_m=clip(std/(mean+delta),0,1); ER_LCB=mean-z*std, z=1.64.
Check selective-diagnosis curve: low-U AUROC > all-case AUROC. Cache; refresh every Krefresh.
"""
def uncertainty(vals: list, z: float = 1.64):
    import numpy as np
    m, s = float(np.mean(vals)), float(np.std(vals))
    return min(1.0, max(0.0, s / (m + 1e-8))), m - z * s
