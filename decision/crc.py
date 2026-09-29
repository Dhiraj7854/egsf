"""CRC certified threshold (Step 11, Gate 2). Owner: C.

lambda* = inf{lambda : (n/(n+1))Rhat_n(lambda)+1/(n+1) <= alpha}, L_i=1[D_i>lambda] on n justified cal-crc.
Needs n >= 1/alpha - 1 minimum. E4: FCR <= alpha over >=1000 splits. Report harm recall too (marginal guarantee only).
"""
import numpy as np
def crc_threshold(scores_justified: np.ndarray, alpha: float = 0.1) -> float:
    n = len(scores_justified)
    for lam in sorted(np.unique(scores_justified), reverse=True):
        rhat = np.mean(scores_justified > lam)
        if (n/(n+1))*rhat + 1/(n+1) <= alpha:
            return float(lam)
    return 1.0
