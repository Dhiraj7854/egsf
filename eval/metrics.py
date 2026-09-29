"""Metrics: AUROC, AUPRC, ECE, Brier, FCR, risk-coverage, BVR. Owner: C."""
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
def auroc(y, s): return float(roc_auc_score(y, s))
def auprc(y, s): return float(average_precision_score(y, s))
def ece(y, p, bins: int = 15) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins+1); e = 0.0
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i+1])
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)
def fcr(flagged_justified: np.ndarray) -> float: return float(np.mean(flagged_justified))
