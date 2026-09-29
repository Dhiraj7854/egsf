"""Paired bootstrap CIs (5 seeds). Owner: C. Used for every ladder comparison."""
import numpy as np
def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed); d = np.asarray(b) - np.asarray(a); n = len(d)
    boots = [rng.choice(d, n, replace=True).mean() for _ in range(n_boot)]
    return float(d.mean()), (float(np.quantile(boots, .025)), float(np.quantile(boots, .975)))
