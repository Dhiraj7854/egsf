"""
egsf/uncertainty/d4_uncertainty.py
───────────────────────────────────
D4: Uncertainty Estimator (Model 12 / Baseline D4) — EGSF v8.0, Step 1.13.

Computes gating & prediction uncertainty via K_replicates stochastic forward draws / MC-dropout:
1. Variance over replicates: var_m(x) = Var_k( g_{m, k}(x) )
2. One-sided 95% LCB gate: g_m_LCB(x) = max(0.0, E[g_m(x)] - z_lcb * std[g_m(x)] )
where z_lcb = 1.64 for 95% confidence.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make project root importable
sys.path.insert(0, str(Path(__file__).parents[2]))
from egsf.utils.reproducibility import seed_everything


def compute_d4_uncertainty(
    model: nn.Module,
    xs: List[torch.Tensor | np.ndarray],
    K_replicates: int = 5,
    z_lcb: float = 1.64,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute D4 Uncertainty statistics & 95% LCB gates.

    Parameters
    ----------
    model : trained PyTorch model with dropout/stochasticity
    xs : list of modality feature tensors
    K_replicates : int (default 5 replicates)
    z_lcb : float (1.64 for 95% one-sided LCB)

    Returns
    -------
    mean_gates : np.ndarray (N, M)
    std_gates  : np.ndarray (N, M)
    lcb_gates  : np.ndarray (N, M)
    """
    seed_everything(seed)
    model.train()  # Enable dropout/stochastic forward passes

    xs_t = [torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x for x in xs]
    N = xs_t[0].shape[0]

    replicate_gates = []

    with torch.no_grad():
        for k in range(K_replicates):
            out = model(xs_t)
            if isinstance(out, tuple):
                gates = out[1]
            else:
                gates = F.softmax(out, dim=-1)

            replicate_gates.append(gates.numpy())

    rep_array = np.array(replicate_gates)  # (K_replicates, N, M)
    mean_gates = np.mean(rep_array, axis=0)
    std_gates  = np.std(rep_array, axis=0)

    # 95% LCB Gate: max(0.0, mean - z_lcb * std)
    lcb_gates = np.maximum(0.0, mean_gates - z_lcb * std_gates)

    return mean_gates, std_gates, lcb_gates


def _self_test() -> int:
    """Self-test D4 Uncertainty Estimator on synthetic data."""
    from egsf.data.jdb_s import generate_jdbs
    from egsf.models.egsf import GEGSF, train_egsf_mod

    print("=" * 62)
    print("D4 UNCERTAINTY ESTIMATOR (MODEL 12) SELF-TEST")
    print("=" * 62)

    failures = 0

    def _check(cond: bool, msg: str) -> None:
        nonlocal failures
        tag = "[PASS]" if cond else "[FAIL]"
        print(f"  {tag} {msg}")
        if not cond:
            failures += 1

    ds = generate_jdbs("R1", rho_corr=0.8, seed=0)
    model, _ = train_egsf_mod(
        [ds["train"]["X1"], ds["train"]["X2"]], ds["train"]["y"],
        [ds["val_id"]["X1"], ds["val_id"]["X2"]], ds["val_id"]["y"],
        in_dims=[8, 8], seed=0
    )

    mean_g, std_g, lcb_g = compute_d4_uncertainty(
        model,
        [ds["val_id"]["X1"], ds["val_id"]["X2"]],
        K_replicates=5,
        z_lcb=1.64,
        seed=0
    )

    _check(mean_g.shape == (200, 2), f"Mean gates shape (200, 2): got {mean_g.shape}")
    _check(std_g.shape == (200, 2), f"Std gates shape (200, 2): got {std_g.shape}")
    _check(lcb_g.shape == (200, 2), f"LCB gates shape (200, 2): got {lcb_g.shape}")
    _check((lcb_g <= mean_g + 1e-6).all(), "LCB gates strictly <= mean gates")

    print("=" * 62)
    if failures == 0:
        print("ALL D4 UNCERTAINTY SELF-TESTS PASSED")
    else:
        print(f"FAILED: {failures} test(s) failed")
    print("=" * 62)

    return failures


if __name__ == "__main__":
    n_fail = _self_test()
    sys.exit(n_fail)
