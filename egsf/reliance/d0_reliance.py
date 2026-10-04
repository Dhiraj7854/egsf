"""
egsf/reliance/d0_reliance.py
─────────────────────────────
D0: Reliance Estimator (Model 8 / Reliance Baseline D0) — EGSF v8.0, Step 1.9.

Estimates instance-level modality reliance R_m(x) via counterfactual resampling / perturbation:
R_m(x) = E_{x_m' ~ P(X_m)} [ D_{KL}( P(y | x_1, ..., x_m, ..., x_M) || P(y | x_1, ..., x_m', ..., x_M) ) ]

Measures the output distribution shift when modality m is perturbed/resampled S times.
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


def compute_d0_reliance(
    model: nn.Module,
    xs: List[torch.Tensor | np.ndarray],
    n_resample: int = 8,
    delta_floor: float = 1e-6,
    is_multimodal: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """
    Compute D0 Reliance scores R_m(x) for each sample and modality.

    Parameters
    ----------
    model : trained PyTorch model
    xs : list of modality feature tensors [(N, d_1), ..., (N, d_M)]
    n_resample : int (S=8 resample draws)
    delta_floor : float (numerical stability)

    Returns
    -------
    reliances : np.ndarray of shape (N, M)
    """
    seed_everything(seed)
    model.eval()

    xs_t = [torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x for x in xs]
    N = xs_t[0].shape[0]
    M = len(xs_t)

    with torch.no_grad():
        # Clean baseline prediction probabilities
        if is_multimodal:
            out = model(xs_t)
            logits_clean = out[0] if isinstance(out, tuple) else out
        else:
            logits_clean = model(xs_t[0])

        probs_clean = F.softmax(logits_clean, dim=-1)  # (N, K)

        reliances = np.zeros((N, M), dtype=np.float32)

        for m in range(M):
            kl_divs = []
            mod_data = xs_t[m]

            for s in range(n_resample):
                # Sample random counterfactuals from empirical marginal distribution of modality m
                rand_indices = torch.randint(0, N, size=(N,))
                xs_perturbed = [x.clone() for x in xs_t]
                xs_perturbed[m] = mod_data[rand_indices]

                if is_multimodal:
                    out_p = model(xs_perturbed)
                    logits_p = out_p[0] if isinstance(out_p, tuple) else out_p
                else:
                    logits_p = model(xs_perturbed[0])

                probs_p = F.softmax(logits_p, dim=-1)

                # KL divergence D_KL(P_clean || P_perturbed)
                kl = torch.sum(probs_clean * torch.log((probs_clean + delta_floor) / (probs_p + delta_floor)), dim=-1)
                kl_divs.append(kl.numpy())

            # Mean KL divergence across S resample draws
            reliances[:, m] = np.maximum(0.0, np.mean(kl_divs, axis=0))

        # Normalize reliance scores across modalities per instance
        rel_sum = reliances.sum(axis=-1, keepdims=True) + delta_floor
        reliances_norm = reliances / rel_sum

    return reliances_norm


def _self_test() -> int:
    """Self-test D0 Reliance Estimator on synthetic JDB-S data."""
    from egsf.data.jdb_s import generate_jdbs
    from egsf.models.bf import BaseFusionMLP, train_bf_mod

    print("=" * 62)
    print("D0 RELIANCE ESTIMATOR (MODEL 8) SELF-TEST")
    print("=" * 62)

    failures = 0

    def _check(cond: bool, msg: str) -> None:
        nonlocal failures
        tag = "[PASS]" if cond else "[FAIL]"
        print(f"  {tag} {msg}")
        if not cond:
            failures += 1

    # 1. Evaluate D0 on trained BF model
    ds = generate_jdbs("R1", rho_corr=0.8, seed=0)
    bf_model, _ = train_bf_mod(
        [ds["train"]["X1"], ds["train"]["X2"]], ds["train"]["y"],
        [ds["val_id"]["X1"], ds["val_id"]["X2"]], ds["val_id"]["y"],
        in_dims=[8, 8], seed=0
    )

    rel_scores = compute_d0_reliance(
        bf_model,
        [ds["val_id"]["X1"], ds["val_id"]["X2"]],
        n_resample=8,
        seed=0
    )

    _check(rel_scores.shape == (200, 2), f"Reliance scores shape (200, 2): got {rel_scores.shape}")
    _check(np.allclose(rel_scores.sum(axis=-1), 1.0, atol=1e-4), "Reliance scores per instance sum to 1.0")

    # In R1 (causal dominant), causal reliance R1 should be higher than cue reliance R2
    mean_r1 = float(np.mean(rel_scores[:, 0]))
    mean_r2 = float(np.mean(rel_scores[:, 1]))
    _check(mean_r1 > mean_r2, f"R1 causal reliance R1 ({mean_r1:.4f}) > cue reliance R2 ({mean_r2:.4f})")

    print("=" * 62)
    if failures == 0:
        print("ALL D0 RELIANCE SELF-TESTS PASSED")
    else:
        print(f"FAILED: {failures} test(s) failed")
    print("=" * 62)

    return failures


if __name__ == "__main__":
    n_fail = _self_test()
    sys.exit(n_fail)
