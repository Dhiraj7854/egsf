"""
egsf/explanations/d1_explanation.py
───────────────────────────────────
D1: Explanation Generator (Model 9 / Baseline D1) — EGSF v8.0, Step 1.10.

Generates feature-level attribution explanations E_m(x) in R^d for each modality m
using Integrated Gradients (or Gradient x Input attribution):
E_m(x) = (x_m - x_{m, baseline}) * integral_{alpha=0}^1 grad_{x_m} f( x_{baseline} + alpha (x - x_{baseline}) ) dalpha

Computes modality attribution magnitudes ||E_m(x)||_1 and normalized modality attributions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Make project root importable
sys.path.insert(0, str(Path(__file__).parents[2]))
from egsf.utils.reproducibility import seed_everything


def compute_d1_explanations(
    model: nn.Module,
    xs: List[torch.Tensor | np.ndarray],
    target_class: Optional[torch.Tensor | np.ndarray] = None,
    n_steps: int = 20,
    is_multimodal: bool = True,
    seed: int = 0,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Compute D1 Integrated Gradients feature attributions for each modality.

    Parameters
    ----------
    model : trained PyTorch model
    xs : list of modality feature tensors [(N, d_1), ..., (N, d_M)]
    target_class : optional target class labels (N,); if None, uses argmax prediction
    n_steps : number of Riemann sum steps for Integrated Gradients (default 20)

    Returns
    -------
    attributions : list of np.ndarray [(N, d_1), ..., (N, d_M)]
    modality_importance : np.ndarray (N, M) normalized ||E_m(x)||_1
    """
    seed_everything(seed)
    model.eval()

    xs_t = [torch.tensor(x, dtype=torch.float32, requires_grad=True) if isinstance(x, np.ndarray) else x.clone().detach().requires_grad_(True) for x in xs]
    N = xs_t[0].shape[0]
    M = len(xs_t)

    with torch.no_grad():
        if is_multimodal:
            out = model(xs_t)
            logits = out[0] if isinstance(out, tuple) else out
        else:
            logits = model(xs_t[0])
        
        if target_class is None:
            targets = logits.argmax(dim=-1)
        else:
            targets = torch.tensor(target_class, dtype=torch.long) if isinstance(target_class, np.ndarray) else target_class

    attributions = [np.zeros_like(x.detach().numpy()) for x in xs_t]

    # Riemann sum for Integrated Gradients against zero baseline
    for step in range(1, n_steps + 1):
        alpha = step / float(n_steps)
        xs_step = [x * alpha for x in xs_t]
        for x in xs_step:
            x.requires_grad_(True)

        if is_multimodal:
            out_step = model(xs_step)
            logits_step = out_step[0] if isinstance(out_step, tuple) else out_step
        else:
            logits_step = model(xs_step[0])

        # Gather target class logits
        target_logits = logits_step[torch.arange(N), targets]
        target_logits.sum().backward()

        for m in range(M):
            if xs_step[m].grad is not None:
                attributions[m] += xs_step[m].grad.detach().numpy() / float(n_steps)

    # Gradient x Input
    for m in range(M):
        attributions[m] = attributions[m] * xs_t[m].detach().numpy()

    # Compute ||E_m(x)||_1
    magnitudes = np.array([np.sum(np.abs(attr), axis=-1) for attr in attributions]).T  # (N, M)
    mag_sum = magnitudes.sum(axis=-1, keepdims=True) + 1e-9
    modality_importance = magnitudes / mag_sum

    return attributions, modality_importance


def _self_test() -> int:
    """Self-test D1 Explanation Generator on synthetic JDB-S data."""
    from egsf.data.jdb_s import generate_jdbs
    from egsf.models.bf import train_bf_mod

    print("=" * 62)
    print("D1 EXPLANATION GENERATOR (MODEL 9) SELF-TEST")
    print("=" * 62)

    failures = 0

    def _check(cond: bool, msg: str) -> None:
        nonlocal failures
        tag = "[PASS]" if cond else "[FAIL]"
        print(f"  {tag} {msg}")
        if not cond:
            failures += 1

    ds = generate_jdbs("R1", rho_corr=0.8, seed=0)
    bf_model, _ = train_bf_mod(
        [ds["train"]["X1"], ds["train"]["X2"]], ds["train"]["y"],
        [ds["val_id"]["X1"], ds["val_id"]["X2"]], ds["val_id"]["y"],
        in_dims=[8, 8], seed=0
    )

    attrs, mod_imp = compute_d1_explanations(
        bf_model,
        [ds["val_id"]["X1"], ds["val_id"]["X2"]],
        target_class=ds["val_id"]["y"],
        n_steps=10,
        seed=0
    )

    _check(len(attrs) == 2 and attrs[0].shape == (200, 8), f"Feature attributions shape: got {attrs[0].shape}")
    _check(mod_imp.shape == (200, 2), f"Modality importance shape: got {mod_imp.shape}")
    _check(np.allclose(mod_imp.sum(axis=-1), 1.0, atol=1e-4), "Modality importances per instance sum to 1.0")

    # In R1 (causal dominant), causal attribution magnitude should dominate cue attribution
    mean_imp1 = float(np.mean(mod_imp[:, 0]))
    mean_imp2 = float(np.mean(mod_imp[:, 1]))
    _check(mean_imp1 > mean_imp2, f"R1 causal attribution ({mean_imp1:.4f}) > cue attribution ({mean_imp2:.4f})")

    print("=" * 62)
    if failures == 0:
        print("ALL D1 EXPLANATION SELF-TESTS PASSED")
    else:
        print(f"FAILED: {failures} test(s) failed")
    print("=" * 62)

    return failures


if __name__ == "__main__":
    n_fail = _self_test()
    sys.exit(n_fail)
