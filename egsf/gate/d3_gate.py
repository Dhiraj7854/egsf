"""
egsf/gate/d3_gate.py
────────────────────
D3: Selective Gate Module (Model 11 / Baseline D3) — EGSF v8.0, Step 1.12.

Computes instance-level selective modality gates g_m(x) in [0, 1] conditioned on:
1. D0 Reliance scores R_m(x)
2. D1 Explanation attributions E_m(x)
3. D2 Budget upper bounds B_m^(kappa)

Enforces budget-constrained gating: g_m(x) = min( g_raw(x, R, E), B_m^(kappa) ).
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


def project_excess_reliance(
    reliances: torch.Tensor,
    explanations: Optional[torch.Tensor],
    budget_bounds: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Excess Reliance (ER) and projects modality allocation q onto the simplex
    sum(q) = 1 subject to q_m <= Cap_m.

    Parameters
    ----------
    reliances : (N, M) D0 interventional reliance scores
    explanations : optional (N, M) D1 explanation attributions
    budget_bounds : (M,) or (N, M) D2 upper budget bounds B_m^(kappa)

    Returns
    -------
    q : (N, M) constrained gate weights (sums to 1.0, respects caps)
    ER : (N, M) Excess Reliance = max(0, reliances - budget_bounds)
    alpha : (N, M) starting distribution
    """
    N, M = reliances.shape
    if budget_bounds.ndim == 1:
        budget_bounds = budget_bounds.unsqueeze(0).expand(N, M)

    # 1. Explicit Excess Reliance calculation: ER_m = max(0, rho_m - B_m^(kappa))
    ER = torch.clamp(reliances - budget_bounds, min=0.0)

    # 2. Starting distribution alpha (explanation-weighted or uniform)
    if explanations is not None and (explanations.sum(dim=-1) > 0).all():
        alpha = explanations / (explanations.sum(dim=-1, keepdim=True) + eps)
    else:
        alpha = torch.ones_like(reliances) / float(M)

    # 3. Penalize starting distribution by Excess Reliance
    alpha_prime = alpha * torch.clamp(1.0 - ER, min=eps)
    alpha_prime = alpha_prime / (alpha_prime.sum(dim=-1, keepdim=True) + eps)

    # 4. Modality Caps
    raw_caps = torch.clamp(budget_bounds, min=0.0, max=1.0)
    tot_caps = raw_caps.sum(dim=-1, keepdim=True)

    # Handle infeasible normalization (sum of caps < 1.0) safely
    is_infeasible = tot_caps < (1.0 - 1e-5)
    eff_caps = torch.where(
        is_infeasible,
        raw_caps / (tot_caps + eps),
        raw_caps
    )

    # 5. Iterative Water-Filling Simplex Projection onto q_m <= eff_caps_m
    q = torch.minimum(alpha_prime, eff_caps)
    for _ in range(M + 1):
        missing = 1.0 - q.sum(dim=-1, keepdim=True)
        missing = torch.clamp(missing, min=0.0)
        if (missing < 1e-6).all():
            break

        headroom = torch.clamp(eff_caps - q, min=0.0)
        tot_headroom = headroom.sum(dim=-1, keepdim=True) + eps
        alloc = missing * (headroom / tot_headroom)
        q = torch.minimum(q + alloc, eff_caps)

    # Final normalization safeguard
    q_sum = q.sum(dim=-1, keepdim=True) + eps
    q = q / q_sum

    return q, ER, alpha


class SelectiveGateD3(nn.Module):
    """
    D3 Selective Gate Module.
    Combines D0 reliance, D1 explanations, and D2 budget bounds into
    Excess Reliance (ER)-constrained simplex modality gate weights q_m(x).
    """

    def __init__(
        self,
        num_modalities: int = 2,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.num_modalities = num_modalities

    def forward(
        self,
        reliances: torch.Tensor,
        explanations: Optional[torch.Tensor],
        budget_bounds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        reliances: (N, M)
        explanations: (N, M) or None
        budget_bounds: (M,) or (N, M)

        Returns:
        constrained_gates (q): (N, M) summing to 1.0 and respecting budget caps
        ER: (N, M) Excess Reliance tensor
        """
        q, ER, _ = project_excess_reliance(reliances, explanations, budget_bounds)
        return q, ER


def _self_test() -> int:
    """Self-test D3 Selective Gate Module on synthetic data."""
    from egsf.data.jdb_s import generate_jdbs
    from egsf.budget.d2_budget import compute_d2_budget

    print("=" * 62)
    print("D3 SELECTIVE GATE MODULE (MODEL 11) SELF-TEST")
    print("=" * 62)

    failures = 0

    def _check(cond: bool, msg: str) -> None:
        nonlocal failures
        tag = "[PASS]" if cond else "[FAIL]"
        print(f"  {tag} {msg}")
        if not cond:
            failures += 1

    # 1. ER = 0 when rho <= B
    rel_in_bounds = torch.tensor([[0.4, 0.3]], dtype=torch.float32)
    b_bounds = torch.tensor([0.5, 0.5], dtype=torch.float32)
    q1, er1, _ = project_excess_reliance(rel_in_bounds, None, b_bounds)
    _check(torch.allclose(er1, torch.tensor([[0.0, 0.0]])), "ER = 0 when rho <= B")
    _check(torch.allclose(q1, torch.tensor([[0.5, 0.5]])), "Identity uniform gate when no cap is active")

    # 2. Positive ER when rho > B
    rel_exceed = torch.tensor([[0.1, 0.9]], dtype=torch.float32)
    b_exceed = torch.tensor([0.8, 0.4], dtype=torch.float32)
    q2, er2, _ = project_excess_reliance(rel_exceed, None, b_exceed)
    _check(er2[0, 1].item() > 0.49, f"Positive ER for cue when rho > B: got {er2[0,1]:.4f}")
    _check(abs(q2.sum().item() - 1.0) < 1e-5, f"Projection preserves sum(q)=1: got {q2.sum().item():.6f}")
    _check((q2[0] <= b_exceed + 1e-5).all(), "q respects all caps when total cap >= 1.0")

    # 3. R2-style case: cue reliance high but budget is zero
    rel_r2 = torch.tensor([[0.15, 0.85]], dtype=torch.float32)
    b_r2 = torch.tensor([0.20, 0.00], dtype=torch.float32)
    q3, er3, _ = project_excess_reliance(rel_r2, None, b_r2)
    _check(er3[0, 1].item() > 0.84, f"R2 cue ER > 0.84: got {er3[0, 1]:.4f}")
    _check(torch.allclose(q3, torch.tensor([[1.0, 0.0]])), f"R2 cue gate capped to 0.0: got {q3.numpy()}")
    _check(abs(q3.sum().item() - 1.0) < 1e-5, "R2 q sums to 1.0")

    # 4. Standard JDB-S R1 integration test
    ds = generate_jdbs("R1", rho_corr=0.8, seed=0)
    B_star, B_kappa = compute_d2_budget(ds, kappa_grid=[0.1])
    b_bound = torch.tensor(B_kappa[0.1], dtype=torch.float32)

    gate_mod = SelectiveGateD3(num_modalities=2)
    rel_mock = torch.tensor([[0.9, 0.1], [0.8, 0.2]], dtype=torch.float32)
    exp_mock = torch.tensor([[0.85, 0.15], [0.75, 0.25]], dtype=torch.float32)

    c_gates, er_gates = gate_mod(rel_mock, exp_mock, b_bound)

    _check(c_gates.shape == (2, 2), f"Constrained gates shape (2, 2): got {c_gates.shape}")
    _check(torch.allclose(c_gates.sum(dim=-1), torch.tensor([1.0, 1.0]), atol=1e-4), "All gate rows sum to 1.0")
    _check((c_gates >= 0.0).all(), "Gates non-negative")

    print("=" * 62)
    if failures == 0:
        print("ALL D3 SELECTIVE GATE SELF-TESTS PASSED")
    else:
        print(f"FAILED: {failures} test(s) failed")
    print("=" * 62)

    return failures


if __name__ == "__main__":
    n_fail = _self_test()
    sys.exit(n_fail)
