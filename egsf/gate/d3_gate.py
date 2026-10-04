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


class SelectiveGateD3(nn.Module):
    """
    D3 Selective Gate Module.
    Combines reliance, explanation features, and budget bounds into constrained gate weights g_m(x).
    """

    def __init__(
        self,
        num_modalities: int = 2,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        # Input features: per-modality [R_m, ||E_m||_1] -> 2 features per modality
        self.gate_net = nn.Sequential(
            nn.Linear(num_modalities * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities),
            nn.Sigmoid(),
        )

    def forward(
        self,
        reliances: torch.Tensor,
        explanations: torch.Tensor,
        budget_bounds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        reliances: (N, M)
        explanations: (N, M)
        budget_bounds: (M,) or (N, M)

        Returns:
        constrained_gates: (N, M) in [0, B_m^(kappa)]
        raw_gates: (N, M) in [0, 1]
        """
        gate_in = torch.cat([reliances, explanations], dim=-1)  # (N, 2M)
        raw_gates = self.gate_net(gate_in)                       # (N, M)

        # Enforce budget constraint: g_m(x) <= B_m^(kappa)
        if budget_bounds.ndim == 1:
            budget_bounds = budget_bounds.unsqueeze(0)  # (1, M)

        constrained_gates = torch.minimum(raw_gates, budget_bounds)
        return constrained_gates, raw_gates


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

    ds = generate_jdbs("R1", rho_corr=0.8, seed=0)
    B_star, B_kappa = compute_d2_budget(ds, kappa_grid=[0.1])
    b_bound = torch.tensor(B_kappa[0.1], dtype=torch.float32)

    gate_mod = SelectiveGateD3(num_modalities=2)
    rel_mock = torch.tensor([[0.9, 0.1], [0.8, 0.2]], dtype=torch.float32)
    exp_mock = torch.tensor([[0.85, 0.15], [0.75, 0.25]], dtype=torch.float32)

    c_gates, r_gates = gate_mod(rel_mock, exp_mock, b_bound)

    _check(c_gates.shape == (2, 2), f"Constrained gates shape (2, 2): got {c_gates.shape}")
    _check((c_gates[:, 1] <= b_bound[1] + 1e-6).all(), f"Cue gate <= budget bound B_cue={b_bound[1]:.4f}")
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
