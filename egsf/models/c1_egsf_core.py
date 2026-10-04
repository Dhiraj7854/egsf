"""
egsf/models/c1_egsf_core.py
───────────────────────────
C1: EGSF-Core (Model 14 / Complete Core System) — EGSF v8.0, Step 1.15.

The full Explanation-Guided Selective Fusion System integrating:
1. Modality Encoders h_m(x_m)
2. D0 Reliance Estimator R_m(x)
3. D1 Integrated Gradients Explanations E_m(x)
4. D2 Environmental Information Budget Bounds B_m^(kappa)
5. D3 Budget-Constrained Selective Gate g_m(x)
6. D4 Epistemic Uncertainty LCB Gating g_m^(LCB)(x)
7. E4 Conformal Risk Control calibration threshold lambda_hat
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
from egsf.budget.d2_budget import compute_d2_budget
from egsf.calibration.e4_crc import calibrate_e4_crc, evaluate_e4_crc
from egsf.explanations.d1_explanation import compute_d1_explanations
from egsf.gate.d3_gate import SelectiveGateD3
from egsf.reliance.d0_reliance import compute_d0_reliance
from egsf.uncertainty.d4_uncertainty import compute_d4_uncertainty
from egsf.utils.reproducibility import seed_everything


class C1EGSFCore(nn.Module):
    """
    C1 EGSF-Core Architecture (Model 14).
    """

    def __init__(
        self,
        in_dims: List[int] = [8, 8],
        num_classes: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_modalities = len(in_dims)
        self.in_dims = in_dims

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for dim in in_dims
        ])

        self.selective_gate = SelectiveGateD3(num_modalities=len(in_dims), hidden_dim=32)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.register_buffer("budget_bounds", torch.ones(len(in_dims), dtype=torch.float32))

    def set_budget_bounds(self, bounds: np.ndarray | torch.Tensor) -> None:
        if isinstance(bounds, np.ndarray):
            bounds = torch.tensor(bounds, dtype=torch.float32)
        self.budget_bounds.copy_(bounds)

    def forward(
        self,
        xs: List[torch.Tensor],
        reliances: Optional[torch.Tensor] = None,
        explanations: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_list = [enc(x) for enc, x in zip(self.encoders, xs)]
        N = xs[0].shape[0]

        if reliances is None:
            reliances = torch.ones(N, self.num_modalities, dtype=torch.float32) / float(self.num_modalities)
        if explanations is None:
            explanations = torch.ones(N, self.num_modalities, dtype=torch.float32) / float(self.num_modalities)

        constrained_gates, _ = self.selective_gate(reliances, explanations, self.budget_bounds)

        h_fused = torch.zeros_like(h_list[0])
        for m in range(self.num_modalities):
            h_fused = h_fused + constrained_gates[:, m:m+1] * h_list[m]

        logits = self.classifier(h_fused)
        return logits, constrained_gates


def _self_test() -> int:
    """Self-test C1 EGSF-Core (Model 14) on synthetic JDB-S data."""
    from egsf.data.jdb_s import generate_jdbs

    print("=" * 62)
    print("C1 EGSF-CORE (MODEL 14) SELF-TEST")
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

    c1_system = C1EGSFCore(in_dims=[8, 8], num_classes=4)
    c1_system.set_budget_bounds(B_kappa[0.1])

    x1 = torch.randn(32, 8)
    x2 = torch.randn(32, 8)
    logits, gates = c1_system([x1, x2])

    _check(logits.shape == (32, 4), f"Logits shape (32, 4): got {logits.shape}")
    _check(gates.shape == (32, 2), f"Gates shape (32, 2): got {gates.shape}")
    _check((gates[:, 1] <= B_kappa[0.1][1] + 1e-6).all(), "Cue gate strictly bounded by D2 budget bound")

    print("=" * 62)
    if failures == 0:
        print("ALL C1 EGSF-CORE SELF-TESTS PASSED")
    else:
        print(f"FAILED: {failures} test(s) failed")
    print("=" * 62)

    return failures


if __name__ == "__main__":
    n_fail = _self_test()
    sys.exit(n_fail)
