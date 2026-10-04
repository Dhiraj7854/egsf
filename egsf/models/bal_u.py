"""
egsf/models/bal_u.py
────────────────────
BAL-U: Budget-Aware Baseline - Unimodal (Model 4) — EGSF v8.0, Step 1.5.

A unimodal baseline scaled by pre-computed or ground-truth information budget B_m.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# Make project root importable
sys.path.insert(0, str(Path(__file__).parents[2]))
from egsf.models.u_mod import UModMLP, train_u_mod
from egsf.utils.reproducibility import seed_everything


class BALUnimodal(nn.Module):
    """
    BAL-U: Budget-Aware Unimodal Model.
    Scales unimodal model output logits by budget fraction B_m.
    """

    def __init__(self, in_dim: int = 8, num_classes: int = 4, hidden_dim: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.u_mod = UModMLP(in_dim=in_dim, num_classes=num_classes, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
        self.budget_m = 1.0

    def set_budget(self, budget_m: float) -> None:
        self.budget_m = budget_m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.u_mod(x)
        return self.budget_m * logits


def _self_test() -> int:
    """Self-test BAL-U (Model 4)."""
    from egsf.data.jdb_s import generate_jdbs

    print("=" * 62)
    print("BAL-U (MODEL 4) SELF-TEST")
    print("=" * 62)

    failures = 0

    def _check(cond: bool, msg: str) -> None:
        nonlocal failures
        tag = "[PASS]" if cond else "[FAIL]"
        print(f"  {tag} {msg}")
        if not cond:
            failures += 1

    bal = BALUnimodal(in_dim=8, num_classes=4)
    bal.set_budget(0.8)
    x = torch.randn(32, 8)
    logits = bal(x)
    _check(logits.shape == (32, 4), f"Logits shape (32, 4): got {logits.shape}")

    print("=" * 62)
    if failures == 0:
        print("ALL BAL-U SELF-TESTS PASSED")
    else:
        print(f"FAILED: {failures} test(s) failed")
    print("=" * 62)

    return failures


if __name__ == "__main__":
    n_fail = _self_test()
    sys.exit(n_fail)
