"""
egsf/models/bal_u.py
────────────────────
BAL-U: Budget-Aware Baseline - Unimodal (Model 4) — EGSF v8.0, Step 1.5.

A unimodal baseline scaled by pre-computed or ground-truth information budget B_m.
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
from egsf.models.u_mod import UModMLP, train_u_mod
from egsf.utils.reproducibility import seed_everything


class BALUnimodal(nn.Module):
    """
    BAL-U: Budget-Aware Unimodal Model.
    Scales unimodal model output logits by budget fraction B_m.
    """

    def __init__(
        self,
        in_dim: int = 8,
        num_classes: int = 4,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.u_mod = UModMLP(
            in_dim=in_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.register_buffer("budget_m", torch.tensor(1.0, dtype=torch.float32))

    def set_budget(self, budget_m: float) -> None:
        self.budget_m = torch.tensor(float(budget_m), dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.u_mod(x)
        return self.budget_m * logits


def train_bal_u(
    X_train: torch.Tensor | np.ndarray,
    y_train: torch.Tensor | np.ndarray,
    X_val: torch.Tensor | np.ndarray,
    y_val: torch.Tensor | np.ndarray,
    budget_m: float = 1.0,
    in_dim: int = 8,
    num_classes: int = 4,
    hidden_dim: int = 64,
    n_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 100,
    patience: int = 10,
    min_delta: float = 1e-4,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[BALUnimodal, Dict[str, list]]:
    """
    Train a BAL-U model.
    """
    seed_everything(seed)
    u_mod, history = train_u_mod(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        in_dim=in_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        seed=seed,
        verbose=verbose,
    )

    bal_model = BALUnimodal(
        in_dim=in_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    )
    bal_model.u_mod = u_mod
    bal_model.set_budget(budget_m)

    return bal_model, history


def _self_test() -> int:
    """Self-test BAL-U (Model 4) on synthetic JDB-S data."""
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

    # 1. Forward pass shape and budget scaling
    bal = BALUnimodal(in_dim=8, num_classes=4)
    bal.set_budget(0.8)
    x = torch.randn(32, 8)
    logits = bal(x)
    _check(logits.shape == (32, 4), f"Logits shape (32, 4): got {logits.shape}")

    # 2. Train BAL-U on R1 causal modality X1 with budget B*_1
    ds = generate_jdbs("R1", rho_corr=0.8, seed=0)
    b_star_1 = float(ds["ground_truth"]["B_star"][0])
    m_bal, hist = train_bal_u(
        ds["train"]["X1"], ds["train"]["y"],
        ds["val_id"]["X1"], ds["val_id"]["y"],
        budget_m=b_star_1,
        in_dim=8, seed=0, verbose=False
    )
    m_bal.eval()
    with torch.no_grad():
        x_val = torch.tensor(ds["val_id"]["X1"], dtype=torch.float32)
        y_val = torch.tensor(ds["val_id"]["y"], dtype=torch.long)
        preds = m_bal(x_val).argmax(dim=-1)
        acc_val = (preds == y_val).float().mean().item()

    _check(acc_val > 0.85, f"R1 BAL-U val_acc > 0.85 (with budget B*_1={b_star_1:.4f}): got {acc_val:.4f}")

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
