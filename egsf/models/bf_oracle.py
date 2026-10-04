"""
egsf/models/bf_oracle.py
─────────────────────────
BF-Oracle: Base Fusion Oracle Baseline (Model 3) — EGSF v8.0, Step 1.4.

An oracle multimodal baseline that trains Base Fusion either:
1. On cue-broken training environments where spurious shortcut correlations are absent, or
2. With oracle masking of non-causal shortcut features.

Provides the upper-bound invariant fusion performance benchmark across all regimes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Make project root importable
sys.path.insert(0, str(Path(__file__).parents[2]))
from egsf.models.bf import BaseFusionMLP, train_bf_mod
from egsf.utils.reproducibility import seed_everything


class BFOracle(nn.Module):
    """
    BF-Oracle Wrapper.

    Uses BaseFusionMLP trained under oracle conditions (e.g., cue-broken envs or masked shortcuts).
    """

    def __init__(
        self,
        in_dims: List[int] = [8, 8],
        num_classes: int = 4,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bf = BaseFusionMLP(
            in_dims=in_dims,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

    def forward(self, xs: List[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        return self.bf(xs)


def train_bf_oracle(
    xs_train: List[torch.Tensor | np.ndarray],
    y_train: torch.Tensor | np.ndarray,
    xs_val: List[torch.Tensor | np.ndarray],
    y_val: torch.Tensor | np.ndarray,
    in_dims: List[int] = [8, 8],
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
) -> Tuple[BFOracle, Dict[str, list]]:
    """
    Train BF-Oracle model.
    """
    seed_everything(seed)
    model, history = train_bf_mod(
        xs_train=xs_train,
        y_train=y_train,
        xs_val=xs_val,
        y_val=y_val,
        in_dims=in_dims,
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

    oracle_wrapper = BFOracle(
        in_dims=in_dims,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    )
    oracle_wrapper.bf = model
    return oracle_wrapper, history


def _self_test() -> int:
    """Self-test BF-Oracle (Model 3) on JDB-S cue-broken split training."""
    from egsf.data.jdb_s import generate_jdbs

    print("=" * 62)
    print("BF-ORACLE (MODEL 3) SELF-TEST")
    print("=" * 62)

    failures = 0

    def _check(cond: bool, msg: str) -> None:
        nonlocal failures
        tag = "[PASS]" if cond else "[FAIL]"
        print(f"  {tag} {msg}")
        if not cond:
            failures += 1

    # 1. Forward pass shape
    oracle = BFOracle(in_dims=[8, 8], num_classes=4)
    x1 = torch.randn(32, 8)
    x2 = torch.randn(32, 8)
    logits = oracle([x1, x2])
    _check(logits.shape == (32, 4), f"Forward pass shape (32, 4): got {logits.shape}")

    # 2. Train BF-Oracle on R2 shortcut regime using cue-broken environment (oracle invariant dataset)
    ds = generate_jdbs("R2", rho_corr=0.9, seed=0)
    # Oracle train split: train on conflict_dev (cue-broken) split, validate on cal_g (cue-broken)
    m_oracle, hist = train_bf_oracle(
        [ds["conflict_dev"]["X1"], ds["conflict_dev"]["X2"]], ds["conflict_dev"]["y"],
        [ds["cal_g"]["X1"], ds["cal_g"]["X2"]], ds["cal_g"]["y"],
        in_dims=[8, 8], seed=0, verbose=False
    )

    # Evaluate on test_conflict (cue-broken test set) — should NOT collapse like standard BF!
    m_oracle.eval()
    with torch.no_grad():
        x1_conf = torch.tensor(ds["test_conflict"]["X1"], dtype=torch.float32)
        x2_conf = torch.tensor(ds["test_conflict"]["X2"], dtype=torch.float32)
        y_conf  = torch.tensor(ds["test_conflict"]["y"], dtype=torch.long)
        preds = m_oracle([x1_conf, x2_conf]).argmax(dim=-1)
        acc_conf = (preds == y_conf).float().mean().item()

    _check(acc_conf > 0.45, f"BF-Oracle avoids shortcut collapse on test_conflict: acc={acc_conf:.4f} > 0.45 (chance=0.25)")

    print("=" * 62)
    if failures == 0:
        print("ALL BF-ORACLE SELF-TESTS PASSED")
    else:
        print(f"FAILED: {failures} test(s) failed")
    print("=" * 62)

    return failures


if __name__ == "__main__":
    n_fail = _self_test()
    sys.exit(n_fail)
