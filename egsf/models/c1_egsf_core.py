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
from torch.utils.data import DataLoader, TensorDataset

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


def train_c1_egsf(
    xs_train: List[torch.Tensor | np.ndarray],
    y_train: torch.Tensor | np.ndarray,
    xs_val: List[torch.Tensor | np.ndarray],
    y_val: torch.Tensor | np.ndarray,
    budget_bounds: List[float] | np.ndarray = [1.0, 1.0],
    rel_train: Optional[np.ndarray] = None,
    rel_val: Optional[np.ndarray] = None,
    in_dims: List[int] = [8, 8],
    num_classes: int = 4,
    hidden_dim: int = 64,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 100,
    patience: int = 10,
    min_delta: float = 1e-4,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[C1EGSFCore, Dict[str, list]]:
    """
    Train C1 EGSF-Core model with budget constraints and early stopping.

    Parameters
    ----------
    rel_train : optional (N_train, M) pre-computed D0 reliance for each training sample.
                If provided, passed as the ``reliances`` arg to the model each batch so
                the D3 gate learns from real reliance signals instead of uniform defaults.
    rel_val   : optional (N_val, M) pre-computed D0 reliance for validation samples.
    """
    seed_everything(seed)
    xs_tr_t = [torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x for x in xs_train]
    xs_v_t  = [torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x for x in xs_val]
    y_tr_t  = torch.tensor(y_train, dtype=torch.long) if isinstance(y_train, np.ndarray) else y_train
    y_v_t   = torch.tensor(y_val, dtype=torch.long) if isinstance(y_val, np.ndarray) else y_val

    # Include pre-computed reliance in dataset if provided
    rel_tr_t = torch.tensor(rel_train, dtype=torch.float32) if rel_train is not None else None
    rel_v_t  = torch.tensor(rel_val,   dtype=torch.float32) if rel_val   is not None else None
    _use_rel = rel_tr_t is not None

    train_ds = TensorDataset(*xs_tr_t, rel_tr_t, y_tr_t) if _use_rel else TensorDataset(*xs_tr_t, y_tr_t)
    val_ds   = TensorDataset(*xs_v_t,  rel_v_t,  y_v_t)  if _use_rel else TensorDataset(*xs_v_t,  y_v_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = C1EGSFCore(in_dims=in_dims, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)
    model.set_budget_bounds(budget_bounds)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            if _use_rel:
                b_xs = list(batch[:-2])
                b_rel = batch[-2]
                by   = batch[-1]
            else:
                b_xs = list(batch[:-1])
                b_rel = None
                by   = batch[-1]

            optimizer.zero_grad()
            logits, gates = model(b_xs, reliances=b_rel)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(by)
            preds = logits.argmax(dim=-1)
            correct += (preds == by).sum().item()
            total += len(by)

        epoch_tr_loss = running_loss / total
        epoch_tr_acc  = correct / total

        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                if _use_rel:
                    b_xs = list(batch[:-2])
                    b_rel = batch[-2]
                    by   = batch[-1]
                else:
                    b_xs = list(batch[:-1])
                    b_rel = None
                    by   = batch[-1]

                logits, gates = model(b_xs, reliances=b_rel)
                loss = criterion(logits, by)
                running_val_loss += loss.item() * len(by)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == by).sum().item()
                val_total += len(by)

        epoch_val_loss = running_val_loss / val_total
        epoch_val_acc  = val_correct / val_total

        history["train_loss"].append(epoch_tr_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_tr_acc)
        history["val_acc"].append(epoch_val_acc)

        if epoch_val_loss < best_val_loss - min_delta:
            best_val_loss = epoch_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history


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

    c1_system, _ = train_c1_egsf(
        [ds["train"]["X1"], ds["train"]["X2"]], ds["train"]["y"],
        [ds["val_id"]["X1"], ds["val_id"]["X2"]], ds["val_id"]["y"],
        budget_bounds=B_kappa[0.1],
        in_dims=[8, 8], seed=0
    )

    x1 = torch.randn(32, 8)
    x2 = torch.randn(32, 8)
    logits, gates = c1_system([x1, x2])

    _check(logits.shape == (32, 4), f"Logits shape (32, 4): got {logits.shape}")
    _check(gates.shape == (32, 2), f"Gates shape (32, 2): got {gates.shape}")
    _check(torch.allclose(gates.sum(dim=-1), torch.ones(32), atol=1e-4), "Gates sum to 1.0 per sample")
    _check((gates[:, 1] <= B_kappa[0.1][1] + 1e-4).all(), "Cue gate strictly bounded by D2 budget bound")

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
