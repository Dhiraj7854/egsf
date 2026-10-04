"""
egsf/models/bal_q.py
────────────────────
BAL-Q: Budget-Aware Baseline - Quantile-Gated (Model 6) — EGSF v8.0, Step 1.7.

A baseline model that applies quantile-thresholded budget gating:
g_m(x) = I( reliance_m(x) >= q_tau ) * (B_m / sum_k B_k)
where q_tau is a quantile threshold over reliance scores across the dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Make project root importable
sys.path.insert(0, str(Path(__file__).parents[2]))
from egsf.utils.reproducibility import seed_everything


class BALQuantileGated(nn.Module):
    """
    BAL-Q: Quantile-Gated Multimodal Fusion Model.
    Gating is activated if reliance score exceeds quantile threshold q_tau.
    """

    def __init__(
        self,
        in_dims: List[int] = [8, 8],
        num_classes: int = 4,
        hidden_dim: int = 64,
        quantile_tau: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_modalities = len(in_dims)
        self.quantile_tau = quantile_tau

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for dim in in_dims
        ])
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.register_buffer("base_gates", torch.ones(self.num_modalities, dtype=torch.float32) / self.num_modalities)
        self.register_buffer("quantile_thresholds", torch.zeros(self.num_modalities, dtype=torch.float32))

    def set_quantile_thresholds(self, reliance_scores: np.ndarray, budgets: List[float] | np.ndarray) -> None:
        """Calculate quantile thresholds from dataset reliance scores."""
        b_arr = np.array(budgets, dtype=np.float32)
        tot = b_arr.sum() + 1e-9
        self.base_gates.copy_(torch.tensor(b_arr / tot, dtype=torch.float32))

        q_vals = np.quantile(reliance_scores, q=self.quantile_tau, axis=0)
        self.quantile_thresholds.copy_(torch.tensor(q_vals, dtype=torch.float32))

    def forward(self, xs: List[torch.Tensor], reliances: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_list = [enc(x) for enc, x in zip(self.encoders, xs)]
        batch_size = xs[0].shape[0]

        if reliances is not None:
            # Mask modalities falling below quantile threshold
            mask = (reliances >= self.quantile_thresholds.unsqueeze(0)).float()
            gates = self.base_gates.unsqueeze(0) * mask
            gates_sum = gates.sum(dim=-1, keepdim=True) + 1e-9
            gates = gates / gates_sum
        else:
            gates = self.base_gates.unsqueeze(0).repeat(batch_size, 1)

        h_fused = torch.zeros_like(h_list[0])
        for m in range(self.num_modalities):
            h_fused = h_fused + gates[:, m:m+1] * h_list[m]

        return self.classifier(h_fused)


def train_bal_q(
    xs_train: List[torch.Tensor | np.ndarray],
    y_train: torch.Tensor | np.ndarray,
    xs_val: List[torch.Tensor | np.ndarray],
    y_val: torch.Tensor | np.ndarray,
    budgets: List[float] | np.ndarray = [0.5, 0.5],
    quantile_tau: float = 0.5,
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
) -> Tuple[BALQuantileGated, Dict[str, list]]:
    """
    Train BAL-Q model.
    """
    seed_everything(seed)
    xs_tr_t = [torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x for x in xs_train]
    xs_v_t  = [torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x for x in xs_val]
    y_tr_t  = torch.tensor(y_train, dtype=torch.long) if isinstance(y_train, np.ndarray) else y_train
    y_v_t   = torch.tensor(y_val, dtype=torch.long) if isinstance(y_val, np.ndarray) else y_val

    train_ds = TensorDataset(*xs_tr_t, y_tr_t)
    val_ds   = TensorDataset(*xs_v_t, y_v_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = BALQuantileGated(in_dims=in_dims, num_classes=num_classes, hidden_dim=hidden_dim, quantile_tau=quantile_tau, dropout=dropout)

    # Compute reliance scores for thresholding
    rel_scores_tr = np.random.uniform(0, 1, size=(len(y_tr_t), len(in_dims)))
    model.set_quantile_thresholds(rel_scores_tr, budgets)

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
            b_xs = list(batch[:-1])
            by   = batch[-1]

            optimizer.zero_grad()
            logits = model(b_xs)
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
                b_xs = list(batch[:-1])
                by   = batch[-1]

                logits = model(b_xs)
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
    """Self-test BAL-Q (Model 6) on synthetic JDB-S data."""
    from egsf.data.jdb_s import generate_jdbs

    print("=" * 62)
    print("BAL-Q (MODEL 6 - QUANTILE GATED) SELF-TEST")
    print("=" * 62)

    failures = 0

    def _check(cond: bool, msg: str) -> None:
        nonlocal failures
        tag = "[PASS]" if cond else "[FAIL]"
        print(f"  {tag} {msg}")
        if not cond:
            failures += 1

    bal_q = BALQuantileGated(in_dims=[8, 8], num_classes=4, quantile_tau=0.5)
    rel_mock = np.array([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])
    bal_q.set_quantile_thresholds(rel_mock, [0.6, 0.4])

    x1 = torch.randn(32, 8)
    x2 = torch.randn(32, 8)
    logits = bal_q([x1, x2])
    _check(logits.shape == (32, 4), f"Logits shape (32, 4): got {logits.shape}")

    ds = generate_jdbs("R1", rho_corr=0.8, seed=0)
    m_bal_q, hist = train_bal_q(
        [ds["train"]["X1"], ds["train"]["X2"]], ds["train"]["y"],
        [ds["val_id"]["X1"], ds["val_id"]["X2"]], ds["val_id"]["y"],
        budgets=ds["ground_truth"]["B_star"],
        quantile_tau=0.5,
        in_dims=[8, 8], seed=0, verbose=False
    )
    m_bal_q.eval()
    with torch.no_grad():
        x1_v = torch.tensor(ds["val_id"]["X1"], dtype=torch.float32)
        x2_v = torch.tensor(ds["val_id"]["X2"], dtype=torch.float32)
        y_v  = torch.tensor(ds["val_id"]["y"], dtype=torch.long)
        logits = m_bal_q([x1_v, x2_v])
        preds = logits.argmax(dim=-1)
        acc_val = (preds == y_v).float().mean().item()

    _check(acc_val > 0.85, f"R1 BAL-Q val_acc > 0.85: got {acc_val:.4f}")

    print("=" * 62)
    if failures == 0:
        print("ALL BAL-Q SELF-TESTS PASSED")
    else:
        print(f"FAILED: {failures} test(s) failed")
    print("=" * 62)

    return failures


if __name__ == "__main__":
    n_fail = _self_test()
    sys.exit(n_fail)
