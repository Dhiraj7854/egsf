"""
egsf/models/u_mod.py
───────────────────
U-Mod: Unimodal MLP Baseline (Model 1) — EGSF v8.0, Step 1.2.

A standalone unimodal classifier taking a single modality X_m in R^d
and mapping to class logits in R^K using a multi-layer perceptron.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Make project root importable
sys.path.insert(0, str(Path(__file__).parents[2]))
from egsf.utils.reproducibility import seed_everything


class UModMLP(nn.Module):
    """
    Unimodal MLP Classifier.

    Parameters
    ----------
    in_dim : int
        Input feature dimension (e.g. d=8)
    num_classes : int
        Number of output classes (e.g. K=4)
    hidden_dim : int
        Hidden dimension size (default 64)
    n_layers : int
        Number of hidden layers (default 2)
    dropout : float
        Dropout probability (default 0.1)
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
        layers = []
        curr_dim = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            curr_dim = hidden_dim

        layers.append(nn.Linear(curr_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (N, in_dim) -> logits: (N, num_classes)."""
        return self.net(x)


def train_u_mod(
    X_train: torch.Tensor | np.ndarray,
    y_train: torch.Tensor | np.ndarray,
    X_val: torch.Tensor | np.ndarray,
    y_val: torch.Tensor | np.ndarray,
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
) -> Tuple[UModMLP, Dict[str, list]]:
    """
    Train a U-Mod model with early stopping based on validation loss.

    Returns
    -------
    model : trained UModMLP (restored to best validation loss weights)
    history : dict containing training and validation loss/acc trajectories
    """
    seed_everything(seed)

    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.long)
    if not isinstance(X_val, torch.Tensor):
        X_val = torch.tensor(X_val, dtype=torch.float32)
    if not isinstance(y_val, torch.Tensor):
        y_val = torch.tensor(y_val, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = UModMLP(
        in_dim=in_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    for epoch in range(max_epochs):
        # --- Training Loop ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(by)
            preds = logits.argmax(dim=-1)
            correct += (preds == by).sum().item()
            total += len(by)

        epoch_tr_loss = running_loss / total
        epoch_tr_acc  = correct / total

        # --- Validation Loop ---
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx)
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

        if verbose:
            print(f"Epoch {epoch+1:03d} | Train Loss: {epoch_tr_loss:.4f} Acc: {epoch_tr_acc:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

        # Early stopping logic
        if epoch_val_loss < best_val_loss - min_delta:
            best_val_loss = epoch_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history


def _self_test() -> int:
    """Self-test U-Mod architecture and training loop on synthetic JDB-S data."""
    import numpy as np
    from egsf.data.jdb_s import generate_jdbs

    print("=" * 62)
    print("U-MOD (MODEL 1) SELF-TEST")
    print("=" * 62)

    failures = 0

    def _check(cond: bool, msg: str) -> None:
        nonlocal failures
        tag = "[PASS]" if cond else "[FAIL]"
        print(f"  {tag} {msg}")
        if not cond:
            failures += 1

    # 1. Forward pass tensor shapes
    model = UModMLP(in_dim=8, num_classes=4, hidden_dim=64, n_layers=2)
    x_dummy = torch.randn(32, 8)
    logits = model(x_dummy)
    _check(logits.shape == (32, 4), f"Forward pass shape (32, 4): got {logits.shape}")

    # 2. Train U-Mod on R1 causal modality X1
    ds = generate_jdbs("R1", rho_corr=0.8, seed=0)
    m_causal, hist_c = train_u_mod(
        ds["train"]["X1"], ds["train"]["y"],
        ds["val_id"]["X1"], ds["val_id"]["y"],
        seed=0, verbose=False
    )
    val_acc_c = hist_c["val_acc"][-1]
    _check(val_acc_c > 0.85, f"R1 Causal modality val_acc > 0.85: got {val_acc_c:.4f}")

    # 3. Evaluate U-Mod on cue-broken test split (causal predictor should stay strong)
    m_causal.eval()
    with torch.no_grad():
        X_conf = torch.tensor(ds["test_conflict"]["X1"], dtype=torch.float32)
        y_conf = torch.tensor(ds["test_conflict"]["y"], dtype=torch.long)
        preds = m_causal(X_conf).argmax(dim=-1)
        acc_conf = (preds == y_conf).float().mean().item()
    _check(acc_conf > 0.85, f"R1 Causal predictor test_conflict acc > 0.85: got {acc_conf:.4f}")

    print("=" * 62)
    if failures == 0:
        print("ALL U-MOD SELF-TESTS PASSED")
    else:
        print(f"FAILED: {failures} test(s) failed")
    print("=" * 62)

    return failures


if __name__ == "__main__":
    n_fail = _self_test()
    sys.exit(n_fail)
