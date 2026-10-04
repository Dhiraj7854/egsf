"""
egsf/models/bf.py
────────────────
BF: Base Fusion (Model 2) — EGSF v8.0, Step 1.3.

A joint multimodal baseline classifier that concatenates feature modalities
X_1, X_2, ..., X_M into a single joint vector and processes them through an MLP.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Make project root importable
sys.path.insert(0, str(Path(__file__).parents[2]))
from egsf.utils.reproducibility import seed_everything


class BaseFusionMLP(nn.Module):
    """
    Base Fusion (BF) Multimodal MLP Classifier.

    Parameters
    ----------
    in_dims : List[int]
        List of feature dimensions for each modality, e.g. [d, d]
    num_classes : int
        Number of target classes (e.g. K=4)
    hidden_dim : int
        Hidden dimension size (default 64)
    n_layers : int
        Number of hidden layers (default 2)
    dropout : float
        Dropout probability (default 0.1)
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
        self.in_dims = in_dims
        total_in_dim = sum(in_dims)

        layers = []
        curr_dim = total_in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            curr_dim = hidden_dim

        layers.append(nn.Linear(curr_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, xs: List[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        xs: list of modality tensors [(N, d1), (N, d2), ...] OR concatenated tensor (N, d1+d2)
        Returns: logits (N, num_classes)
        """
        if isinstance(xs, list):
            x_cat = torch.cat(xs, dim=-1)
        else:
            x_cat = xs
        return self.net(x_cat)


def train_bf_mod(
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
) -> Tuple[BaseFusionMLP, Dict[str, list]]:
    """
    Train a Base Fusion (BF) model with early stopping based on validation loss.

    Returns
    -------
    model : trained BaseFusionMLP
    history : dict containing training and validation metrics trajectory
    """
    seed_everything(seed)

    # Concatenate modal inputs along feature dim
    cat_tr = np.concatenate(xs_train, axis=-1) if isinstance(xs_train[0], np.ndarray) else torch.cat(xs_train, dim=-1)
    cat_v  = np.concatenate(xs_val, axis=-1)   if isinstance(xs_val[0], np.ndarray)   else torch.cat(xs_val, dim=-1)

    if not isinstance(cat_tr, torch.Tensor):
        cat_tr = torch.tensor(cat_tr, dtype=torch.float32)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.long)
    if not isinstance(cat_v, torch.Tensor):
        cat_v = torch.tensor(cat_v, dtype=torch.float32)
    if not isinstance(y_val, torch.Tensor):
        y_val = torch.tensor(y_val, dtype=torch.long)

    train_ds = TensorDataset(cat_tr, y_train)
    val_ds   = TensorDataset(cat_v, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = BaseFusionMLP(
        in_dims=in_dims,
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
    """Self-test BF architecture and training loop on synthetic JDB-S data (R4 synergy test)."""
    from egsf.data.jdb_s import generate_jdbs

    print("=" * 62)
    print("BF (MODEL 2 - BASE FUSION) SELF-TEST")
    print("=" * 62)

    failures = 0

    def _check(cond: bool, msg: str) -> None:
        nonlocal failures
        tag = "[PASS]" if cond else "[FAIL]"
        print(f"  {tag} {msg}")
        if not cond:
            failures += 1

    # 1. Forward pass tensor shapes
    model = BaseFusionMLP(in_dims=[8, 8], num_classes=4, hidden_dim=64, n_layers=2)
    x1 = torch.randn(32, 8)
    x2 = torch.randn(32, 8)
    logits = model([x1, x2])
    _check(logits.shape == (32, 4), f"Forward pass shape (32, 4): got {logits.shape}")

    # 2. Train BF on R4 synergy regime (requires joint information to solve)
    ds = generate_jdbs("R4", rho_corr=0.8, seed=0)
    m_bf, hist_bf = train_bf_mod(
        [ds["train"]["X1"], ds["train"]["X2"]], ds["train"]["y"],
        [ds["val_id"]["X1"], ds["val_id"]["X2"]], ds["val_id"]["y"],
        in_dims=[8, 8], seed=0, verbose=False
    )
    val_acc = hist_bf["val_acc"][-1]
    _check(val_acc > 0.85, f"R4 Synergy joint val_acc > 0.85: got {val_acc:.4f}")

    print("=" * 62)
    if failures == 0:
        print("ALL BF SELF-TESTS PASSED")
    else:
        print(f"FAILED: {failures} test(s) failed")
    print("=" * 62)

    return failures


if __name__ == "__main__":
    n_fail = _self_test()
    sys.exit(n_fail)
