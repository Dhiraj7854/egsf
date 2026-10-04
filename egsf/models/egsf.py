"""
egsf/models/egsf.py
───────────────────
G-EGSF: Gated Explanation-Guided Selective Fusion (Model 3) — EGSF v8.0, Step 1.4.

Core G-EGSF Architecture:
1. Modality-specific Encoders: h_m = Layer_m(X_m)
2. Explanation/Reliance Module: Computes per-modality reliance/importance scores e_m
3. Selective Gate: Gating mechanism g_m = sigmoid(W_g [h_1, ..., h_M] + b_g) constrained by budget B_m
4. Selective Fusion: h_fused = sum_m g_m * h_m
5. Classifier Head: Predicts class logits y_hat from h_fused
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Make project root importable
sys.path.insert(0, str(Path(__file__).parents[2]))
from egsf.utils.reproducibility import seed_everything


class GEGSF(nn.Module):
    """
    G-EGSF Architecture for M Modality Fusion.

    Parameters
    ----------
    in_dims : List[int]
        Input feature dimensions for each modality [d_1, ..., d_M]
    num_classes : int
        Number of output target classes (K=4)
    hidden_dim : int
        Feature embedding & hidden layer dimension (default 64)
    dropout : float
        Dropout probability (default 0.1)
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
        self.hidden_dim = hidden_dim

        # 1. Modality Encoders
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
            for dim in in_dims
        ])

        # 2. Gate Generator (takes concatenated embeddings)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * self.num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_modalities),
            nn.Sigmoid(),
        )

        # 3. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, xs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        xs: list of tensors [(N, d_1), ..., (N, d_M)]
        Returns:
            logits : (N, num_classes)
            gates  : (N, M) gating weights
        """
        # Encode modalities into hidden representation
        h_list = [enc(x) for enc, x in zip(self.encoders, xs)]  # M * (N, hidden_dim)

        # Concatenate for gate calculation
        h_cat = torch.cat(h_list, dim=-1)  # (N, M * hidden_dim)
        gates = self.gate_net(h_cat)       # (N, M)

        # Apply gates to modality representations
        h_fused = torch.zeros_like(h_list[0])
        for m in range(self.num_modalities):
            h_fused = h_fused + gates[:, m:m+1] * h_list[m]

        # Classification logits
        logits = self.classifier(h_fused)  # (N, num_classes)

        return logits, gates


def train_egsf_mod(
    xs_train: List[torch.Tensor | np.ndarray],
    y_train: torch.Tensor | np.ndarray,
    xs_val: List[torch.Tensor | np.ndarray],
    y_val: torch.Tensor | np.ndarray,
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
) -> Tuple[GEGSF, Dict[str, list]]:
    """
    Train G-EGSF model with early stopping.

    Returns
    -------
    model : trained GEGSF model
    history : dict containing training and validation trajectory
    """
    seed_everything(seed)

    xs_tr_t = [torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x for x in xs_train]
    xs_v_t  = [torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x for x in xs_val]
    y_tr_t  = torch.tensor(y_train, dtype=torch.long) if isinstance(y_train, np.ndarray) else y_train
    y_v_t   = torch.tensor(y_val, dtype=torch.long) if isinstance(y_val, np.ndarray) else y_val

    # Zip inputs for dataset
    train_ds = TensorDataset(*xs_tr_t, y_tr_t)
    val_ds   = TensorDataset(*xs_v_t, y_v_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = GEGSF(in_dims=in_dims, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
            logits, gates = model(b_xs)
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

                logits, gates = model(b_xs)
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
    """Self-test G-EGSF model architecture and gating on JDB-S synthetic data."""
    from egsf.data.jdb_s import generate_jdbs

    print("=" * 62)
    print("G-EGSF (MODEL 3 - GATED SELECTIVE FUSION) SELF-TEST")
    print("=" * 62)

    failures = 0

    def _check(cond: bool, msg: str) -> None:
        nonlocal failures
        tag = "[PASS]" if cond else "[FAIL]"
        print(f"  {tag} {msg}")
        if not cond:
            failures += 1

    # 1. Forward pass shapes and gate bounds
    model = GEGSF(in_dims=[8, 8], num_classes=4, hidden_dim=64)
    x1 = torch.randn(32, 8)
    x2 = torch.randn(32, 8)
    logits, gates = model([x1, x2])
    _check(logits.shape == (32, 4), f"Logits shape (32, 4): got {logits.shape}")
    _check(gates.shape == (32, 2), f"Gates shape (32, 2): got {gates.shape}")
    _check((gates >= 0.0).all() and (gates <= 1.0).all(), "Gates strictly bounded in [0, 1]")

    # 2. Train G-EGSF on R1 regime (causal dominant)
    ds = generate_jdbs("R1", rho_corr=0.8, seed=0)
    m_egsf, hist = train_egsf_mod(
        [ds["train"]["X1"], ds["train"]["X2"]], ds["train"]["y"],
        [ds["val_id"]["X1"], ds["val_id"]["X2"]], ds["val_id"]["y"],
        in_dims=[8, 8], seed=0, verbose=False
    )
    val_acc = hist["val_acc"][-1]
    _check(val_acc > 0.85, f"R1 G-EGSF val_acc > 0.85: got {val_acc:.4f}")

    print("=" * 62)
    if failures == 0:
        print("ALL G-EGSF SELF-TESTS PASSED")
    else:
        print(f"FAILED: {failures} test(s) failed")
    print("=" * 62)

    return failures


if __name__ == "__main__":
    n_fail = _self_test()
    sys.exit(n_fail)
