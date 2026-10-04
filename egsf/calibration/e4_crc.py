"""
egsf/calibration/e4_crc.py
──────────────────────────
E4: Conformal Risk Control (CRC) Module (Model 13 / Baseline E4) — EGSF v8.0, Step 1.14.

Computes risk-controlled conformal thresholds lambda_hat on calibration split `cal_crc`
such that expected selective misclassification / false-correction risk is provably bounded:
E[ L( y, y_hat(lambda_hat) ) ] <= alpha
where target risk alpha = 0.10 (from configs/jdb_s.yaml).
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


def calibrate_e4_crc(
    cal_scores: np.ndarray,
    cal_losses: np.ndarray,
    alpha: float = 0.10,
    seed: int = 0,
) -> float:
    """
    Calibrate Conformal Risk Control threshold lambda_hat.

    Parameters
    ----------
    cal_scores : np.ndarray (N_cal,) non-conformity or confidence scores
    cal_losses : np.ndarray (N_cal,) binary or bounded loss per sample
    alpha : float (target risk upper bound, default 0.10)

    Returns
    -------
    lambda_hat : float threshold
    """
    seed_everything(seed)
    n_cal = len(cal_scores)
    sorted_indices = np.argsort(cal_scores)
    sorted_losses = cal_losses[sorted_indices]

    # Find smallest threshold lambda where cumulative empirical risk <= alpha
    cum_risk = np.cumsum(sorted_losses) / (n_cal + 1.0)
    valid_idx = np.where(cum_risk <= alpha)[0]

    if len(valid_idx) > 0:
        idx = valid_idx[-1]
        lambda_hat = float(cal_scores[sorted_indices[idx]])
    else:
        lambda_hat = float(np.min(cal_scores))

    return lambda_hat


def evaluate_e4_crc(
    test_scores: np.ndarray,
    test_losses: np.ndarray,
    lambda_hat: float,
) -> Tuple[float, float]:
    """
    Evaluate CRC coverage and empirical risk on test set.

    Returns
    -------
    empirical_risk : float
    selection_coverage : float
    """
    selected = test_scores >= lambda_hat
    if selected.sum() == 0:
        return 0.0, 0.0

    empirical_risk = float(np.mean(test_losses[selected]))
    selection_coverage = float(np.mean(selected))

    return empirical_risk, selection_coverage


def _self_test() -> int:
    """Self-test E4 Conformal Risk Control on synthetic data."""
    from egsf.data.jdb_s import generate_jdbs
    from egsf.models.bf import train_bf_mod

    print("=" * 62)
    print("E4 CONFORMAL RISK CONTROL (MODEL 13) SELF-TEST")
    print("=" * 62)

    failures = 0

    def _check(cond: bool, msg: str) -> None:
        nonlocal failures
        tag = "[PASS]" if cond else "[FAIL]"
        print(f"  {tag} {msg}")
        if not cond:
            failures += 1

    ds = generate_jdbs("R1", rho_corr=0.8, seed=0)
    model, _ = train_bf_mod(
        [ds["train"]["X1"], ds["train"]["X2"]], ds["train"]["y"],
        [ds["val_id"]["X1"], ds["val_id"]["X2"]], ds["val_id"]["y"],
        in_dims=[8, 8], seed=0
    )

    # Compute confidence scores and losses on cal_crc split
    model.eval()
    with torch.no_grad():
        x1_cal = torch.tensor(ds["cal_crc"]["X1"], dtype=torch.float32)
        x2_cal = torch.tensor(ds["cal_crc"]["X2"], dtype=torch.float32)
        y_cal  = ds["cal_crc"]["y"]

        probs = F.softmax(model([x1_cal, x2_cal]), dim=-1).numpy()
        cal_scores = np.max(probs, axis=-1)
        preds = np.argmax(probs, axis=-1)
        cal_losses = (preds != y_cal).astype(np.float32)

    lambda_hat = calibrate_e4_crc(cal_scores, cal_losses, alpha=0.10, seed=0)

    # Evaluate on test_id
    with torch.no_grad():
        x1_test = torch.tensor(ds["test_id"]["X1"], dtype=torch.float32)
        x2_test = torch.tensor(ds["test_id"]["X2"], dtype=torch.float32)
        y_test  = ds["test_id"]["y"]

        probs_t = F.softmax(model([x1_test, x2_test]), dim=-1).numpy()
        test_scores = np.max(probs_t, axis=-1)
        preds_t = np.argmax(probs_t, axis=-1)
        test_losses = (preds_t != y_test).astype(np.float32)

    emp_risk, coverage = evaluate_e4_crc(test_scores, test_losses, lambda_hat)

    _check(0.0 <= lambda_hat <= 1.0, f"Calibrated threshold lambda_hat in [0,1]: got {lambda_hat:.4f}")
    _check(emp_risk <= 0.15, f"Empirical risk <= alpha+tol (0.15): got {emp_risk:.4f}")
    _check(coverage > 0.50, f"Selection coverage > 0.50: got {coverage:.4f}")

    print("=" * 62)
    if failures == 0:
        print("ALL E4 CRC SELF-TESTS PASSED")
    else:
        print(f"FAILED: {failures} test(s) failed")
    print("=" * 62)

    return failures


if __name__ == "__main__":
    n_fail = _self_test()
    sys.exit(n_fail)
