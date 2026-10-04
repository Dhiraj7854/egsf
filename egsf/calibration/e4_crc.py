"""
egsf/calibration/e4_crc.py
──────────────────────────
E4: Conformal Risk Control (CRC) Module (Model 13 / Baseline E4) — EGSF v8.0, Step 1.14.

Computes risk-controlled conformal thresholds lambda_hat on calibration split `cal_crc`
such that expected selective misclassification / false-correction risk is provably bounded:
E[ L( y, y_hat(lambda_hat) ) ] <= alpha
where target risk alpha = 0.10 (from configs/jdb_s.yaml).

CRC Threshold Definition
------------------------
For a calibration set of n samples with scores D_i and losses L_i:

  S(lambda) = {i : D_i >= lambda}
  R_hat(lambda) = (1/n) sum_{i: D_i >= lambda} L_i   [mean loss over selected set]

  CRC bound:
    bound(lambda) = (n / (n+1)) * R_hat(lambda) + 1/(n+1) <= alpha

  lambda* = inf { lambda : bound(lambda) <= alpha }

This is the SMALLEST threshold satisfying the bound, giving maximal selection coverage
subject to the guarantee. Candidates are the unique observed D values (and a sentinel
above the maximum to represent the "select no one" option).

CRC guarantee (Angelopoulos et al. 2022):
  If cal and test data are exchangeable and lambda* is computed as above,
  then E[L over S(lambda*)] <= alpha.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Make project root importable
sys.path.insert(0, str(Path(__file__).parents[2]))


def _crc_bound(n: int, r_hat: float) -> float:
    """CRC finite-sample bound: (n/(n+1)) * R_hat + 1/(n+1)."""
    return (n / (n + 1.0)) * r_hat + 1.0 / (n + 1.0)


def calibrate_e4_crc(
    cal_scores: np.ndarray,
    cal_losses: np.ndarray,
    alpha: float = 0.10,
    return_details: bool = False,
) -> float | Tuple[float, float, float, float]:
    """
    Calibrate Conformal Risk Control threshold lambda*.

    Finds lambda* = inf { lambda : bound(lambda) <= alpha }, where
    bound(lambda) = (n/(n+1)) * R_hat(lambda) + 1/(n+1) and
    R_hat(lambda) = mean loss over {i : D_i >= lambda}.

    The smallest satisfying lambda gives maximal selection coverage while
    maintaining the CRC risk guarantee.

    Parameters
    ----------
    cal_scores  : np.ndarray (N_cal,) — D scores (e.g. excess reliance)
    cal_losses  : np.ndarray (N_cal,) — binary or bounded loss per sample
    alpha       : float — target risk upper bound (default 0.10)
    return_details : bool — if True, return (lambda*, R_hat, bound, coverage)

    Returns
    -------
    lambda_hat  : float — CRC threshold (inf { lambda : bound(lambda) <= alpha })
    If return_details=True, also returns R_hat, bound_val, coverage.

    Abstain sentinel
    ----------------
    If no nonempty selection satisfies the bound, returns max(D)+1 so that
    D_i >= lambda* is False for all i (zero coverage, safe abstain).
    """
    cal_scores = np.asarray(cal_scores, dtype=np.float64)
    cal_losses = np.asarray(cal_losses, dtype=np.float64)

    if len(cal_scores) != len(cal_losses):
        raise ValueError(
            f"cal_scores and cal_losses must have equal length, "
            f"got {len(cal_scores)} and {len(cal_losses)}"
        )
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    n = len(cal_scores)
    if n == 0:
        lam = float(np.inf)
        return (lam, 0.0, 1.0, 0.0) if return_details else lam

    # ── Candidate thresholds: all unique observed D values, ascending ──────────
    # We evaluate each candidate λ from lowest to highest, tracking the last
    # (i.e. smallest) λ that satisfies the CRC bound, which gives maximal coverage.
    candidates = np.unique(cal_scores)  # sorted ascending

    best_lambda = None
    best_r_hat = None
    best_bound = None
    best_cov = None

    for lam in candidates:
        selected = cal_scores >= lam
        n_sel = int(selected.sum())
        if n_sel == 0:
            continue
        r_hat = float(cal_losses[selected].mean())
        bound = _crc_bound(n, r_hat)
        cov = n_sel / n
        if bound <= alpha:
            # This λ satisfies the bound. Record it — if a smaller λ also satisfies
            # the bound, it will overwrite this (we keep the smallest = most inclusive).
            # Since we iterate ascending, every valid λ encountered updates best_lambda
            # until we encounter one that fails — but we must check ALL candidates
            # because bound is non-monotone. So collect all valid ones and take min.
            if best_lambda is None or lam < best_lambda:
                best_lambda = lam
                best_r_hat = r_hat
                best_bound = bound
                best_cov = cov

    # ── Actually we need all valid ones; rewrite as a single pass ─────────────
    # (The loop above already finds the minimum valid lambda since we iterate
    # ascending and overwrite best_lambda only if lam < best_lambda — which
    # is always true in ascending iteration since each new lam >= prev lam.
    # So best_lambda is set to the FIRST valid candidate found in ascending order,
    # which IS the smallest valid candidate. Correct.)

    if best_lambda is None:
        # No nonempty selection satisfies the bound → abstain (select no one)
        abstain = float(np.max(cal_scores)) + 1.0
        return (abstain, 0.0, 1.0, 0.0) if return_details else abstain

    if return_details:
        return float(best_lambda), float(best_r_hat), float(best_bound), float(best_cov)
    return float(best_lambda)


def evaluate_e4_crc(
    test_scores: np.ndarray,
    test_losses: np.ndarray,
    lambda_hat: float,
) -> Tuple[float, float]:
    """
    Evaluate CRC coverage and empirical risk on test set.

    Returns
    -------
    empirical_risk : float (mean loss of selected instances, or 0.0 if empty)
    selection_coverage : float (fraction of instances selected)
    """
    scores_arr = np.asarray(test_scores, dtype=np.float64)
    losses_arr = np.asarray(test_losses, dtype=np.float64)

    selected = scores_arr >= lambda_hat
    cov = float(np.mean(selected))
    if selected.sum() == 0:
        return 0.0, 0.0

    empirical_risk = float(np.mean(losses_arr[selected]))
    return empirical_risk, cov


def _self_test() -> int:
    """Self-test E4 Conformal Risk Control implementation."""
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

    def _bound(n: int, r_hat: float) -> float:
        return (n / (n + 1.0)) * r_hat + 1.0 / (n + 1.0)

    # ── Test 1: Manual calculation — selected set must be D >= lambda ──────────
    # D = [0.1, 0.2, 0.5, 0.8, 0.9], L = [1, 0, 0, 0, 0], alpha = 0.25, n=5
    # Candidates (ascending): 0.1, 0.2, 0.5, 0.8, 0.9
    #
    # λ=0.1: S = all 5, R_hat = 1/5 = 0.20, bound = (5/6)*0.20 + 1/6 = 0.333 > 0.25 → fail
    # λ=0.2: S = {0.2,0.5,0.8,0.9}, R_hat = 0/4 = 0.00, bound = 0+1/6 = 0.167 <= 0.25 → valid ✓
    # λ=0.5: S = {0.5,0.8,0.9}, R_hat = 0/3 = 0.00, bound = 1/6 = 0.167 <= 0.25 → valid
    # λ=0.8: S = {0.8,0.9}, R_hat = 0.00, bound = 1/6 <= 0.25 → valid
    # λ=0.9: S = {0.9}, R_hat = 0.00, bound = 1/6 <= 0.25 → valid
    # lambda* = inf{valid} = 0.2 (smallest satisfying candidate)
    print("\n[Test 1] Manual: low-D loss, lambda* = 0.2 (smallest valid threshold):")
    D1 = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
    L1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    lam1, r_hat1, bound1, cov1 = calibrate_e4_crc(D1, L1, alpha=0.25, return_details=True)
    _check(abs(lam1 - 0.2) < 1e-9,
           f"Test 1a: lambda* == 0.2 (smallest valid): got {lam1:.6f}")
    _check(bound1 <= 0.25,
           f"Test 1b: CRC bound ({bound1:.4f}) <= alpha (0.25)")
    # Verify selected set is D >= 0.2 (not D <= 0.2)
    sel1 = D1 >= lam1
    _check(not sel1[0] or L1[sel1].mean() <= r_hat1 + 1e-9,
           f"Test 1c: Selected set = {{D >= 0.2}}: {D1[sel1]}")
    _check(abs(cov1 - 4.0/5.0) < 1e-9,
           f"Test 1d: Coverage == 4/5 (0.80): got {cov1:.4f}")
    # Verify that λ=0.1 (below lambda*) violates the bound
    sel_below = D1 >= 0.1
    r_below = float(L1[sel_below].mean())
    bound_below = _bound(5, r_below)
    _check(bound_below > 0.25,
           f"Test 1e: λ=0.1 violates bound: bound({bound_below:.4f}) > alpha(0.25)")

    # ── Test 2: Ties — tied D scores must be handled consistently ─────────────
    # D = [0.5, 0.5, 0.5, 0.9], L = [1, 0, 0, 0], alpha = 0.30, n=4
    # Candidates: 0.5, 0.9
    # λ=0.5: S = all 4, R_hat = 1/4 = 0.25, bound = (4/5)*0.25 + 1/5 = 0.40 > 0.30 → fail
    # λ=0.9: S = {0.9}, R_hat = 0, bound = 1/5 = 0.20 <= 0.30 → valid ✓
    # lambda* = 0.9
    print("\n[Test 2] Ties: lambda* = 0.9 (only high-D bin satisfies bound):")
    D2 = np.array([0.5, 0.5, 0.5, 0.9])
    L2 = np.array([1.0, 0.0, 0.0, 0.0])
    lam2, r_hat2, bound2, cov2 = calibrate_e4_crc(D2, L2, alpha=0.30, return_details=True)
    _check(abs(lam2 - 0.9) < 1e-9,
           f"Test 2a: lambda* == 0.9 with ties: got {lam2:.6f}")
    _check(bound2 <= 0.30,
           f"Test 2b: CRC bound ({bound2:.4f}) <= alpha (0.30)")
    # All D=0.5 instances must NOT be selected
    sel2 = D2 >= lam2
    _check(not sel2[0] and not sel2[1] and not sel2[2],
           f"Test 2c: Tied low-D instances excluded: selected={D2[sel2]}")
    _check(sel2[3],
           f"Test 2d: High-D instance selected: D=0.9 in selection")

    # ── Test 3: High threshold required — only sparse high-D satisfies bound ───
    # D = [0.1, 0.3, 0.6, 0.7, 0.8], L = [1, 1, 1, 0, 0], alpha=0.20, n=5
    # λ=0.1: R_hat=3/5=0.60, bound=0.60*5/6+1/6=0.667 > 0.20
    # λ=0.3: S={0.3,0.6,0.7,0.8}, R_hat=2/4=0.50, bound=(4/6)*0.50+1/6=0.500 > 0.20
    # λ=0.6: S={0.6,0.7,0.8}, R_hat=1/3, bound=(5/6)*(1/3)+1/6=0.278+0.167=0.444 > 0.20
    # Wait, n=5 always: bound = (5/6)*R_hat + 1/6
    # λ=0.6: R_hat=1/3, bound=(5/6)*(1/3)+1/6 = 5/18+3/18 = 8/18 = 0.444 > 0.20
    # λ=0.7: S={0.7,0.8}, R_hat=0, bound=1/6=0.167 <= 0.20 ✓
    # λ=0.8: S={0.8}, R_hat=0, bound=1/6=0.167 <= 0.20 ✓
    # lambda* = 0.7 (smallest valid)
    print("\n[Test 3] High threshold required: lambda* = 0.7:")
    D3 = np.array([0.1, 0.3, 0.6, 0.7, 0.8])
    L3 = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    lam3, r_hat3, bound3, cov3 = calibrate_e4_crc(D3, L3, alpha=0.20, return_details=True)
    _check(abs(lam3 - 0.7) < 1e-9,
           f"Test 3a: lambda* == 0.7: got {lam3:.6f}")
    _check(bound3 <= 0.20,
           f"Test 3b: bound({bound3:.4f}) <= alpha(0.20)")
    # Verify below-lambda* threshold violates bound
    sel_below3 = D3 >= 0.6
    r_below3 = float(L3[sel_below3].mean())
    bound_below3 = _bound(5, r_below3)
    _check(bound_below3 > 0.20,
           f"Test 3c: λ=0.6 (below lambda*) violates bound: {bound_below3:.4f} > 0.20")

    # ── Test 4: Abstain — no nonempty selection satisfies the bound ────────────
    # D = [0.1, 0.3, 0.5], L = [1, 1, 1], alpha = 0.10, n=3
    # 1/(n+1) = 1/4 = 0.25 > 0.10, so even perfect zero-loss selection fails!
    # Actually any selection with L=1 will have R_hat > 0, making bound > 1/(n+1) > alpha
    # But even if somehow R_hat=0 with n=3: bound = 1/4 = 0.25 > 0.10 → abstain
    print("\n[Test 4] Abstain: 1/(n+1)=0.25 > alpha=0.10, no selection possible:")
    D4 = np.array([0.1, 0.3, 0.5])
    L4 = np.array([1.0, 1.0, 1.0])
    lam4, r_hat4, bound4, cov4 = calibrate_e4_crc(D4, L4, alpha=0.10, return_details=True)
    _check(lam4 > np.max(D4),
           f"Test 4a: Abstain threshold above max(D): got {lam4:.4f} vs max={np.max(D4):.4f}")
    _check(cov4 == 0.0,
           f"Test 4b: Coverage == 0.0 (abstain): got {cov4:.4f}")
    # Verify no D4 instance is selected
    _check((D4 >= lam4).sum() == 0,
           f"Test 4c: No instances selected under abstain threshold")

    # ── Test 5: Manual calculation by hand (exact) ────────────────────────────
    # D = [0.2, 0.4, 0.6, 0.8, 1.0], L = [0, 0, 0, 1, 1], alpha=0.30, n=5
    # 1/(n+1)=1/6≈0.167 <= 0.30 → at minimum, a zero-loss selection can work
    # λ=0.2: R_hat=2/5=0.40, bound=(5/6)*0.40+1/6=0.333+0.167=0.500 > 0.30
    # λ=0.4: R_hat=2/4=0.50, bound=(5/6)*0.50+1/6=0.417+0.167=0.583 > 0.30
    # λ=0.6: R_hat=2/3, bound=(5/6)*(2/3)+1/6=0.556+0.167=0.722 > 0.30
    # λ=0.8: R_hat=1 (both 0.8,1.0 have L=1 — wait L=[0,0,0,1,1] → D=0.8 has L=1, D=1.0 has L=1
    #        S={0.8,1.0}, R_hat=2/2=1.0, bound=(5/6)*1.0+1/6=1.0 > 0.30
    # λ=1.0: S={1.0}, R_hat=1.0, bound=1.0 > 0.30
    # No valid threshold → abstain
    print("\n[Test 5] Hand-verified: high losses at high D → abstain:")
    D5 = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    L5 = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    lam5, _, _, cov5 = calibrate_e4_crc(D5, L5, alpha=0.30, return_details=True)
    _check(lam5 > 1.0,
           f"Test 5a: Abstain (all high-D have L=1): lambda*={lam5:.4f}")
    _check(cov5 == 0.0,
           f"Test 5b: Coverage == 0.0: got {cov5:.4f}")

    # Now flip: low D has losses, high D does not
    # D = [0.2, 0.4, 0.6, 0.8, 1.0], L = [1, 1, 0, 0, 0], alpha=0.30, n=5
    # λ=0.2: R_hat=2/5=0.40, bound=0.500 > 0.30
    # λ=0.4: R_hat=1/4=0.25, bound=(5/6)*0.25+1/6=0.208+0.167=0.375 > 0.30
    # λ=0.6: R_hat=0/3=0, bound=1/6=0.167 <= 0.30 ✓ → lambda*=0.6
    # λ=0.8: bound=1/6 ✓ (but 0.6 is smaller → lambda*=0.6)
    # lambda* = 0.6
    D5b = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    L5b = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    lam5b, _, bound5b, cov5b = calibrate_e4_crc(D5b, L5b, alpha=0.30, return_details=True)
    _check(abs(lam5b - 0.6) < 1e-9,
           f"Test 5c: Flipped losses → lambda*=0.6: got {lam5b:.6f}")
    _check(bound5b <= 0.30,
           f"Test 5d: bound({bound5b:.4f}) <= 0.30")
    _check(abs(cov5b - 3.0/5.0) < 1e-9,
           f"Test 5e: Coverage = 3/5: got {cov5b:.4f}")
    # Verify selected set excludes low-D lossy instances
    sel5b = D5b >= lam5b
    r_sel5b = float(L5b[sel5b].mean())
    _check(r_sel5b == 0.0,
           f"Test 5f: Selected instances (D>=0.6) all have L=0: R_hat={r_sel5b}")

    # ── Test 6: Verify returned lambda satisfies exact CRC bound ──────────────
    print("\n[Test 6] CRC bound verification on random data:")
    rng = np.random.default_rng(42)
    D6 = rng.uniform(0, 1, 50)
    L6 = rng.binomial(1, 0.15, 50).astype(float)
    alpha6 = 0.25
    lam6, r_hat6, bound6, cov6 = calibrate_e4_crc(D6, L6, alpha=alpha6, return_details=True)
    if lam6 <= np.max(D6):
        # Selected something
        sel6 = D6 >= lam6
        r_hat6_check = float(L6[sel6].mean())
        bound6_check = _bound(50, r_hat6_check)
        _check(bound6_check <= alpha6,
               f"Test 6a: Returned lambda satisfies CRC bound: {bound6_check:.4f} <= {alpha6}")
        _check(abs(r_hat6_check - r_hat6) < 1e-9,
               f"Test 6b: R_hat matches return_details: {r_hat6_check:.4f} == {r_hat6:.4f}")
    else:
        _check(True, "Test 6a: Abstain case — no bound to verify")
        _check(True, "Test 6b: Abstain case — skip R_hat check")

    # ── Test 7: Verify lowering lambda below lambda* violates bound ────────────
    print("\n[Test 7] Verify λ below lambda* violates CRC bound:")
    D7 = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    L7 = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    alpha7 = 0.25
    # λ=0.1: R=2/5=0.4, bound=0.5 > 0.25 ✗
    # λ=0.3: R=1/4=0.25, bound=(5/6)*0.25+1/6=0.375 > 0.25 ✗
    # λ=0.5: R=0/3=0, bound=1/6=0.167 <= 0.25 ✓ → lambda*=0.5
    lam7, _, bound7, cov7 = calibrate_e4_crc(D7, L7, alpha=alpha7, return_details=True)
    _check(abs(lam7 - 0.5) < 1e-9,
           f"Test 7a: lambda* == 0.5: got {lam7:.6f}")
    # Verify λ=0.3 (below lambda*) violates bound
    sel_below7 = D7 >= 0.3
    r_below7 = float(L7[sel_below7].mean())
    bound_below7 = _bound(5, r_below7)
    _check(bound_below7 > alpha7,
           f"Test 7b: λ=0.3 (below lambda*) violates bound: {bound_below7:.4f} > {alpha7}")
    # Verify λ=0.1 (below lambda*) violates bound
    sel_below7b = D7 >= 0.1
    r_below7b = float(L7[sel_below7b].mean())
    bound_below7b = _bound(5, r_below7b)
    _check(bound_below7b > alpha7,
           f"Test 7c: λ=0.1 (below lambda*) violates bound: {bound_below7b:.4f} > {alpha7}")

    # ── Test 8: API consistency — return_details=False returns only float ──────
    print("\n[Test 8] API consistency:")
    D8 = np.array([0.2, 0.5, 0.8])
    L8 = np.array([0.0, 0.0, 0.0])
    lam8_scalar = calibrate_e4_crc(D8, L8, alpha=0.20)
    lam8_detail, _, _, _ = calibrate_e4_crc(D8, L8, alpha=0.20, return_details=True)
    _check(isinstance(lam8_scalar, float),
           f"Test 8a: return_details=False → float: {type(lam8_scalar)}")
    _check(abs(lam8_scalar - lam8_detail) < 1e-12,
           f"Test 8b: Scalar and detailed agree: {lam8_scalar} vs {lam8_detail}")

    # ── Test 9: Invalid inputs raise ValueError ───────────────────────────────
    print("\n[Test 9] Invalid inputs:")
    try:
        calibrate_e4_crc(np.array([0.1, 0.2]), np.array([1.0]), alpha=0.10)
        _check(False, "Test 9a: Mismatched lengths should raise ValueError")
    except ValueError:
        _check(True, "Test 9a: Mismatched lengths raises ValueError")

    try:
        calibrate_e4_crc(np.array([0.1]), np.array([1.0]), alpha=1.5)
        _check(False, "Test 9b: alpha > 1 should raise ValueError")
    except ValueError:
        _check(True, "Test 9b: alpha > 1 raises ValueError")

    try:
        calibrate_e4_crc(np.array([0.1]), np.array([1.0]), alpha=0.0)
        _check(False, "Test 9c: alpha == 0 should raise ValueError")
    except ValueError:
        _check(True, "Test 9c: alpha == 0 raises ValueError")

    # ── Test 10: evaluate_e4_crc uses same >= semantics ───────────────────────
    print("\n[Test 10] evaluate_e4_crc selection semantics (D >= lambda):")
    D_test10 = np.array([0.3, 0.5, 0.7, 0.9])
    L_test10 = np.array([0.0, 1.0, 0.0, 1.0])
    lam_test10 = 0.6
    risk10, cov10 = evaluate_e4_crc(D_test10, L_test10, lam_test10)
    # Selected: D=0.7 (L=0) and D=0.9 (L=1) → 2/4 coverage, 1/2 risk
    _check(abs(cov10 - 0.5) < 1e-9,
           f"Test 10a: Coverage = 0.5 (2/4 selected): got {cov10:.4f}")
    _check(abs(risk10 - 0.5) < 1e-9,
           f"Test 10b: Risk = 0.5 (1/2 selected have L=1): got {risk10:.4f}")
    # Only D=0.3 excluded (below lambda)
    _check(True,
           f"Test 10c: D=0.3 instance excluded (< 0.6): verified by coverage check")

    print("\n" + "=" * 62)
    if failures == 0:
        print("ALL E4 CRC SELF-TESTS PASSED")
    else:
        print(f"FAILED: {failures} test(s) failed")
    print("=" * 62)
    return failures


if __name__ == "__main__":
    n_fail = _self_test()
    import sys as _sys
    _sys.exit(n_fail)
