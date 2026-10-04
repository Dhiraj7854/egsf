"""
egsf/data/jdb_s.py
──────────────────
JDB-S: JointDB Synthetic Benchmark  —  EGSF v8.0, Step 1.

Structural Causal Model
-----------------------
y  ~ Uniform({0,...,K-1})

Causal modality (invariant across environments):
  X1 = μ_y + σ1 ε,    ε ~ N(0, I_d)

Cue variable (spurious, changes across environments):
  In train envs:       c = y  w.p. ρ,  else Uniform({0,...,K-1}∖{y})
  In cue-broken env:   c ~ Uniform({0,...,K-1})   (independent of y)

Cue modality:
  X2 = ν_c + σ2 ε,    ε ~ N(0, I_d)

Regimes
-------
R1  causal-dominant    σ1 small, σ2 large
R2  shortcut-dominant  σ1 large, σ2 small
R3  redundant          σ1=σ2, ρ forced to 1 (c=y always in train)
R4  synergy            XOR prototype structure: y only recoverable from X1+X2 jointly
R5  mixed per-instance per-instance i.i.d. draw of R1 or R2 parameters

Ground truth B*_m
-----------------
Computed via Monte-Carlo from known Gaussians:
  b_m,e = I(X_m;y|env=e) / Σ_k I(X_k;y|env=e)
  B*_m  = min_e  b_m,e

In the cue-broken environment, I(X2;y) → 0, so B*_cue → 0 in every regime.

Split layout
------------
  train, val_id          — training-environment distribution
  conflict_dev           — cue-broken env (budget/anchor tuning)
  cal_g                  — cue-broken env (calibration maps)
  cal_crc                — fresh in-distribution ID env (CRC risk calibration)
  test_id, test_conflict — held-out evaluation
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Make project root importable when run as __main__
sys.path.insert(0, str(Path(__file__).parents[2]))


# ─── Regime parameter table ────────────────────────────────────────────────────
#   sigma_causal : noise std on causal modality X1
#   sigma_cue    : noise std on cue modality X2
#   force_rho    : override ρ to 1.0 in train envs (R3 redundancy)
#   synergy      : use XOR prototype structure (R4)
#   mixed        : per-instance regime selection (R5)

REGIME_PARAMS: Dict[str, Dict] = {
    "R1": dict(sigma_causal=0.30, sigma_cue=2.00, force_rho=False, synergy=False, mixed=False),
    "R2": dict(sigma_causal=5.00, sigma_cue=0.30, force_rho=False, synergy=False, mixed=False),
    "R3": dict(sigma_causal=0.50, sigma_cue=0.50, force_rho=True,  synergy=False, mixed=False),
    "R4": dict(sigma_causal=0.50, sigma_cue=0.50, force_rho=True,  synergy=True,  mixed=False),
    "R5": dict(sigma_causal=None, sigma_cue=None,  force_rho=False, synergy=False, mixed=True),
}

# R5 mixes these two base regimes per instance
_R5_SUB = ["R1", "R2"]


# ─── Prototype construction ────────────────────────────────────────────────────

def _make_prototypes(
    K: int,
    d: int,
    rng: np.random.Generator,
    synergy: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build class prototype matrices for causal (X1) and cue (X2) modalities.

    Synergy (R4) — requires K=4:
      b1 = y // 2 ∈ {0,1}  —  encoded by X1  (proto_causal[0]==proto_causal[1], [2]==[3])
      b2 = y  % 2 ∈ {0,1}  —  encoded by X2  (proto_cue[0]==proto_cue[2], [1]==[3])
      ⟹  X1 alone → b1 only,  X2 alone → b2 only,  joint → y

    Returns
    -------
    proto_causal : (K, d)
    proto_cue    : (K, d)
    """
    if synergy:
        assert K == 4, "Synergy (R4) requires K=4"
        p1 = rng.normal(0.0, 2.0, (2, d))   # for b1 ∈ {0,1}
        p2 = rng.normal(0.0, 2.0, (2, d))   # for b2 ∈ {0,1}
        proto_causal = np.array([p1[y // 2] for y in range(K)])  # (K, d)
        proto_cue    = np.array([p2[y  % 2] for y in range(K)])  # (K, d)
    else:
        proto_causal = rng.normal(0.0, 2.0, (K, d))
        proto_cue    = rng.normal(0.0, 2.0, (K, d))

    return proto_causal, proto_cue


# ─── Sampling helpers ──────────────────────────────────────────────────────────

def _sample_cue(
    y: np.ndarray,
    K: int,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    c[i] = y[i] w.p. rho, else Uniform({0,...,K-1} \\ {y[i]}).
    Vectorised via shift-trick: sample from {0,...,K-2} then skip y.
    """
    c = y.copy()
    flip = rng.random(len(y)) > rho
    n_flip = int(flip.sum())
    if n_flip > 0:
        alt = rng.integers(0, K - 1, size=n_flip)
        alt = np.where(alt >= y[flip], alt + 1, alt)
        c[flip] = alt
    return c


def _sample_features(
    proto: np.ndarray,
    labels: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """X ~ N(proto[labels], σ² I).  Returns (N, d)."""
    means = proto[labels]
    return means + sigma * rng.standard_normal(means.shape)


# ─── Ground-truth mutual information via MC ────────────────────────────────────

def _gaussian_MI(
    proto: np.ndarray,
    sigma: float,
    K: int,
    n_mc: int = 8_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Monte-Carlo estimate of I(X; y) for X | y=k ~ N(proto[k], σ² I).

    I(X; y) = H(y) − E_x[ H(y|x) ]

    Posterior p(y=k|x) is computed exactly from the known Gaussians.
    Works correctly even when proto rows are duplicated (R4 synergy case).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    d = proto.shape[1]
    y_mc = rng.integers(0, K, size=n_mc)
    X_mc = proto[y_mc] + sigma * rng.standard_normal((n_mc, d))

    # log p(x | y=k) — constant -d/2 log(2πσ²) cancels in posterior
    diffs    = X_mc[:, None, :] - proto[None, :, :]   # (n_mc, K, d)
    sq_dists = np.sum(diffs ** 2, axis=2)              # (n_mc, K)
    log_liks = -0.5 * sq_dists / (sigma ** 2)          # (n_mc, K)

    # Numerically stable softmax → posterior
    log_liks -= log_liks.max(axis=1, keepdims=True)
    post = np.exp(log_liks)
    post /= post.sum(axis=1, keepdims=True)            # (n_mc, K)

    H_cond = -np.sum(post * np.log(post + 1e-30), axis=1)   # (n_mc,)
    MI = float(np.log(K)) - float(np.mean(H_cond))
    return max(0.0, MI)


# ─── Main generator ────────────────────────────────────────────────────────────

def generate_jdbs(
    regime: str,
    rho_corr: float,
    K: int = 4,
    d: int = 8,
    n_per_env: Optional[Dict[str, int]] = None,
    n_envs: int = 3,
    seed: int = 0,
) -> Dict:
    """
    Generate a complete JDB-S dataset for one (regime, rho_corr, seed) triple.

    Parameters
    ----------
    regime   : one of {'R1','R2','R3','R4','R5'}
    rho_corr : shortcut correlation in train envs (0–1); overridden to 1.0 for R3/R4
    K        : number of classes (must be 4 for R4)
    d        : feature dimension per modality
    n_per_env: override default split sizes
    n_envs   : number of independent training environments
    seed     : base random seed

    Returns
    -------
    dict with keys:
      'train', 'val_id', 'conflict_dev', 'cal_g', 'cal_crc', 'test_id',
      'test_conflict'   — each is a sub-dict with arrays
      'ground_truth'    — true B*_m and generating parameters
      'metadata'        — provenance
    """
    assert regime in REGIME_PARAMS, f"Unknown regime '{regime}'"
    assert K == 4, "Current implementation fixes K=4"

    if n_per_env is None:
        n_per_env = dict(
            train=1000, val_id=200, conflict_dev=200,
            cal_g=200, cal_crc=200, test_id=500, test_conflict=500,
        )

    params = REGIME_PARAMS[regime]
    rng    = np.random.default_rng(seed)

    # ── Class prototypes (fixed; same across all environments) ─────────────
    proto_causal, proto_cue = _make_prototypes(K, d, rng, synergy=params["synergy"])

    # ── Effective rho for training environments ────────────────────────────
    rho_train: float = 1.0 if params["force_rho"] else rho_corr

    # ── Sigma (None for R5; handled per-instance inside _gen_split) ────────
    sigma_causal: Optional[float] = params["sigma_causal"]
    sigma_cue:    Optional[float] = params["sigma_cue"]

    # ── Ground-truth I(X_m ; y) and B*_m ──────────────────────────────────
    if not params["mixed"]:
        assert sigma_causal is not None and sigma_cue is not None
        MI_c_train = _gaussian_MI(proto_causal, sigma_causal, K, rng=rng)
        MI_q_full  = _gaussian_MI(proto_cue,    sigma_cue,    K, rng=rng)
        # Scale cue MI by correlation strength (rho=1 → full MI; rho=0 → zero MI)
        MI_q_train = rho_train * MI_q_full

        # Cue-broken environment: c ⊥ y  →  I(X2; y) = 0;  I(X1; y) unchanged
        MI_c_broken = MI_c_train
        MI_q_broken = 0.0

        def _bfrac(mi_c: float, mi_q: float) -> Tuple[float, float]:
            tot = mi_c + mi_q + 1e-9
            return mi_c / tot, mi_q / tot

        b_c_train,  b_q_train  = _bfrac(MI_c_train,  MI_q_train)
        b_c_broken, b_q_broken = _bfrac(MI_c_broken, MI_q_broken)

        # B*_m = min over environments
        B_star_causal = float(min(b_c_train, b_c_broken))
        B_star_cue    = float(min(b_q_train, b_q_broken))   # → 0 via broken env

        MI_summary = dict(
            MI_causal_train=float(MI_c_train),
            MI_cue_train_full=float(MI_q_full),
            MI_cue_train_eff=float(MI_q_train),
        )
    else:
        # R5: average over the two base regimes
        s_c = {r: REGIME_PARAMS[r]["sigma_causal"] for r in _R5_SUB}
        s_q = {r: REGIME_PARAMS[r]["sigma_cue"]    for r in _R5_SUB}
        MI_c_parts = [_gaussian_MI(proto_causal, s_c[r], K, rng=rng) for r in _R5_SUB]
        MI_q_parts = [_gaussian_MI(proto_cue, s_q[r], K, rng=rng) * rho_corr for r in _R5_SUB]
        MI_c_train = float(np.mean(MI_c_parts))
        MI_q_train = float(np.mean(MI_q_parts))
        tot = MI_c_train + MI_q_train + 1e-9
        B_star_causal = MI_c_train / tot
        B_star_cue    = 0.0          # broken env always drives cue budget to 0
        MI_summary = dict(
            MI_causal_train=MI_c_train,
            MI_cue_train_eff=MI_q_train,
        )

    B_star = np.array([B_star_causal, B_star_cue])

    # ── Inner split generator ──────────────────────────────────────────────
    def _gen_split(n: int, cue_broken: bool, env_idx: int) -> Dict:
        y_arr = rng.integers(0, K, size=n)

        if cue_broken:
            c_arr = rng.integers(0, K, size=n)        # independent of y
        else:
            c_arr = _sample_cue(y_arr, K, rho_train, rng)

        conflict = (c_arr != y_arr).astype(np.int8)
        env_arr  = np.full(n, env_idx, dtype=np.int32)

        if params["mixed"]:
            # Vectorised: per-instance R1/R2 selection
            choice = rng.integers(0, 2, size=n).astype(np.int32)
            sc_arr = np.where(choice == 0,
                              REGIME_PARAMS["R1"]["sigma_causal"],
                              REGIME_PARAMS["R2"]["sigma_causal"]).astype(float)
            sq_arr = np.where(choice == 0,
                              REGIME_PARAMS["R1"]["sigma_cue"],
                              REGIME_PARAMS["R2"]["sigma_cue"]).astype(float)
            eps1 = rng.standard_normal((n, d))
            eps2 = rng.standard_normal((n, d))
            X1 = proto_causal[y_arr] + sc_arr[:, None] * eps1
            X2 = proto_cue[c_arr]    + sq_arr[:, None] * eps2
            planted = choice
        else:
            assert sigma_causal is not None and sigma_cue is not None
            X1 = _sample_features(proto_causal, y_arr, sigma_causal, rng)
            X2 = _sample_features(proto_cue,    c_arr, sigma_cue,    rng)
            planted = np.zeros(n, dtype=np.int32)

        return dict(
            X1=X1,
            X2=X2,
            y=y_arr,
            c=c_arr,
            env=env_arr,
            conflict=conflict,
            planted_mechanism=planted,
        )

    # ── Generate all 7 splits ──────────────────────────────────────────────
    # Train: pool n_envs independent training environments
    n_per_train = n_per_env["train"] // n_envs
    train_parts = [_gen_split(n_per_train, cue_broken=False, env_idx=e)
                   for e in range(n_envs)]
    splits: Dict[str, Dict] = {
        "train": {k: np.concatenate([p[k] for p in train_parts], axis=0)
                  for k in train_parts[0]}
    }
    splits["val_id"]        = _gen_split(n_per_env["val_id"],        False,  0)
    splits["conflict_dev"]  = _gen_split(n_per_env["conflict_dev"],  True,  -1)
    splits["cal_g"]         = _gen_split(n_per_env["cal_g"],         True,  -1)
    splits["cal_crc"]       = _gen_split(n_per_env["cal_crc"],       False,  0)
    splits["test_id"]       = _gen_split(n_per_env["test_id"],       False,  0)
    splits["test_conflict"] = _gen_split(n_per_env["test_conflict"], True,  -1)

    ground_truth = dict(
        B_star=B_star,
        regime=regime,
        rho_corr=rho_corr,
        rho_train=rho_train,
        sigma_causal=params["sigma_causal"],
        sigma_cue=params["sigma_cue"],
        proto_causal=proto_causal,
        proto_cue=proto_cue,
        synergy=params["synergy"],
        K=K, d=d,
        **MI_summary,
    )
    metadata = dict(
        regime=regime, rho_corr=rho_corr, seed=seed,
        K=K, d=d, n_envs=n_envs,
        splits=sorted(splits.keys()),
    )

    return {**splits, "ground_truth": ground_truth, "metadata": metadata}


# ─── Self-test ─────────────────────────────────────────────────────────────────

def _self_test() -> int:
    """
    Run all JDB-S self-tests. Returns number of failures.
    Tests:
      1  All 5 regimes generate without error
      2  All 4 rho_corr levels generate without error
      3  All 7 split keys present
      4  Cue correlation matches rho_corr in train (within tolerance)
      5  Cue is near-chance (1/K) in cue-broken splits
      6  B*_m shape (2,), values in [0,1], B*_cue ≈ 0 for all regimes
      7  Planted reliance labels valid (R5: {0,1}; others: all 0)
      8  PID structural checks via linear classifier
         8a R1: acc(X1) >> acc(X2) in train
         8b R2: acc(X2) >> acc(X1) in train
         8c R3: acc(X1) ≈ acc(X2) in train (redundancy)
         8d R4: acc(X1+X2) >> max(acc(X1), acc(X2)) in train (synergy)
         8e R2 cue collapse in cue-broken env
         8f Causal predictor stable across envs
    """
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegression

    failures = 0

    def _check(cond: bool, msg: str) -> None:
        nonlocal failures
        tag = "[PASS]" if cond else "[FAIL]"
        print(f"  {tag} {msg}")
        if not cond:
            failures += 1

    def _lin_acc(X_tr: np.ndarray, y_tr: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Train on X_tr/y_tr, evaluate on X_val/y_val."""
        lr = LogisticRegression(max_iter=500, C=1.0, random_state=0)
        lr.fit(X_tr, y_tr)
        return float(lr.score(X_val, y_val))

    print("=" * 62)
    print("JDB-S SELF-TEST")
    print("=" * 62)

    # ── Test 1: All regimes generate ─────────────────────────────────────
    print("\n[Test 1] All 5 regimes generate without error:")
    for reg in ["R1", "R2", "R3", "R4", "R5"]:
        try:
            generate_jdbs(reg, rho_corr=0.8, seed=42)
            _check(True, f"Regime {reg} generated OK")
        except Exception as exc:
            _check(False, f"Regime {reg} raised: {exc}")

    # ── Test 2: All rho_corr levels ──────────────────────────────────────
    print("\n[Test 2] All rho_corr levels generate without error:")
    for rho in [0.6, 0.7, 0.8, 0.9]:
        try:
            generate_jdbs("R2", rho_corr=rho, seed=7)
            _check(True, f"rho_corr={rho} OK")
        except Exception as exc:
            _check(False, f"rho_corr={rho} raised: {exc}")

    # ── Test 3: All 7 split keys present ─────────────────────────────────
    print("\n[Test 3] All 7 split keys present:")
    ds = generate_jdbs("R1", rho_corr=0.8, seed=0)
    required = ["train", "val_id", "conflict_dev",
                "cal_g", "cal_crc", "test_id", "test_conflict"]
    for sp in required:
        _check(sp in ds, f"Split '{sp}' present")

    # ── Test 4: Cue correlation matches rho_corr ──────────────────────────
    print("\n[Test 4] Cue correlation matches rho_corr in train (tol=0.06):")
    for rho in [0.6, 0.8, 0.9]:
        ds = generate_jdbs("R2", rho_corr=rho, seed=1)
        obs = float(np.mean(ds["train"]["c"] == ds["train"]["y"]))
        _check(abs(obs - rho) < 0.06,
               f"rho={rho}: observed={obs:.3f}")

    # ── Test 5: Cue near-chance in cue-broken splits ──────────────────────
    print("\n[Test 5] Cue P(c=y) ~= 1/K=0.25 in cue-broken splits (tol=0.08):")
    ds = generate_jdbs("R2", rho_corr=0.9, seed=2)
    expected = 1.0 / 4
    for sp in ["conflict_dev", "cal_g", "test_conflict"]:
        obs = float(np.mean(ds[sp]["c"] == ds[sp]["y"]))
        _check(abs(obs - expected) < 0.08,
               f"'{sp}': P(c=y)={obs:.3f}, expected~={expected:.3f}")

    # ── Test 6: B*_m shape, bounds, B*_cue ≈ 0 ───────────────────────────
    print("\n[Test 6] B*_m shape=(2,), values in [0,1], B*_cue<0.05 (all regimes):")
    for reg in ["R1", "R2", "R3", "R4", "R5"]:
        ds = generate_jdbs(reg, rho_corr=0.8, seed=3)
        B  = ds["ground_truth"]["B_star"]
        _check(B.shape == (2,), f"{reg}: shape==(2,) got {B.shape}")
        _check(0.0 <= B[0] <= 1.0 and 0.0 <= B[1] <= 1.0,
               f"{reg}: values in [0,1] -- B=[{B[0]:.4f}, {B[1]:.4f}]")
        _check(B[1] < 0.05,
               f"{reg}: B*_cue<0.05 -- got {B[1]:.4f}")

    # ── Test 7: Planted reliance labels ───────────────────────────────────
    print("\n[Test 7] Planted reliance labels valid:")
    ds5 = generate_jdbs("R5", rho_corr=0.8, seed=4)
    pm5 = ds5["train"]["planted_mechanism"]
    _check(set(np.unique(pm5)).issubset({0, 1}),
           f"R5 planted_mechanism subset of {{0,1}}: unique={np.unique(pm5)}")
    both = (0 in pm5) and (1 in pm5)
    _check(both, f"R5 planted_mechanism contains both 0 and 1")

    ds1 = generate_jdbs("R1", rho_corr=0.8, seed=5)
    pm1 = ds1["train"]["planted_mechanism"]
    _check(np.all(pm1 == 0), f"R1 planted_mechanism all zeros: unique={np.unique(pm1)}")

    # ── Test 8: PID structural checks ─────────────────────────────────────
    print("\n[Test 8] PID structural checks (logistic regression, train->val):")

    # 8a R1: causal >> cue (val generalisation)
    ds = generate_jdbs("R1", rho_corr=0.8, seed=6)
    ac = _lin_acc(ds["train"]["X1"], ds["train"]["y"], ds["val_id"]["X1"], ds["val_id"]["y"])
    aq = _lin_acc(ds["train"]["X2"], ds["train"]["y"], ds["val_id"]["X2"], ds["val_id"]["y"])
    _check(ac > aq + 0.10,
           f"8a R1: val acc(X1)={ac:.3f} >> acc(X2)={aq:.3f} (+0.10 margin)")

    # 8b R2: cue >> causal (val generalisation)
    ds = generate_jdbs("R2", rho_corr=0.8, seed=6)
    ac = _lin_acc(ds["train"]["X1"], ds["train"]["y"], ds["val_id"]["X1"], ds["val_id"]["y"])
    aq = _lin_acc(ds["train"]["X2"], ds["train"]["y"], ds["val_id"]["X2"], ds["val_id"]["y"])
    _check(aq > ac + 0.10,
           f"8b R2: val acc(X2)={aq:.3f} >> acc(X1)={ac:.3f} (+0.10 margin)")

    # 8c R3: causal ~= cue in val (redundancy)
    ds = generate_jdbs("R3", rho_corr=0.8, seed=6)
    ac = _lin_acc(ds["train"]["X1"], ds["train"]["y"], ds["val_id"]["X1"], ds["val_id"]["y"])
    aq = _lin_acc(ds["train"]["X2"], ds["train"]["y"], ds["val_id"]["X2"], ds["val_id"]["y"])
    _check(abs(ac - aq) < 0.15,
           f"8c R3: val acc(X1)={ac:.3f} ~= acc(X2)={aq:.3f} (|delta|<0.15)")

    # 8d R4: joint >> max(X1, X2) in val (synergy)
    ds = generate_jdbs("R4", rho_corr=0.8, seed=6)
    ac = _lin_acc(ds["train"]["X1"], ds["train"]["y"], ds["val_id"]["X1"], ds["val_id"]["y"])
    aq = _lin_acc(ds["train"]["X2"], ds["train"]["y"], ds["val_id"]["X2"], ds["val_id"]["y"])
    X_tr_j  = np.concatenate([ds["train"]["X1"], ds["train"]["X2"]], axis=1)
    X_val_j = np.concatenate([ds["val_id"]["X1"], ds["val_id"]["X2"]], axis=1)
    aj = _lin_acc(X_tr_j, ds["train"]["y"], X_val_j, ds["val_id"]["y"])
    _check(aj > max(ac, aq) + 0.15,
           f"8d R4: val joint={aj:.3f} >> max(X1={ac:.3f}, X2={aq:.3f}) (+0.15 margin)")

    # 8e R2 cue collapse in cue-broken env
    ds = generate_jdbs("R2", rho_corr=0.9, seed=6)
    lr_q = LogisticRegression(max_iter=500, C=1.0, random_state=0)
    lr_q.fit(ds["train"]["X2"], ds["train"]["y"])
    acc_conf = float(lr_q.score(ds["test_conflict"]["X2"], ds["test_conflict"]["y"]))
    _check(acc_conf < 0.50,
           f"8e R2 cue collapse in conflict env: acc={acc_conf:.3f} < 0.50")

    # 8f Causal predictor stable across envs
    lr_c = LogisticRegression(max_iter=500, C=1.0, random_state=0)
    lr_c.fit(ds["train"]["X1"], ds["train"]["y"])
    acc_id   = float(lr_c.score(ds["test_id"]["X1"],       ds["test_id"]["y"]))
    acc_conf = float(lr_c.score(ds["test_conflict"]["X1"], ds["test_conflict"]["y"]))
    _check(abs(acc_id - acc_conf) < 0.15,
           f"8f Causal stable: test_id={acc_id:.3f}, conflict={acc_conf:.3f} (|delta|<0.15)")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    if failures == 0:
        print("ALL SELF-TESTS PASSED")
    else:
        print(f"FAILED: {failures} test(s) failed")
    print("=" * 62)
    return failures


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    n_fail = _self_test()
    sys.exit(n_fail)
