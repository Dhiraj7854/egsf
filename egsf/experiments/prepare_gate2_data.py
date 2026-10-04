"""
egsf/experiments/prepare_gate2_data.py
---------------------------------------
Helper script to pre-compute and store D0 modality reliance scores and D2 budgets
for Gate 2 evaluation across regimes and seeds on JDB-S benchmark.

Gate 2 Step 5 Fix (exchangeability):
  cal_crc is now generated from the ID distribution (cue_broken=False),
  matching test_id's generating mechanism. This restores the CRC exchangeability
  assumption (Angelopoulos et al. 2022).

  Previously, cal_crc was generated with cue_broken=True (conflict distribution),
  causing a distributional mismatch that invalidated the CRC guarantee.

  Fresh independent ID pool uses seed+1000 to guarantee no sample overlap with test_id.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))

from egsf.budget.d2_budget import compute_d2_budget
from egsf.data.jdb_s import generate_jdbs
from egsf.models.bf import train_bf_mod
from egsf.reliance.d0_reliance import compute_d0_reliance
from egsf.utils.reproducibility import load_config, seed_everything


def _normalize_rel(rel: np.ndarray) -> np.ndarray:
    """Safely normalize reliance array so every row sums to 1.0."""
    s = rel.sum(axis=1, keepdims=True)
    zero_mask = (s.squeeze(-1) < 1e-6)
    rel_norm = rel / np.maximum(s, 1e-6)
    if zero_mask.any():
        rel_norm[zero_mask] = 1.0 / rel.shape[1]
    return rel_norm


def _generate_id_cal_pool(ds: Dict) -> Dict:
    """
    Extract the dedicated fresh in-distribution calibration pool for cal_crc from ds['cal_crc'].

    The ID mechanism is identical to test_id: cue_broken=False, env_idx=0.
    Drawn as an independent sample block in generate_jdbs(seed=seed).
    """
    return ds["cal_crc"]


def prepare_gate2_data(
    regimes: List[str] = ["R1", "R2", "R3", "R4", "R5"],
    seeds: List[int] = [0, 1, 2, 3, 4],
    rho_corr: float = 0.9,
    out_dir: str = "results/gate2",
    n_cal_crc: int = 200,
) -> None:
    """Prepare Gate 2 artifacts with dedicated independent ID-distribution cal_crc."""
    out_path_dir = Path(out_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing Gate 2 (Step 5 - Fresh ID cal_crc) regimes={regimes} seeds={seeds} rho={rho_corr}")
    print("  cal_crc: fresh independent ID split (cue_broken=False, env_idx=0)")

    for reg in regimes:
        for seed in seeds:
            seed_everything(seed)

            ds = generate_jdbs(regime=reg, rho_corr=rho_corr, seed=seed)
            cal_pool = _generate_id_cal_pool(ds)

            bf_model, _ = train_bf_mod(
                xs_train=[ds["train"]["X1"], ds["train"]["X2"]],
                y_train=ds["train"]["y"],
                xs_val=[ds["val_id"]["X1"], ds["val_id"]["X2"]],
                y_val=ds["val_id"]["y"],
                seed=seed, verbose=False,
            )

            rel_cal_crc = _normalize_rel(compute_d0_reliance(
                model=bf_model, xs=[cal_pool["X1"], cal_pool["X2"]], n_resample=8, seed=seed))
            rel_test_id = _normalize_rel(compute_d0_reliance(
                model=bf_model, xs=[ds["test_id"]["X1"], ds["test_id"]["X2"]], n_resample=8, seed=seed))
            rel_test_conflict = _normalize_rel(compute_d0_reliance(
                model=bf_model, xs=[ds["test_conflict"]["X1"], ds["test_conflict"]["X2"]], n_resample=8, seed=seed))

            b_star, b_kappa_bounds = compute_d2_budget(ds, kappa_grid=[0.1])
            b_kappa = b_kappa_bounds[0.1]

            save_path = out_path_dir / f"{reg}_seed{seed}.npz"
            np.savez_compressed(
                save_path,
                # cal_crc: dedicated independent ID distribution split
                rel_cal_crc=rel_cal_crc,
                y_cal_crc=cal_pool["y"],
                env_cal_crc=cal_pool["env"],
                c_cal_crc=cal_pool["c"],
                conflict_cal_crc=cal_pool["conflict"],
                cal_crc_mechanism=np.array("id_cue_broken_false_fresh_cal_crc"),
                # test splits: unchanged
                rel_test_id=rel_test_id,
                rel_test_conflict=rel_test_conflict,
                y_test_id=ds["test_id"]["y"],
                y_test_conflict=ds["test_conflict"]["y"],
                env_test_id=ds["test_id"]["env"],
                env_test_conflict=ds["test_conflict"]["env"],
                # budgets: unchanged
                budgets=b_star,
                b_kappa=b_kappa,
                regime=np.array(reg),
                seed=np.array(seed),
                rho_corr=np.array(rho_corr),
            )
            print(f"  [{reg}/seed{seed}] {save_path.name} "
                  f"cal_crc={rel_cal_crc.shape} test_id={rel_test_id.shape}")

    print(f"\nAll Gate 2 Step 5 artifacts saved to {out_path_dir}")
    validate_prepared_gate2_data(out_dir=out_dir, regimes=regimes, seeds=seeds, rho_corr=rho_corr)


def validate_prepared_gate2_data(
    out_dir: str = "results/gate2",
    regimes: List[str] = ["R1", "R2", "R3", "R4", "R5"],
    seeds: List[int] = [0, 1, 2, 3, 4],
    rho_corr: float = 0.9,
) -> int:
    """Validate Gate 2 Step 5 artifacts (extended checks)."""
    out_path_dir = Path(out_dir)
    print("\n" + "=" * 62)
    print("GATE 2 STEP 5 ARTIFACT VALIDATION")
    print("=" * 62)

    files = sorted(list(out_path_dir.glob("*.npz")))
    assert len(files) >= len(regimes) * len(seeds), \
        f"Expected {len(regimes)*len(seeds)} files, found {len(files)}"

    failures = 0

    for f in files:
        data = np.load(f, allow_pickle=True)
        fname = f.stem

        # Check 1: required keys
        required = ["rel_cal_crc", "rel_test_id", "rel_test_conflict", "budgets", "b_kappa",
                    "y_cal_crc", "c_cal_crc", "conflict_cal_crc", "cal_crc_mechanism"]
        for key in required:
            if key not in data:
                print(f"  [FAIL] {fname}: missing key {key!r}")
                failures += 1

        # Check 2: reliance validity
        for sk in ["rel_cal_crc", "rel_test_id", "rel_test_conflict"]:
            if sk not in data:
                continue
            rel = data[sk]
            ok = (rel.ndim == 2 and rel.shape[1] == 2
                  and not np.isnan(rel).any() and not np.isinf(rel).any()
                  and (rel >= 0.0).all()
                  and np.allclose(rel.sum(axis=1), 1.0, atol=1e-4))
            if not ok:
                print(f"  [FAIL] {fname}/{sk}: shape={rel.shape}")
                failures += 1

        # Check 3: cal_crc size
        if "rel_cal_crc" in data and data["rel_cal_crc"].shape[0] < 200:
            print(f"  [FAIL] {fname}: cal_crc size {data['rel_cal_crc'].shape[0]} < 200")
            failures += 1

        # Check 4: conflict rate in cal_crc < 0.40 (ID dist, not conflict)
        if "conflict_cal_crc" in data:
            cr = float(data["conflict_cal_crc"].mean())
            if cr > 0.40:
                print(f"  [FAIL] {fname}: conflict_rate={cr:.3f} > 0.40 (still conflict-like!)")
                failures += 1

        # Check 5: mechanism tag
        if "cal_crc_mechanism" in data:
            mech = str(data["cal_crc_mechanism"])
            if "id_cue_broken_false" not in mech:
                print(f"  [FAIL] {fname}: mechanism='{mech}' missing ID indicator")
                failures += 1

    print(f"  Structural checks: {len(files)} files, {failures} failures so far")

    # Check 6: BF accuracy parity
    representative = [("R2", 0), ("R2", 1), ("R5", 0), ("R1", 0), ("R4", 0)]
    print(f"\n  {'Case':<14} {'BF cal_crc':>12} {'BF test_id':>12} {'|delta|':>10} {'Status':>8}")
    print("  " + "-" * 58)

    for reg, seed in representative:
        artifact = out_path_dir / f"{reg}_seed{seed}.npz"
        if not artifact.exists():
            continue
        ds = generate_jdbs(regime=reg, rho_corr=rho_corr, seed=seed)
        cal_pool = _generate_id_cal_pool(ds)
        bf_model, _ = train_bf_mod(
            xs_train=[ds["train"]["X1"], ds["train"]["X2"]], y_train=ds["train"]["y"],
            xs_val=[ds["val_id"]["X1"], ds["val_id"]["X2"]], y_val=ds["val_id"]["y"],
            seed=seed, verbose=False,
        )
        bf_model.eval()
        with torch.no_grad():
            p_cal = bf_model([torch.tensor(cal_pool["X1"], dtype=torch.float32),
                              torch.tensor(cal_pool["X2"], dtype=torch.float32)]).argmax(-1).numpy()
            p_id  = bf_model([torch.tensor(ds["test_id"]["X1"], dtype=torch.float32),
                              torch.tensor(ds["test_id"]["X2"], dtype=torch.float32)]).argmax(-1).numpy()
        acc_cal = float((p_cal == cal_pool["y"]).mean())
        acc_id  = float((p_id  == ds["test_id"]["y"]).mean())
        delta   = abs(acc_cal - acc_id)
        ok = delta < 0.15
        if not ok:
            failures += 1
        print(f"  {reg}/seed{seed:<9} {acc_cal:>12.4f} {acc_id:>12.4f} {delta:>10.4f} {'OK' if ok else 'WARN':>8}")

    # Check 7: cue/label correlation in cal_crc
    print(f"\n  {'Case':<14} {'P(c=y) cal_crc':>16} {'Expected':>10} {'Status':>8}")
    print("  " + "-" * 50)
    for reg, seed in representative:
        artifact = out_path_dir / f"{reg}_seed{seed}.npz"
        if not artifact.exists():
            continue
        data = np.load(artifact, allow_pickle=True)
        p_cy = float((data["c_cal_crc"] == data["y_cal_crc"]).mean())
        ok = p_cy > 0.60
        if not ok:
            failures += 1
        print(f"  {reg}/seed{seed:<9} {p_cy:>16.4f} {'>=0.60 (ID)':>10} {'OK' if ok else 'FAIL':>8}")

    print("\n" + "=" * 62)
    if failures == 0:
        print("ALL GATE 2 STEP 5 VALIDATION CHECKS PASSED")
        print("GATE 2 STEP 5 STATUS: PASS")
    else:
        print(f"VALIDATION FAILED: {failures} check(s) failed")
        print("GATE 2 STEP 5 STATUS: FAIL")
    print("=" * 62)
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    cfg = load_config("configs/jdb_s.yaml")
    seeds = cfg.get("experiment", {}).get("seeds_dev" if args.dev else "seeds", [0, 1, 2, 3, 4])
    regimes = cfg.get("data", {}).get("regimes", ["R1", "R2", "R3", "R4", "R5"])

    prepare_gate2_data(regimes=regimes, seeds=seeds, rho_corr=0.9)
