"""
egsf/experiments/prepare_gate1_data.py
───────────────────────────────────────
Helper script to pre-compute and store D0 modality reliance scores and D2 budgets
for Gate 1 evaluation across regimes and dev seeds on JDB-S benchmark.

Output files saved to: results/gate1_prepared/<regime>_seed<seed>.npz
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch

# Make project root importable
sys.path.insert(0, str(Path(__file__).parents[2]))

from egsf.budget.d2_budget import compute_d2_budget
from egsf.data.jdb_s import generate_jdbs
from egsf.models.bf import train_bf_mod
from egsf.reliance.d0_reliance import compute_d0_reliance
from egsf.utils.reproducibility import load_config, seed_everything


def prepare_gate1_data(
    regimes: List[str] = ["R1", "R2", "R3", "R4", "R5"],
    seeds: List[int] = [0, 1, 2],
    rho_corr: float = 0.9,
    out_dir: str = "results/gate1_prepared",
) -> None:
    """
    Train BF Model 2, compute D0 reliance for train and val_id splits,
    and save NPZ artifacts for Gate 1 evaluation.
    """
    out_path_dir = Path(out_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing Gate 1 data for regimes={regimes}, seeds={seeds}, rho={rho_corr}...")

    for reg in regimes:
        for seed in seeds:
            seed_everything(seed)

            # 1. Generate JDB-S dataset for regime and seed
            ds = generate_jdbs(regime=reg, rho_corr=rho_corr, seed=seed)

            # 2. Train actual Base Fusion (BF, Model 2) classifier
            xs_train = [ds["train"]["X1"], ds["train"]["X2"]]
            y_train = ds["train"]["y"]
            xs_val = [ds["val_id"]["X1"], ds["val_id"]["X2"]]
            y_val = ds["val_id"]["y"]

            bf_model, _ = train_bf_mod(
                xs_train=xs_train,
                y_train=y_train,
                xs_val=xs_val,
                y_val=y_val,
                seed=seed,
                verbose=False,
            )

            # 3. Compute D0 Reliance scores separately on train and val_id splits
            rel_train = compute_d0_reliance(
                model=bf_model,
                xs=xs_train,
                n_resample=8,
                seed=seed,
            )

            rel_val = compute_d0_reliance(
                model=bf_model,
                xs=xs_val,
                n_resample=8,
                seed=seed,
            )

            # 4. Compute D2 Information Budget
            b_star, b_kappa_bounds = compute_d2_budget(ds, kappa_grid=[0.1])

            # 5. Save NPZ artifact with full (N, 2) reliance arrays and metadata
            artifact_name = f"{reg}_seed{seed}.npz"
            save_path = out_path_dir / artifact_name

            np.savez_compressed(
                save_path,
                rel_train=rel_train,
                rel_val=rel_val,
                budgets=b_star,
                regime=np.array(reg),
                seed=np.array(seed),
                rho_corr=np.array(rho_corr),
            )

            print(f"  Saved: {save_path} (rel_train: {rel_train.shape}, rel_val: {rel_val.shape})")

    print(f"Successfully generated all prepared data in {out_path_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Run dev preparation mode")
    args = parser.parse_args()

    cfg = load_config("configs/jdb_s.yaml")
    seeds = cfg.get("experiment", {}).get("seeds_dev" if args.dev else "seeds", [0, 1, 2])
    regimes = cfg.get("data", {}).get("regimes", ["R1", "R2", "R3", "R4", "R5"])

    prepare_gate1_data(regimes=regimes, seeds=seeds, rho_corr=0.9)

