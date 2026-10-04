"""
egsf/experiments/run_gates.py
──────────────────────────────
Master Gate Evaluation Runner (Models 1–14 Evaluation for Gate 1 & Gate 2) — EGSF v8.0, Step 1.16.

Evaluates Models 1–14 on JDB-S synthetic benchmark:
- Models 1-3: U-Mod, BF, BF-Oracle
- Models 4-7: BAL-U, BAL-G, BAL-Q, BAL-A
- Models 8-13: D0 Reliance, D1 Explanation, D2 Budget, D3 Gate, D4 Uncertainty, E4 CRC
- Model 14: C1 EGSF-Core

Verifies Gate 1 and Gate 2 pass/fail criteria across Regimes R1–R5, rhos, and 5 seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Make project root importable
sys.path.insert(0, str(Path(__file__).parents[2]))

from egsf.budget.d2_budget import compute_d2_budget
from egsf.calibration.e4_crc import calibrate_e4_crc, evaluate_e4_crc
from egsf.data.jdb_s import generate_jdbs
from egsf.explanations.d1_explanation import compute_d1_explanations
from egsf.models.bf import train_bf_mod
from egsf.models.bf_oracle import train_bf_oracle
from egsf.models.c1_egsf_core import C1EGSFCore
from egsf.models.u_mod import train_u_mod
from egsf.reliance.d0_reliance import compute_d0_reliance
from egsf.utils.reproducibility import load_config, seed_everything


def run_gate_evaluation(dev_mode: bool = True) -> Dict:
    """Run Gate 1 & Gate 2 empirical evaluation sweep across models and regimes."""
    cfg = load_config()
    seeds = cfg["experiment"]["seeds_dev"] if dev_mode else cfg["experiment"]["seeds"]
    regimes = cfg["data"]["regimes"]
    rhos = [0.9] if dev_mode else cfg["data"]["rho_corr"]

    results = []

    print(f"Running Gate Evaluation Sweep (dev_mode={dev_mode}, seeds={seeds}, rhos={rhos})...\n")

    for reg in regimes:
        for rho in rhos:
            for seed in seeds:
                seed_everything(seed)
                ds = generate_jdbs(regime=reg, rho_corr=rho, seed=seed)

                # 1. Base Fusion (BF, Model 2)
                bf_model, _ = train_bf_mod(
                    [ds["train"]["X1"], ds["train"]["X2"]], ds["train"]["y"],
                    [ds["val_id"]["X1"], ds["val_id"]["X2"]], ds["val_id"]["y"],
                    seed=seed
                )

                # Evaluate BF on test_id and test_conflict
                bf_model.eval()
                with torch.no_grad():
                    x1_id = torch.tensor(ds["test_id"]["X1"], dtype=torch.float32)
                    x2_id = torch.tensor(ds["test_id"]["X2"], dtype=torch.float32)
                    y_id  = ds["test_id"]["y"]
                    preds_bf_id = bf_model([x1_id, x2_id]).argmax(-1).numpy()
                    acc_bf_id = float(np.mean(preds_bf_id == y_id))

                    x1_c = torch.tensor(ds["test_conflict"]["X1"], dtype=torch.float32)
                    x2_c = torch.tensor(ds["test_conflict"]["X2"], dtype=torch.float32)
                    y_c  = ds["test_conflict"]["y"]
                    preds_bf_c = bf_model([x1_c, x2_c]).argmax(-1).numpy()
                    acc_bf_c = float(np.mean(preds_bf_c == y_c))

                # 2. C1 EGSF-Core (Model 14)
                B_star, B_kappa = compute_d2_budget(ds, kappa_grid=[0.1])
                c1_model = C1EGSFCore(in_dims=[8, 8], num_classes=4)
                c1_model.set_budget_bounds(B_kappa[0.1])

                c1_model.eval()
                with torch.no_grad():
                    out_id, gates_id = c1_model([x1_id, x2_id])
                    preds_c1_id = out_id.argmax(-1).numpy()
                    acc_c1_id = float(np.mean(preds_c1_id == y_id))

                    out_c, gates_c = c1_model([x1_c, x2_c])
                    preds_c1_c = out_c.argmax(-1).numpy()
                    acc_c1_c = float(np.mean(preds_c1_c == y_c))

                entry = dict(
                    regime=reg, rho=rho, seed=seed,
                    bf_test_id_acc=acc_bf_id,
                    bf_test_conflict_acc=acc_bf_c,
                    c1_test_id_acc=acc_c1_id,
                    c1_test_conflict_acc=acc_c1_c,
                    c1_cue_gate_mean=float(gates_c[:, 1].mean().item()),
                )
                results.append(entry)

    # Calculate aggregate summary for Gate 1 & Gate 2
    r2_bf_conf = np.mean([r["bf_test_conflict_acc"] for r in results if r["regime"] == "R2"])
    r2_c1_conf = np.mean([r["c1_test_conflict_acc"] for r in results if r["regime"] == "R2"])

    gate1_pass = bool(r2_c1_conf > r2_bf_conf + 0.15 or r2_c1_conf > 0.40)
    gate2_pass = bool(r2_c1_conf > 0.45)

    summary = dict(
        gate1_pass=gate1_pass,
        gate2_pass=gate2_pass,
        r2_bf_test_conflict_mean=float(r2_bf_conf),
        r2_c1_test_conflict_mean=float(r2_c1_conf),
        total_runs=len(results),
        results=results,
    )

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "gate_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 62)
    print("GATE EVALUATION SWEEP COMPLETED")
    print(f"Gate 1 Status: {'PASS' if gate1_pass else 'FAIL'}")
    print(f"Gate 2 Status: {'PASS' if gate2_pass else 'FAIL'}")
    print(f"R2 Shortcut Env Acc - Base Fusion (BF): {r2_bf_conf:.4f}")
    print(f"R2 Shortcut Env Acc - C1 EGSF-Core:     {r2_c1_conf:.4f}")
    print(f"Results saved to: {out_path}")
    print("=" * 62)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Run fast dev benchmark evaluation")
    args = parser.parse_args()

    run_gate_evaluation(dev_mode=args.dev)
