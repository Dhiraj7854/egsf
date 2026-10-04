"""
egsf/experiments/run_phase1.py
───────────────────────────────
Phase 1 Benchmarking: Models 1–3 across Regimes R1–R5 and Shortcut Correlations rho.

Models evaluated:
1. U-Mod (Causal): Unimodal MLP on X1
2. U-Mod (Cue):    Unimodal MLP on X2
3. Base Fusion (BF): Concatenated MLP on [X1, X2]
4. G-EGSF:          Gated Selective Fusion on [X1, X2]

Evaluates:
- val_id accuracy
- test_id accuracy
- test_conflict accuracy (OOD performance under shortcut collapse)
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

from egsf.data.jdb_s import generate_jdbs
from egsf.models.bf import train_bf_mod
from egsf.models.egsf import train_egsf_mod
from egsf.models.u_mod import train_u_mod
from egsf.utils.reproducibility import load_config, seed_everything


def evaluate_model(model: torch.nn.Module, X1: np.ndarray, X2: np.ndarray, y: np.ndarray, is_multimodal: bool = False) -> float:
    """Evaluate accuracy on a split."""
    model.eval()
    with torch.no_grad():
        if is_multimodal:
            x1_t = torch.tensor(X1, dtype=torch.float32)
            x2_t = torch.tensor(X2, dtype=torch.float32)
            out = model([x1_t, x2_t])
            logits = out[0] if isinstance(out, tuple) else out
        else:
            x_t = torch.tensor(X1, dtype=torch.float32)
            logits = model(x_t)
        
        preds = logits.argmax(dim=-1).numpy()
        return float(np.mean(preds == y))


def run_phase1_benchmark(dev_mode: bool = True) -> Dict:
    """Run Phase 1 benchmarking sweep across models, regimes, rhos, and seeds."""
    cfg = load_config()
    
    seeds = cfg["experiment"]["seeds_dev"] if dev_mode else cfg["experiment"]["seeds"]
    regimes = cfg["data"]["regimes"]
    rhos = [0.8] if dev_mode else cfg["data"]["rho_corr"]

    results = []

    print(f"Running Phase 1 Benchmark (dev_mode={dev_mode}, seeds={seeds}, rhos={rhos})...\n")

    for reg in regimes:
        for rho in rhos:
            for seed in seeds:
                seed_everything(seed)
                ds = generate_jdbs(regime=reg, rho_corr=rho, seed=seed)

                # 1. U-Mod (Causal X1)
                u_causal, _ = train_u_mod(
                    ds["train"]["X1"], ds["train"]["y"],
                    ds["val_id"]["X1"], ds["val_id"]["y"],
                    seed=seed
                )
                acc_c_id   = evaluate_model(u_causal, ds["test_id"]["X1"], ds["test_id"]["X2"], ds["test_id"]["y"], False)
                acc_c_ood  = evaluate_model(u_causal, ds["test_conflict"]["X1"], ds["test_conflict"]["X2"], ds["test_conflict"]["y"], False)

                # 2. U-Mod (Cue X2)
                u_cue, _ = train_u_mod(
                    ds["train"]["X2"], ds["train"]["y"],
                    ds["val_id"]["X2"], ds["val_id"]["y"],
                    seed=seed
                )
                acc_q_id   = evaluate_model(u_cue, ds["test_id"]["X2"], ds["test_id"]["X2"], ds["test_id"]["y"], False)
                acc_q_ood  = evaluate_model(u_cue, ds["test_conflict"]["X2"], ds["test_conflict"]["X2"], ds["test_conflict"]["y"], False)

                # 3. Base Fusion (BF)
                bf_mod, _ = train_bf_mod(
                    [ds["train"]["X1"], ds["train"]["X2"]], ds["train"]["y"],
                    [ds["val_id"]["X1"], ds["val_id"]["X2"]], ds["val_id"]["y"],
                    seed=seed
                )
                acc_bf_id  = evaluate_model(bf_mod, ds["test_id"]["X1"], ds["test_id"]["X2"], ds["test_id"]["y"], True)
                acc_bf_ood = evaluate_model(bf_mod, ds["test_conflict"]["X1"], ds["test_conflict"]["X2"], ds["test_conflict"]["y"], True)

                # 4. G-EGSF
                egsf_mod, _ = train_egsf_mod(
                    [ds["train"]["X1"], ds["train"]["X2"]], ds["train"]["y"],
                    [ds["val_id"]["X1"], ds["val_id"]["X2"]], ds["val_id"]["y"],
                    seed=seed
                )
                acc_egsf_id  = evaluate_model(egsf_mod, ds["test_id"]["X1"], ds["test_id"]["X2"], ds["test_id"]["y"], True)
                acc_egsf_ood = evaluate_model(egsf_mod, ds["test_conflict"]["X1"], ds["test_conflict"]["X2"], ds["test_conflict"]["y"], True)

                res_entry = dict(
                    regime=reg, rho=rho, seed=seed,
                    u_causal_id=acc_c_id, u_causal_ood=acc_c_ood,
                    u_cue_id=acc_q_id, u_cue_ood=acc_q_ood,
                    bf_id=acc_bf_id, bf_ood=acc_bf_ood,
                    egsf_id=acc_egsf_id, egsf_ood=acc_egsf_ood,
                )
                results.append(res_entry)

    # Save results JSON
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / ("phase1_dev.json" if dev_mode else "phase1_full.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPhase 1 benchmark completed. Results saved to {out_path}")
    return {"n_runs": len(results), "path": str(out_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Run fast dev benchmark subset")
    args = parser.parse_args()
    
    run_phase1_benchmark(dev_mode=args.dev)
