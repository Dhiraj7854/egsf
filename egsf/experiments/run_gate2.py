"""
egsf/experiments/run_gate2.py
Gate 2 - C1 EGSF-Core Certification (Proper Criterion)

Preregistered Gate 2 criterion:
  1. Empirical FCR <= alpha (0.10) over >= 1000 random cal/test splits
  2. Conflict accuracy improves over BF on R2 and R5 (the shortcut regimes)
  3. Justified performance: test_id accuracy within 5pp of BF (no catastrophic regression)

Evaluation design:
  - For each seed and regime: train BF and C1 models once on train/val_id.
  - FCR sweep: randomly split cal_crc + test_id into 1000 (cal_sub, test_sub) splits,
    calibrate E4 CRC on cal_sub, evaluate false-correction rate on test_sub.
  - Conflict improvement: compare BF vs C1 on test_conflict split.
  - Justified performance: BF test_id accuracy - C1 test_id accuracy <= 0.05.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parents[2]))
from egsf.budget.d2_budget import compute_d2_budget
from egsf.calibration.e4_crc import calibrate_e4_crc, evaluate_e4_crc
from egsf.data.jdb_s import generate_jdbs
from egsf.models.bf import train_bf_mod
from egsf.models.c1_egsf_core import train_c1_egsf
from egsf.utils.reproducibility import load_config, seed_everything


def _get_scores_losses(model, xs_list, y, is_c1=False):
    model.eval()
    with torch.no_grad():
        out = model(xs_list)
        logits = out[0] if isinstance(out, tuple) else out
        probs = F.softmax(logits, dim=-1).numpy()
    scores = np.max(probs, axis=-1)
    preds = np.argmax(probs, axis=-1)
    losses = (preds != y).astype(np.float32)
    acc = float(np.mean(preds == y))
    return scores, losses, acc


def run_gate2_evaluation(dev_mode: bool = True, n_splits: int = None) -> Dict:
    cfg = load_config("configs/jdb_s.yaml")
    seeds = cfg["experiment"]["seeds_dev"] if dev_mode else cfg["experiment"]["seeds"]
    regimes = cfg["data"]["regimes"]
    alpha = cfg["crc"]["alpha"]
    if n_splits is None:
        n_splits = 50 if dev_mode else cfg["crc"]["n_splits"]
    rho = 0.9

    print(f"Gate 2 Evaluation (dev_mode={dev_mode}, seeds={seeds}, rho={rho}, n_splits={n_splits})")
    print(f"Criterion: (1) Empirical FCR <= {alpha} over {n_splits} cal/test splits")
    print(f"           (2) Conflict accuracy improvement over BF on R2 and R5")
    print(f"           (3) test_id accuracy regression <= 5pp vs BF\n")

    results = []

    for reg in regimes:
        for seed in seeds:
            seed_everything(seed)
            ds = generate_jdbs(regime=reg, rho_corr=rho, seed=seed)

            # Train BF
            bf, _ = train_bf_mod(
                [ds["train"]["X1"], ds["train"]["X2"]], ds["train"]["y"],
                [ds["val_id"]["X1"], ds["val_id"]["X2"]], ds["val_id"]["y"],
                seed=seed
            )

            # Train C1
            B_star, B_kappa = compute_d2_budget(ds, kappa_grid=[0.1])
            c1, _ = train_c1_egsf(
                [ds["train"]["X1"], ds["train"]["X2"]], ds["train"]["y"],
                [ds["val_id"]["X1"], ds["val_id"]["X2"]], ds["val_id"]["y"],
                budget_bounds=B_kappa[0.1], seed=seed
            )

            # Prepare tensors
            def _t(k, m): return torch.tensor(ds[k][m], dtype=torch.float32)

            # BF test_id acc
            _, _, bf_id_acc = _get_scores_losses(bf, [_t("test_id","X1"), _t("test_id","X2")], ds["test_id"]["y"])
            # C1 test_id acc
            _, _, c1_id_acc = _get_scores_losses(c1, [_t("test_id","X1"), _t("test_id","X2")], ds["test_id"]["y"])
            # BF test_conflict acc
            _, _, bf_conf_acc = _get_scores_losses(bf, [_t("test_conflict","X1"), _t("test_conflict","X2")], ds["test_conflict"]["y"])
            # C1 test_conflict acc
            _, _, c1_conf_acc = _get_scores_losses(c1, [_t("test_conflict","X1"), _t("test_conflict","X2")], ds["test_conflict"]["y"])

            # FCR sweep on C1 using cal_crc pool
            c1_cal_scores, c1_cal_losses, _ = _get_scores_losses(
                c1, [_t("cal_crc","X1"), _t("cal_crc","X2")], ds["cal_crc"]["y"])
            c1_test_scores, c1_test_losses, _ = _get_scores_losses(
                c1, [_t("test_id","X1"), _t("test_id","X2")], ds["test_id"]["y"])

            # Pool cal + test for random split FCR
            pooled_scores = np.concatenate([c1_cal_scores, c1_test_scores])
            pooled_losses = np.concatenate([c1_cal_losses, c1_test_losses])
            N_pool = len(pooled_scores)
            n_cal_sub = len(c1_cal_scores)

            rng = np.random.RandomState(seed)
            fcr_list = []
            coverage_list = []
            for _ in range(n_splits):
                idx = rng.permutation(N_pool)
                cal_idx = idx[:n_cal_sub]
                test_idx = idx[n_cal_sub:]
                lam = calibrate_e4_crc(pooled_scores[cal_idx], pooled_losses[cal_idx], alpha=alpha)
                emp_risk, cov = evaluate_e4_crc(pooled_scores[test_idx], pooled_losses[test_idx], lam)
                fcr_list.append(emp_risk)
                coverage_list.append(cov)

            mean_fcr = float(np.mean(fcr_list))
            mean_cov = float(np.mean(coverage_list))
            fcr_pass = bool(mean_fcr <= alpha)

            entry = dict(
                regime=reg, seed=seed,
                bf_id_acc=bf_id_acc, c1_id_acc=c1_id_acc,
                bf_conf_acc=bf_conf_acc, c1_conf_acc=c1_conf_acc,
                id_regression=bf_id_acc - c1_id_acc,
                conf_improvement=c1_conf_acc - bf_conf_acc,
                mean_fcr=mean_fcr, mean_coverage=mean_cov,
                fcr_pass=fcr_pass,
            )
            results.append(entry)
            print(f"  {reg} seed={seed}: FCR={mean_fcr:.4f}({'OK' if fcr_pass else 'OVER'}), "
                  f"conf_improve={c1_conf_acc - bf_conf_acc:+.4f}, id_regress={bf_id_acc - c1_id_acc:+.4f}")

    # Gate 2 decisions
    print()
    # (1) FCR: ALL runs must have mean_fcr <= alpha
    all_fcr_pass = all(r["fcr_pass"] for r in results)
    # (2) Conflict improvement in shortcut regimes R2 and R5 (mean across seeds)
    r2_conf = np.mean([r["conf_improvement"] for r in results if r["regime"] == "R2"])
    r5_conf = np.mean([r["conf_improvement"] for r in results if r["regime"] == "R5"])
    conf_pass = bool(r2_conf > 0.0 and r5_conf > 0.0)
    # (3) Justified performance: mean id_regression <= 0.05 across all regimes
    mean_regress = float(np.mean([r["id_regression"] for r in results]))
    justified_pass = bool(mean_regress <= 0.05)

    gate2_pass = bool(all_fcr_pass and conf_pass and justified_pass)

    print("=" * 62)
    print("GATE 2 EVALUATION RESULTS")
    print("=" * 62)
    print(f"  (1) FCR <= {alpha} in all runs:           {'PASS' if all_fcr_pass else 'FAIL'}")
    print(f"  (2) Conflict improvement R2={r2_conf:+.4f}, R5={r5_conf:+.4f}: {'PASS' if conf_pass else 'FAIL'}")
    print(f"  (3) test_id regression <= 5pp (mean={mean_regress:+.4f}): {'PASS' if justified_pass else 'FAIL'}")
    print(f"\n  GATE 2: {'PASS' if gate2_pass else 'FAIL'}")
    print("=" * 62)

    summary = dict(
        gate2_pass=gate2_pass, alpha=alpha, n_splits=n_splits,
        all_fcr_pass=all_fcr_pass, conf_pass=conf_pass, justified_pass=justified_pass,
        r2_conf_improvement=float(r2_conf), r5_conf_improvement=float(r5_conf),
        mean_id_regression=mean_regress, results=results, dev_mode=dev_mode, seeds=seeds, rho=rho
    )
    out = Path("results") / "gate2_fcr_results.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f: json.dump(summary, f, indent=2, default=str)
    print(f"Results saved to: {out}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dev", action="store_true")
    p.add_argument("--n-splits", type=int, default=None)
    args = p.parse_args()
    r = run_gate2_evaluation(dev_mode=args.dev, n_splits=args.n_splits)
    sys.exit(0 if r["gate2_pass"] else 1)
