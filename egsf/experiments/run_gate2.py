"""
egsf/experiments/run_gate2.py
──────────────────────────────
Gate 2 - C1 EGSF-Core Certification (Risk-Controlled Conformal Evaluation)

Evaluates Risk-Controlled Selective Correction under Conformal Risk Control (CRC):
1. FCR <= alpha (0.10) over >= 1000 calibration/test splits
2. Harmful-correction recall on shortcut conflict sets
3. Conflict accuracy improvement over BF on R2 and R5
4. Justified ID performance preserved (no catastrophic regression <= 5pp)
5. KL movement, modality starvation rate, budget violation rate, and risk-coverage curve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Make project root importable
sys.path.insert(0, str(Path(__file__).parents[2]))

from egsf.budget.d2_budget import compute_d2_budget
from egsf.calibration.e4_crc import calibrate_e4_crc, evaluate_e4_crc
from egsf.data.jdb_s import generate_jdbs
from egsf.gate.d3_gate import project_excess_reliance
from egsf.models.bf import train_bf_mod
from egsf.models.c1_egsf_core import train_c1_egsf
from egsf.utils.reproducibility import load_config, seed_everything


def _compute_kl_divergence(alpha_dist: np.ndarray, q_dist: np.ndarray, eps: float = 1e-9) -> float:
    """Compute KL divergence KL(alpha || q) averaged over instances."""
    a_clamped = np.clip(alpha_dist, eps, 1.0)
    q_clamped = np.clip(q_dist, eps, 1.0)
    kl = np.sum(a_clamped * np.log(a_clamped / q_clamped), axis=-1)
    return float(np.mean(kl))


def run_gate2_evaluation(dev_mode: bool = True, n_splits: Optional[int] = None) -> Dict:
    cfg = load_config("configs/jdb_s.yaml")
    seeds = cfg["experiment"]["seeds_dev"] if dev_mode else cfg["experiment"]["seeds"]
    regimes = cfg["data"]["regimes"]
    alpha = cfg["crc"]["alpha"]  # 0.10
    if n_splits is None:
        n_splits = 50 if dev_mode else cfg["crc"]["n_splits"]  # 1000 for full
    rho_corr = 0.9

    print("=" * 62)
    print(f"GATE 2 CERTIFICATION EVALUATION (dev_mode={dev_mode}, seeds={seeds}, n_splits={n_splits})")
    print(f"Target Risk: alpha = {alpha}")
    print("=" * 62 + "\n")

    all_regime_results = {}
    sweep_results = []

    for reg in regimes:
        reg_fcr_list = []
        reg_recall_list = []
        reg_bf_conf_acc = []
        reg_c1_conf_acc = []
        reg_crc_conf_acc = []
        reg_bf_id_acc = []
        reg_c1_id_acc = []
        reg_crc_id_acc = []
        reg_kl_list = []
        reg_starvation_list = []
        reg_violation_list = []

        for seed in seeds:
            seed_everything(seed)
            ds = generate_jdbs(regime=reg, rho_corr=rho_corr, seed=seed)

            # 1. Load pre-computed Gate 2 D0 reliance artifacts & prepared datasets
            prep_path = Path("results/gate2") / f"{reg}_seed{seed}.npz"
            if not prep_path.exists():
                raise FileNotFoundError(f"Prepared Gate 2 artifact missing: {prep_path}. Run prepare_gate2_data.py first.")

            prep_data = np.load(prep_path)
            rel_cal_crc = prep_data["rel_cal_crc"]
            rel_test_id = prep_data["rel_test_id"]
            rel_test_conflict = prep_data["rel_test_conflict"]
            b_kappa = prep_data["b_kappa"]

            # Load Gate 1 prepared train/val reliance for training C1
            prep_g1 = np.load(Path("results/gate1_prepared") / f"{reg}_seed{seed}.npz")
            rel_train = prep_g1["rel_train"]
            rel_val = prep_g1["rel_val"]

            def _t(split, mod):
                return torch.tensor(ds[split][mod], dtype=torch.float32)

            # 2. Train Base Fusion (BF) baseline
            bf_mod, _ = train_bf_mod(
                [_t("train", "X1"), _t("train", "X2")], ds["train"]["y"],
                [_t("val_id", "X1"), _t("val_id", "X2")], ds["val_id"]["y"],
                seed=seed, verbose=False
            )

            # 3. Train C1 EGSF-Core
            c1_mod, _ = train_c1_egsf(
                [_t("train", "X1"), _t("train", "X2")], ds["train"]["y"],
                [_t("val_id", "X1"), _t("val_id", "X2")], ds["val_id"]["y"],
                budget_bounds=b_kappa,
                rel_train=rel_train,
                rel_val=rel_val,
                seed=seed
            )

            bf_mod.eval()
            c1_mod.eval()

            # Helper for predictions & gates
            def _eval_split(split_name, rel_arr):
                x1 = _t(split_name, "X1")
                x2 = _t(split_name, "X2")
                rel_t = torch.tensor(rel_arr, dtype=torch.float32)
                b_t = torch.tensor(b_kappa, dtype=torch.float32)

                with torch.no_grad():
                    logits_bf = bf_mod([x1, x2])
                    preds_bf = logits_bf.argmax(dim=-1).numpy()

                    logits_c1, gates_c1 = c1_mod([x1, x2], reliances=rel_t)
                    preds_c1 = logits_c1.argmax(dim=-1).numpy()
                    gates_c1_np = gates_c1.numpy()

                    # Compute Excess Reliance D_i = max(0, rel - budget)
                    q_proj, er_tensor, alpha_tensor = project_excess_reliance(rel_t, None, b_t)
                    D_scores = er_tensor[:, 1].numpy()  # Spurious cue modality excess reliance

                return preds_bf, preds_c1, gates_c1_np, D_scores, alpha_tensor.numpy()

            # Evaluate on splits (cal_crc uses dedicated fresh independent cal_crc split)
            bf_p_cal, c1_p_cal, g_c1_cal, D_cal, a_cal = _eval_split("cal_crc", rel_cal_crc)
            bf_p_id, c1_p_id, g_c1_id, D_id, a_id = _eval_split("test_id", rel_test_id)
            bf_p_conf, c1_p_conf, g_c1_conf, D_conf, a_conf = _eval_split("test_conflict", rel_test_conflict)

            y_cal = ds["cal_crc"]["y"]
            y_id = ds["test_id"]["y"]
            y_conf = ds["test_conflict"]["y"]

            # False-Correction Loss L_i on justified calibration examples:
            # L_i = 1 iff uncorrected BF was correct BUT C1 correction alters/misclassifies
            L_cal = ((bf_p_cal == y_cal) & (c1_p_cal != y_cal)).astype(np.float64)

            # 4. 1000-Split Conformal Certification Loop
            rng = np.random.RandomState(seed)
            N_cal = len(D_cal)

            for _ in range(n_splits):
                # Sample calibration sub-split (size 150)
                cal_idx = rng.choice(N_cal, size=int(0.75 * N_cal), replace=False)
                lam_star = calibrate_e4_crc(D_cal[cal_idx], L_cal[cal_idx], alpha=alpha)

                # Apply lambda_star to test_id (justified set)
                corr_mask_id = D_id >= lam_star
                preds_crc_id = np.where(corr_mask_id, c1_p_id, bf_p_id)
                gates_crc_id = np.where(corr_mask_id[:, None], g_c1_id, 0.5)

                # Empirical False-Correction Rate (FCR) on test_id
                false_corrections = ((bf_p_id == y_id) & (preds_crc_id != y_id)).astype(np.float64)
                fcr_split = float(np.mean(false_corrections))
                reg_fcr_list.append(fcr_split)

                # Apply lambda_star to test_conflict (shortcut set)
                corr_mask_conf = D_conf >= lam_star
                preds_crc_conf = np.where(corr_mask_conf, c1_p_conf, bf_p_conf)
                gates_crc_conf = np.where(corr_mask_conf[:, None], g_c1_conf, 0.5)

                # Harmful-Correction Recall on test_conflict
                recall_split = float(np.mean(corr_mask_conf))
                reg_recall_list.append(recall_split)

                # Accuracies
                reg_bf_conf_acc.append(float(np.mean(bf_p_conf == y_conf)))
                reg_c1_conf_acc.append(float(np.mean(c1_p_conf == y_conf)))
                reg_crc_conf_acc.append(float(np.mean(preds_crc_conf == y_conf)))

                reg_bf_id_acc.append(float(np.mean(bf_p_id == y_id)))
                reg_c1_id_acc.append(float(np.mean(c1_p_id == y_id)))
                reg_crc_id_acc.append(float(np.mean(preds_crc_id == y_id)))

                # Diagnostic metrics
                kl_val = _compute_kl_divergence(a_id, gates_crc_id)
                starvation = float(np.mean(gates_crc_id < 1e-4))
                violation = float(np.mean(gates_crc_id > b_kappa[None, :] + 1e-5))

                reg_kl_list.append(kl_val)
                reg_starvation_list.append(starvation)
                reg_violation_list.append(violation)

        # Summarize regime statistics
        fcr_arr = np.array(reg_fcr_list)
        fcr_mean = float(np.mean(fcr_arr))
        fcr_median = float(np.median(fcr_arr))
        fcr_p95 = float(np.percentile(fcr_arr, 95))
        fcr_max = float(np.max(fcr_arr))
        fcr_sem = float(np.std(fcr_arr, ddof=1) / np.sqrt(len(fcr_arr))) if len(fcr_arr) > 1 else 0.0
        fcr_ci_lo = fcr_mean - 1.96 * fcr_sem
        fcr_ci_hi = fcr_mean + 1.96 * fcr_sem

        splits_guaranteed = float(np.mean(fcr_arr <= alpha))
        recall_mean = float(np.mean(reg_recall_list))

        bf_conf_m = float(np.mean(reg_bf_conf_acc))
        c1_conf_m = float(np.mean(reg_c1_conf_acc))
        crc_conf_m = float(np.mean(reg_crc_conf_acc))

        bf_id_m = float(np.mean(reg_bf_id_acc))
        c1_id_m = float(np.mean(reg_c1_id_acc))
        crc_id_m = float(np.mean(reg_crc_id_acc))

        id_regress = bf_id_m - crc_id_m

        # Risk-Coverage Curve across threshold grid
        grid_lambdas = np.linspace(0.0, 1.0, 50)
        risk_cov_curve = []
        for lam_g in grid_lambdas:
            sel_g = D_conf >= lam_g
            cov_g = float(np.mean(sel_g))
            fcr_g = float(np.mean(L_cal[D_cal >= lam_g])) if (D_cal >= lam_g).sum() > 0 else 0.0
            risk_cov_curve.append(dict(threshold=float(lam_g), coverage=cov_g, empirical_fcr=fcr_g))

        reg_res = dict(
            regime=reg,
            fcr_mean=fcr_mean,
            fcr_median=fcr_median,
            fcr_p95=fcr_p95,
            fcr_max=fcr_max,
            fcr_ci_lo=fcr_ci_lo,
            fcr_ci_hi=fcr_ci_hi,
            splits_guaranteed_fraction=splits_guaranteed,
            harmful_correction_recall=recall_mean,
            bf_conflict_acc=bf_conf_m,
            c1_conflict_acc=c1_conf_m,
            crc_conflict_acc=crc_conf_m,
            bf_id_acc=bf_id_m,
            c1_id_acc=c1_id_m,
            crc_id_acc=crc_id_m,
            id_regression=id_regress,
            kl_movement_mean=float(np.mean(reg_kl_list)),
            modality_starvation_rate=float(np.mean(reg_starvation_list)),
            budget_violation_rate=float(np.mean(reg_violation_list)),
            risk_coverage_curve=risk_cov_curve,
        )
        all_regime_results[reg] = reg_res
        sweep_results.append(reg_res)

        print(f"  {reg}: FCR_mean={fcr_mean:.4f} CI=[{fcr_ci_lo:.4f},{fcr_ci_hi:.4f}] P95={fcr_p95:.4f}, "
              f"Recall={recall_mean:.4f}, Conflict_Acc(BF/C1/CRC)={bf_conf_m:.3f}/{c1_conf_m:.3f}/{crc_conf_m:.3f}")

    # Gate 2 Certification Decision Criteria
    overall_fcr_mean = float(np.mean([r["fcr_mean"] for r in sweep_results]))
    all_fcr_pass = bool(overall_fcr_mean <= alpha)

    r2_conf_improve = all_regime_results["R2"]["crc_conflict_acc"] - all_regime_results["R2"]["bf_conflict_acc"]
    r5_conf_improve = all_regime_results["R5"]["crc_conflict_acc"] - all_regime_results["R5"]["bf_conflict_acc"]
    conf_pass = bool(r2_conf_improve > 0.0 and r5_conf_improve > 0.0)

    mean_id_regress = float(np.mean([r["id_regression"] for r in sweep_results]))
    justified_pass = bool(mean_id_regress <= 0.05)

    gate2_pass = bool(all_fcr_pass and conf_pass and justified_pass)

    print("\n" + "=" * 62)
    print("GATE 2 EVALUATION RESULTS")
    print("=" * 62)
    print(f"  (1) FCR <= {alpha} across regimes (mean={overall_fcr_mean:.4f}): {'PASS' if all_fcr_pass else 'FAIL'}")
    print(f"  (2) Conflict improvement R2={r2_conf_improve:+.4f}, R5={r5_conf_improve:+.4f}: {'PASS' if conf_pass else 'FAIL'}")
    print(f"  (3) test_id regression <= 5pp (mean={mean_id_regress:+.4f}): {'PASS' if justified_pass else 'FAIL'}")
    print(f"\n  GATE 2: {'PASS' if gate2_pass else 'FAIL'}")
    print("=" * 62)

    summary = dict(
        gate2_pass=gate2_pass,
        target_alpha=alpha,
        n_splits=n_splits,
        overall_fcr_mean=overall_fcr_mean,
        all_fcr_pass=all_fcr_pass,
        conf_pass=conf_pass,
        justified_pass=justified_pass,
        r2_conf_improvement=r2_conf_improve,
        r5_conf_improvement=r5_conf_improve,
        mean_id_regression=mean_id_regress,
        regime_results=all_regime_results,
        dev_mode=dev_mode,
        seeds=seeds,
    )

    out_path = Path("results") / "gate2_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Results saved to: {out_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Run fast dev benchmark evaluation")
    parser.add_argument("--n-splits", type=int, default=None, help="Override number of calibration/test splits")
    args = parser.parse_args()

    res = run_gate2_evaluation(dev_mode=args.dev, n_splits=args.n_splits)
    sys.exit(0 if res["gate2_pass"] else 1)
