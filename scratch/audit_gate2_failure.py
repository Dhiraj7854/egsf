"""
Scratch script for read-only diagnostic audit of Gate 2 failure.
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, ".")

from egsf.data.jdb_s import generate_jdbs
from egsf.models.bf import train_bf_mod
from egsf.models.c1_egsf_core import train_c1_egsf
from egsf.gate.d3_gate import project_excess_reliance
from egsf.calibration.e4_crc import calibrate_e4_crc
from egsf.utils.reproducibility import load_config, seed_everything

def run_audit():
    cfg = load_config("configs/jdb_s.yaml")
    seeds = cfg["experiment"]["seeds"] # [0, 1, 2, 3, 4]
    regimes = cfg["data"]["regimes"]   # ["R1", "R2", "R3", "R4", "R5"]
    alpha = 0.10
    rho_corr = 0.9

    print("=" * 70)
    print("GATE 2 READ-ONLY DIAGNOSTIC AUDIT")
    print("=" * 70)

    for reg in regimes:
        print(f"\n==================== REGIME {reg} ====================")
        
        bf_acc_cal_list, L_cal_rate_list = [], []
        bf_acc_id_list, c1_acc_id_list = [], []
        bf_acc_conf_list, c1_acc_conf_list = [], []
        
        D_cal_all, D_id_all, D_conf_all = [], [], []
        lam_star_list = []
        
        # Contingency counts over all test_id instances across splits
        breakdown_id = {"correct_to_wrong": 0, "wrong_to_correct": 0, "correct_to_correct": 0, "wrong_to_wrong": 0}
        breakdown_conf = {"correct_to_wrong": 0, "wrong_to_correct": 0, "correct_to_correct": 0, "wrong_to_wrong": 0}
        
        total_selected_id = 0
        total_selected_conf = 0

        for seed in seeds:
            seed_everything(seed)
            ds = generate_jdbs(regime=reg, rho_corr=rho_corr, seed=seed)

            prep_path = Path("results/gate2") / f"{reg}_seed{seed}.npz"
            prep_data = np.load(prep_path)
            rel_cal_crc = prep_data["rel_cal_crc"]
            rel_test_id = prep_data["rel_test_id"]
            rel_test_conflict = prep_data["rel_test_conflict"]
            b_kappa = prep_data["b_kappa"]

            prep_g1 = np.load(Path("results/gate1_prepared") / f"{reg}_seed{seed}.npz")
            rel_train = prep_g1["rel_train"]
            rel_val = prep_g1["rel_val"]

            def _t(split, mod):
                return torch.tensor(ds[split][mod], dtype=torch.float32)

            bf_mod, _ = train_bf_mod(
                [_t("train", "X1"), _t("train", "X2")], ds["train"]["y"],
                [_t("val_id", "X1"), _t("val_id", "X2")], ds["val_id"]["y"],
                seed=seed, verbose=False
            )

            c1_mod, _ = train_c1_egsf(
                [_t("train", "X1"), _t("train", "X2")], ds["train"]["y"],
                [_t("val_id", "X1"), _t("val_id", "X2")], ds["val_id"]["y"],
                budget_bounds=b_kappa,
                rel_train=rel_train, rel_val=rel_val,
                seed=seed
            )

            bf_mod.eval()
            c1_mod.eval()

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

                    q_proj, er_tensor, alpha_tensor = project_excess_reliance(rel_t, None, b_t)
                    D_scores = er_tensor[:, 1].numpy()

                return preds_bf, preds_c1, gates_c1_np, D_scores

            bf_p_cal, c1_p_cal, g_c1_cal, D_cal = _eval_split("cal_crc", rel_cal_crc)
            bf_p_id, c1_p_id, g_c1_id, D_id = _eval_split("test_id", rel_test_id)
            bf_p_conf, c1_p_conf, g_c1_conf, D_conf = _eval_split("test_conflict", rel_test_conflict)

            y_cal = ds["cal_crc"]["y"]
            y_id = ds["test_id"]["y"]
            y_conf = ds["test_conflict"]["y"]

            bf_acc_cal = np.mean(bf_p_cal == y_cal)
            L_cal = ((bf_p_cal == y_cal) & (c1_p_cal != y_cal)).astype(np.float64)
            L_cal_rate = np.mean(L_cal)

            bf_acc_cal_list.append(bf_acc_cal)
            L_cal_rate_list.append(L_cal_rate)
            
            bf_acc_id_list.append(np.mean(bf_p_id == y_id))
            c1_acc_id_list.append(np.mean(c1_p_id == y_id))
            bf_acc_conf_list.append(np.mean(bf_p_conf == y_conf))
            c1_acc_conf_list.append(np.mean(c1_p_conf == y_conf))

            D_cal_all.extend(D_cal)
            D_id_all.extend(D_id)
            D_conf_all.extend(D_conf)

            # 1000 splits simulation for threshold inspection
            rng = np.random.RandomState(seed)
            N_cal = len(D_cal)

            for _ in range(100): # 100 subsamples per seed for fast audit
                cal_idx = rng.choice(N_cal, size=int(0.75 * N_cal), replace=False)
                lam_star = calibrate_e4_crc(D_cal[cal_idx], L_cal[cal_idx], alpha=alpha)
                lam_star_list.append(lam_star)

                # Breakdown on test_id
                sel_id = D_id >= lam_star
                total_selected_id += sel_id.sum()
                for i in np.where(sel_id)[0]:
                    bf_c = (bf_p_id[i] == y_id[i])
                    c1_c = (c1_p_id[i] == y_id[i])
                    if bf_c and not c1_c:
                        breakdown_id["correct_to_wrong"] += 1
                    elif not bf_c and c1_c:
                        breakdown_id["wrong_to_correct"] += 1
                    elif bf_c and c1_c:
                        breakdown_id["correct_to_correct"] += 1
                    else:
                        breakdown_id["wrong_to_wrong"] += 1

                # Breakdown on test_conflict
                sel_conf = D_conf >= lam_star
                total_selected_conf += sel_conf.sum()
                for i in np.where(sel_conf)[0]:
                    bf_c = (bf_p_conf[i] == y_conf[i])
                    c1_c = (c1_p_conf[i] == y_conf[i])
                    if bf_c and not c1_c:
                        breakdown_conf["correct_to_wrong"] += 1
                    elif not bf_c and c1_c:
                        breakdown_conf["wrong_to_correct"] += 1
                    elif bf_c and c1_c:
                        breakdown_conf["correct_to_correct"] += 1
                    else:
                        breakdown_conf["wrong_to_wrong"] += 1

        print(f"A. Cal-CRC BF Accuracy: {np.mean(bf_acc_cal_list):.4f}")
        print(f"   Cal-CRC L=1 Loss Rate (BF correct & C1 wrong): {np.mean(L_cal_rate_list):.4f}")
        print(f"   Accuracies (ID): BF={np.mean(bf_acc_id_list):.4f}, C1={np.mean(c1_acc_id_list):.4f}")
        print(f"   Accuracies (Conf): BF={np.mean(bf_acc_conf_list):.4f}, C1={np.mean(c1_acc_conf_list):.4f}")

        D_cal_arr = np.array(D_cal_all)
        D_id_arr = np.array(D_id_all)
        D_conf_arr = np.array(D_conf_all)
        print(f"B. D_cal quantiles  [min, 25%, 50%, 75%, max]: {np.quantile(D_cal_arr, [0, 0.25, 0.5, 0.75, 1.0])}")
        print(f"   D_id quantiles   [min, 25%, 50%, 75%, max]: {np.quantile(D_id_arr, [0, 0.25, 0.5, 0.75, 1.0])}")
        print(f"   D_conf quantiles [min, 25%, 50%, 75%, max]: {np.quantile(D_conf_arr, [0, 0.25, 0.5, 0.75, 1.0])}")

        lam_arr = np.array(lam_star_list)
        print(f"C. lambda* quantiles [min, 25%, 50%, 75%, max]: {np.quantile(lam_arr, [0, 0.25, 0.5, 0.75, 1.0])}")
        print(f"   Fraction lambda* <= 0: {np.mean(lam_arr <= 1e-6):.4f}")

        print(f"D. Selection fraction test_id: {total_selected_id / (500 * len(lam_arr)):.4f}")
        print(f"   Selection fraction test_conf: {total_selected_conf / (500 * len(lam_arr)):.4f}")

        print("E. Breakdown on CRC-selected test_id instances:")
        if total_selected_id > 0:
            for k, v in breakdown_id.items():
                print(f"   - {k:<20}: {v} ({v / total_selected_id * 100:.2f}%)")
        print("   Breakdown on CRC-selected test_conflict instances:")
        if total_selected_conf > 0:
            for k, v in breakdown_conf.items():
                print(f"   - {k:<20}: {v} ({v / total_selected_conf * 100:.2f}%)")

if __name__ == "__main__":
    run_audit()
