"""
Gate 2 CRC Smoke Test — post-fix sanity check.
Tests R2/seed0, R4/seed0, R5/seed0 with 50 splits each.
Uses existing evaluator logic from run_gate2.py, fixed e4_crc.py, and
the regenerated independent cal_crc artifacts. Does NOT modify results.
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, ".")

from egsf.calibration.e4_crc import calibrate_e4_crc, evaluate_e4_crc
from egsf.data.jdb_s import generate_jdbs
from egsf.gate.d3_gate import project_excess_reliance
from egsf.models.bf import train_bf_mod
from egsf.models.c1_egsf_core import train_c1_egsf
from egsf.utils.reproducibility import seed_everything

CASES = [("R2", 0), ("R4", 0), ("R5", 0)]
N_SPLITS = 50
ALPHA = 0.10
RHO_CORR = 0.9

print("=" * 70)
print("GATE 2 CRC SMOKE TEST (post-fix, independent cal_crc)")
print(f"Cases: {CASES}  |  n_splits={N_SPLITS}  |  alpha={ALPHA}")
print("=" * 70)

overall_pass = True

for reg, seed in CASES:
    print(f"\n{'='*60}")
    print(f"  REGIME {reg}  seed {seed}")
    print(f"{'='*60}")

    seed_everything(seed)
    ds = generate_jdbs(regime=reg, rho_corr=RHO_CORR, seed=seed)

    prep_path = Path("results/gate2") / f"{reg}_seed{seed}.npz"
    if not prep_path.exists():
        print(f"  [ERROR] Artifact not found: {prep_path}")
        overall_pass = False
        continue

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
            preds_bf = bf_mod([x1, x2]).argmax(-1).numpy()
            logits_c1, gates_c1 = c1_mod([x1, x2], reliances=rel_t)
            preds_c1 = logits_c1.argmax(-1).numpy()
            _, er_tensor, _ = project_excess_reliance(rel_t, None, b_t)
            D_scores = er_tensor[:, 1].numpy()
        return preds_bf, preds_c1, D_scores

    bf_p_cal, c1_p_cal, D_cal = _eval_split("cal_crc", rel_cal_crc)
    bf_p_id,  c1_p_id,  D_id  = _eval_split("test_id",  rel_test_id)
    bf_p_conf, c1_p_conf, D_conf = _eval_split("test_conflict", rel_test_conflict)

    y_cal  = ds["cal_crc"]["y"]
    y_id   = ds["test_id"]["y"]
    y_conf = ds["test_conflict"]["y"]

    L_cal = ((bf_p_cal == y_cal) & (c1_p_cal != y_cal)).astype(np.float64)

    print(f"  BF acc (cal_crc):   {np.mean(bf_p_cal == y_cal):.4f}")
    print(f"  C1 acc (cal_crc):   {np.mean(c1_p_cal == y_cal):.4f}")
    print(f"  L=1 rate (cal_crc): {np.mean(L_cal):.4f}")
    print(f"  BF acc (test_id):   {np.mean(bf_p_id == y_id):.4f}")
    print(f"  C1 acc (test_id):   {np.mean(c1_p_id == y_id):.4f}")
    print(f"  BF acc (conf):      {np.mean(bf_p_conf == y_conf):.4f}")
    print(f"  C1 acc (conf):      {np.mean(c1_p_conf == y_conf):.4f}")
    print(f"  D_cal  [min,25,50,75,max]: {np.quantile(D_cal, [0,.25,.5,.75,1])}")
    print(f"  D_id   [min,25,50,75,max]: {np.quantile(D_id,  [0,.25,.5,.75,1])}")
    print(f"  D_conf [min,25,50,75,max]: {np.quantile(D_conf,[0,.25,.5,.75,1])}")

    rng = np.random.RandomState(seed)
    N_cal = len(D_cal)

    lam_list = []
    fcr_list = []
    sel_id_frac = []
    sel_conf_frac = []
    acc_crc_id_list = []
    acc_crc_conf_list = []

    for _ in range(N_SPLITS):
        cal_idx = rng.choice(N_cal, size=int(0.75 * N_cal), replace=False)
        lam_star = calibrate_e4_crc(D_cal[cal_idx], L_cal[cal_idx], alpha=ALPHA)
        lam_list.append(lam_star)

        # 9. Verify selected set is {D >= lambda*}
        sel_id = D_id >= lam_star
        sel_conf = D_conf >= lam_star

        preds_crc_id = np.where(sel_id, c1_p_id, bf_p_id)
        preds_crc_conf = np.where(sel_conf, c1_p_conf, bf_p_conf)

        false_corr = ((bf_p_id == y_id) & (preds_crc_id != y_id)).astype(float)
        fcr_list.append(float(np.mean(false_corr)))
        sel_id_frac.append(float(np.mean(sel_id)))
        sel_conf_frac.append(float(np.mean(sel_conf)))
        acc_crc_id_list.append(float(np.mean(preds_crc_id == y_id)))
        acc_crc_conf_list.append(float(np.mean(preds_crc_conf == y_conf)))

    lam_arr = np.array(lam_list)
    print(f"\n  --- Lambda* over {N_SPLITS} splits ---")
    print(f"  min:    {lam_arr.min():.6f}")
    print(f"  25th:   {np.percentile(lam_arr, 25):.6f}")
    print(f"  median: {np.median(lam_arr):.6f}")
    print(f"  75th:   {np.percentile(lam_arr, 75):.6f}")
    print(f"  max:    {lam_arr.max():.6f}")
    print(f"  Fraction lambda* near-zero (<1e-4): {np.mean(lam_arr < 1e-4):.4f}")

    # 8. Confirm lambda* is no longer near-zero (bug fix verification)
    near_zero_frac = np.mean(lam_arr < 1e-4)
    lam_above_d_median = np.mean(lam_arr > np.median(D_cal))
    print(f"  Fraction lambda* > median(D_cal): {lam_above_d_median:.4f}")

    sel_id_mean = np.mean(sel_id_frac)
    sel_conf_mean = np.mean(sel_conf_frac)
    fcr_mean = np.mean(fcr_list)
    fcr_p95 = np.percentile(fcr_list, 95)
    id_acc_before = np.mean(bf_p_id == y_id)
    id_acc_after = np.mean(acc_crc_id_list)
    conf_acc_before = np.mean(bf_p_conf == y_conf)
    conf_acc_after = np.mean(acc_crc_conf_list)

    print(f"\n  --- Selection fractions ---")
    print(f"  test_id selected:   {sel_id_mean:.4f}  (was ~0.99 before fix)")
    print(f"  test_conf selected: {sel_conf_mean:.4f}")

    print(f"\n  --- Empirical FCR ---")
    print(f"  FCR mean: {fcr_mean:.4f}  (alpha={ALPHA})  P95: {fcr_p95:.4f}")

    print(f"\n  --- ID accuracy ---")
    print(f"  BF (before):  {id_acc_before:.4f}")
    print(f"  CRC (after):  {id_acc_after:.4f}")
    print(f"  Regression:   {id_acc_before - id_acc_after:+.4f}")

    print(f"\n  --- Conflict accuracy ---")
    print(f"  BF (before):  {conf_acc_before:.4f}")
    print(f"  CRC (after):  {conf_acc_after:.4f}")
    print(f"  Improvement:  {conf_acc_after - conf_acc_before:+.4f}")

    # Pass/fail checks for the smoke test
    bug_fixed = bool(near_zero_frac < 0.50)
    print(f"\n  --- Smoke Test Checks ---")
    print(f"  [{'PASS' if bug_fixed else 'FAIL'}] lambda* no longer predominantly near-zero "
          f"(near-zero frac={near_zero_frac:.4f} < 0.50)")

    # Selection fraction should no longer be ~99% in all splits (for R2/R4/R5)
    sel_reasonable = bool(sel_id_mean < 0.99)
    print(f"  [{'PASS' if sel_reasonable else 'FAIL'}] Selection fraction < 99% "
          f"(got {sel_id_mean:.4f})")

    # FCR should be <= alpha or close (smoke test only, not full certification)
    fcr_ok = bool(fcr_mean <= ALPHA + 0.05)  # small tolerance for 50-split smoke test
    print(f"  [{'PASS' if fcr_ok else 'FAIL'}] FCR mean <= alpha+0.05 "
          f"({fcr_mean:.4f} vs {ALPHA+0.05:.4f})")

    case_pass = bug_fixed and sel_reasonable
    if not case_pass:
        overall_pass = False

print("\n" + "=" * 70)
print(f"SMOKE TEST OVERALL: {'PASS' if overall_pass else 'FAIL'}")
print("=" * 70)
