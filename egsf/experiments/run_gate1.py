"""
egsf/experiments/run_gate1.py  (v3 - correct accuracy-based criterion)
Gate 1 - ER vs Q-only Classification Performance Comparison

Preregistered Gate 1 criterion:
  - C1 EGSF-Core (ER-gated, reliance-aware) beats BAL-Q (Q-only, quantile-gated)
    by >= 0.05 accuracy on test_conflict
  - In >= 3/5 regimes including R2 and R5
  - No regression in R1/R3: C1 test_id accuracy >= BAL-Q test_id accuracy - 0.05

Rationale:
  BAL-Q (Q-only) gates based on confidence quantiles, with no reliance information.
  C1 EGSF-Core uses D0 reliance + D3 budget-constrained gate. Gate 1 certifies
  that the reliance-aware approach provides meaningful improvement over Q-only gating
  in shortcut-dominant environments (conflict set), without hurting causal regimes.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parents[2]))
from egsf.budget.d2_budget import compute_d2_budget
from egsf.data.jdb_s import generate_jdbs
from egsf.models.bf import train_bf_mod
from egsf.models.bal_q import train_bal_q
from egsf.models.c1_egsf_core import train_c1_egsf
from egsf.reliance.d0_reliance import compute_d0_reliance
from egsf.utils.reproducibility import load_config, seed_everything


def _acc(model, xs_list, y, reliances=None):
    model.eval()
    with torch.no_grad():
        if reliances is not None:
            rel_t = torch.tensor(reliances, dtype=torch.float32) if isinstance(reliances, np.ndarray) else reliances
            out = model(xs_list, reliances=rel_t)
        else:
            out = model(xs_list)
        logits = out[0] if isinstance(out, tuple) else out
        preds = logits.argmax(dim=-1).numpy()
    return float(np.mean(preds == y))


def run_gate1_evaluation(dev_mode: bool = True) -> Dict:
    cfg = load_config("configs/jdb_s.yaml")
    seeds = cfg["experiment"]["seeds_dev"] if dev_mode else cfg["experiment"]["seeds"]
    regimes = cfg["data"]["regimes"]
    rho = 0.9

    print(f"Gate 1 Evaluation v3 (dev_mode={dev_mode}, seeds={seeds}, rho={rho})")
    print("Criterion: C1 (ER-gated) conflict accuracy > BAL-Q (Q-only) + 0.05")
    print("           In >= 3/5 regimes including R2 and R5")
    print("           No regression in R1/R3: C1 id_acc >= BAL-Q id_acc - 0.05\n")

    regime_results = {}

    for reg in regimes:
        seed_margins = []       # C1 conflict acc - BAL-Q conflict acc
        seed_regressions = []   # BAL-Q id_acc - C1 id_acc (positive = regression)

        for seed in seeds:
            seed_everything(seed)
            ds = generate_jdbs(regime=reg, rho_corr=rho, seed=seed)

            # Load pre-computed D0 reliance scores and D2 budgets
            prep_path = Path("results/gate1_prepared") / f"{reg}_seed{seed}.npz"
            prep_data = np.load(prep_path)
            rel_train = prep_data["rel_train"]
            rel_val = prep_data["rel_val"]
            budgets = prep_data["budgets"]

            def _t(split, mod): return torch.tensor(ds[split][mod], dtype=torch.float32)

            # Train Base Fusion (BF) to compute test-time D0 reliance scores
            bf_mod, _ = train_bf_mod(
                [ds["train"]["X1"], ds["train"]["X2"]], ds["train"]["y"],
                [ds["val_id"]["X1"], ds["val_id"]["X2"]], ds["val_id"]["y"],
                seed=seed, verbose=False
            )
            rel_test_conflict = compute_d0_reliance(
                bf_mod, [ds["test_conflict"]["X1"], ds["test_conflict"]["X2"]], seed=seed
            )
            rel_test_id = compute_d0_reliance(
                bf_mod, [ds["test_id"]["X1"], ds["test_id"]["X2"]], seed=seed
            )

            # Train BAL-Q (Q-only baseline) and set thresholds from actual D0 reliance (without oracle budgets)
            balq, _ = train_bal_q(
                [ds["train"]["X1"], ds["train"]["X2"]], ds["train"]["y"],
                [ds["val_id"]["X1"], ds["val_id"]["X2"]], ds["val_id"]["y"],
                quantile_tau=0.5, seed=seed
            )
            balq.set_quantile_thresholds(rel_train)

            # Train C1 (ER-gated) with pre-computed D0 reliance and D2 budget bounds
            B_star, B_kappa = compute_d2_budget(ds, kappa_grid=[0.1])
            c1, _ = train_c1_egsf(
                [ds["train"]["X1"], ds["train"]["X2"]], ds["train"]["y"],
                [ds["val_id"]["X1"], ds["val_id"]["X2"]], ds["val_id"]["y"],
                budget_bounds=B_kappa[0.1],
                rel_train=rel_train,
                rel_val=rel_val,
                seed=seed
            )

            balq_conf_acc = _acc(balq, [_t("test_conflict","X1"), _t("test_conflict","X2")], ds["test_conflict"]["y"], reliances=rel_test_conflict)
            c1_conf_acc   = _acc(c1,   [_t("test_conflict","X1"), _t("test_conflict","X2")], ds["test_conflict"]["y"], reliances=rel_test_conflict)
            balq_id_acc   = _acc(balq, [_t("test_id","X1"), _t("test_id","X2")], ds["test_id"]["y"], reliances=rel_test_id)
            c1_id_acc     = _acc(c1,   [_t("test_id","X1"), _t("test_id","X2")], ds["test_id"]["y"], reliances=rel_test_id)

            margin = c1_conf_acc - balq_conf_acc
            regression = balq_id_acc - c1_id_acc  # positive = regression

            seed_margins.append(margin)
            seed_regressions.append(regression)

        mean_margin = float(np.mean(seed_margins))
        mean_regress = float(np.mean(seed_regressions))
        sem_m = float(np.std(seed_margins, ddof=1) / np.sqrt(len(seed_margins))) if len(seed_margins) > 1 else 0.0
        ci_lo = mean_margin - 1.96 * sem_m
        ci_hi = mean_margin + 1.96 * sem_m

        regime_results[reg] = dict(
            mean_conflict_margin=mean_margin,
            ci_lo=ci_lo, ci_hi=ci_hi,
            mean_id_regression=mean_regress,
            seed_margins=seed_margins,
            seed_regressions=seed_regressions,
        )
        print(f"  {reg}: conflict_margin={mean_margin:+.4f} CI=[{ci_lo:+.4f},{ci_hi:+.4f}], id_regress={mean_regress:+.4f}")

    # Gate 1 decisions
    MARGIN_THRESH = 0.05    # C1 must beat BAL-Q by >= 0.05 on conflict
    REGRESS_TOL   = 0.05    # C1 allowed to lose <= 0.05 on test_id vs BAL-Q

    def _margin_pass(reg):
        v = regime_results.get(reg, {})
        return v.get("mean_conflict_margin", -1) >= MARGIN_THRESH and v.get("ci_lo", -1) > 0.0

    def _no_regress(reg):
        v = regime_results.get(reg, {})
        return v.get("mean_id_regression", 99) <= REGRESS_TOL

    r2_pass = _margin_pass("R2")
    r5_pass = _margin_pass("R5")
    r1_ok   = _no_regress("R1")
    r3_ok   = _no_regress("R3")
    n_meet  = sum(1 for r in regimes if _margin_pass(r))
    gate1_pass = bool(r2_pass and r5_pass and r1_ok and r3_ok and n_meet >= 3)

    print()
    print("=" * 62)
    print("GATE 1 EVALUATION RESULTS")
    print("=" * 62)
    print(f"  R2 C1>BAL-Q by >=0.05 on conflict, CI_lo>0:  {'PASS' if r2_pass else 'FAIL'}")
    print(f"  R5 C1>BAL-Q by >=0.05 on conflict, CI_lo>0:  {'PASS' if r5_pass else 'FAIL'}")
    print(f"  Regimes with margin >= 0.05 (need >=3):       {n_meet}/5")
    print(f"  R1 no regression (id_regress <= 0.05):        {'PASS' if r1_ok else 'FAIL'}")
    print(f"  R3 no regression (id_regress <= 0.05):        {'PASS' if r3_ok else 'FAIL'}")
    print(f"\n  GATE 1: {'PASS' if gate1_pass else 'FAIL'}")
    print("=" * 62)

    summary = dict(
        gate1_pass=gate1_pass, margin_threshold=MARGIN_THRESH, regress_tolerance=REGRESS_TOL,
        r2_pass=r2_pass, r5_pass=r5_pass, r1_no_regress=r1_ok, r3_no_regress=r3_ok,
        n_regimes_meeting_threshold=n_meet,
        regime_results=regime_results, dev_mode=dev_mode, seeds=seeds, rho=rho
    )
    out = Path("results") / "gate1_results.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Results saved to: {out}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dev", action="store_true")
    args = p.parse_args()
    r = run_gate1_evaluation(dev_mode=args.dev)
    sys.exit(0 if r["gate1_pass"] else 1)