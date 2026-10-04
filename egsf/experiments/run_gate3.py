"""
egsf/experiments/run_gate3.py
──────────────────────────────
Gate 3 - C3 Real Data Transfer (Oracle vs. Inferred Environments)

Uses the actual frozen D0 reliance and D2 budget machinery.
A1 (Oracle): Uses JDB-R oracle cue labels (c != y) as the conflict anchor.
A2 (Inferred): Uses Biased Predictor disagreement (pred_X2 != y) as the conflict anchor.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parents[2]))

from egsf.data.jdb_r import generate_jdbr
from egsf.models.bf import train_bf_mod
from egsf.models.bal_q import train_bal_q
from egsf.models.c1_egsf_core import train_c1_egsf
from egsf.reliance.d0_reliance import compute_d0_reliance
from egsf.utils.reproducibility import seed_everything

# Simple MLP for the modality-specific biased predictor
class SimpleMLP(nn.Module):
    def __init__(self, in_dim=8, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_unimodal(X_train, y_train, X_val, y_val, in_dim=8, epochs=15, lr=1e-3, seed=0):
    seed_everything(seed)
    model = SimpleMLP(in_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(Xt)
        loss = loss_fn(out, yt)
        loss.backward()
        optimizer.step()
        
    model.eval()
    return model

def evaluate_metrics(y_true, scores):
    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, scores)
        auprc = average_precision_score(y_true, scores)
    else:
        auroc, auprc = 1.0, 1.0
    return auroc, auprc

def get_acc(model, ds, split):
    model.eval()
    x1 = torch.tensor(ds[split]["X1"], dtype=torch.float32)
    x2 = torch.tensor(ds[split]["X2"], dtype=torch.float32)
    y = ds[split]["y"]
    with torch.no_grad():
        out = model([x1, x2])
        logits = out[0] if isinstance(out, tuple) else out
        preds = logits.argmax(dim=-1).numpy()
    return np.mean(preds == y)

def run_experiment(seeds, dev_mode=False):
    print("=" * 62)
    print("GATE 3 C3 REAL DATA TRANSFER (Oracle vs Inferred)")
    print("=" * 62)
    
    metrics = {
        "a1_auroc": [], "a1_auprc": [],
        "a2_auroc": [], "a2_auprc": [],
        "bf_conf": [], "bf_id": [],
        "balq_conf": [], "balq_id": [],
        "c3_conf": [], "c3_id": [],
        "a2_diff": []
    }
    
    # In JDB-R, X1 is Audio (dim 8), X2 is Image (dim 8, Modality 1 = Shortcut)
    in_dims = [8, 8]
    num_classes = 4
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        seed_everything(seed)
        ds = generate_jdbr(seed=seed)
        
        y_train = ds["train"]["y"]
        y_val = ds["val_id"]["y"]
        xs_train = [ds["train"]["X1"], ds["train"]["X2"]]
        xs_val = [ds["val_id"]["X1"], ds["val_id"]["X2"]]
        
        # 1. Train BF model for reliance estimation and baseline
        bf_model, _ = train_bf_mod(xs_train, y_train, xs_val, y_val, in_dims=in_dims, num_classes=num_classes, seed=seed, verbose=False)
        
        # 2. D0 Reliance
        rel_train = compute_d0_reliance(bf_model, xs_train, seed=seed)
        rel_val = compute_d0_reliance(bf_model, xs_val, seed=seed)
        
        # 3. A1 Oracle Anchor (Diagnostic Score)
        # Evaluated on conflict_dev
        c_oracle = ds["conflict_dev"]["oracle_meta"]["c"]
        y_dev = ds["conflict_dev"]["y"]
        anchor_a1 = (c_oracle != y_dev).astype(float)
        
        # 4. A2 Inferred Anchor (Diagnostic Score)
        # Train biased predictor on Modality 1 (X2, Image)
        biased_model = train_unimodal(
            ds["train"]["X2"], ds["train"]["y"],
            ds["val_id"]["X2"], ds["val_id"]["y"],
            in_dim=8, seed=seed
        )
        with torch.no_grad():
            preds_biased = biased_model(torch.tensor(ds["conflict_dev"]["X2"], dtype=torch.float32)).argmax(dim=-1).numpy()
        anchor_a2 = (preds_biased != y_dev).astype(float)
        
        # 5. Diagnostic Metrics (AUROC predicting cue_broken=True)
        # We need a mixed set to compute AUROC properly. We combine val_id (ID) and conflict_dev (OOD).
        def get_mixed_scores(get_score_fn):
            scores_val = get_score_fn("val_id")
            scores_dev = get_score_fn("conflict_dev")
            return np.concatenate([scores_val, scores_dev])
            
        y_true_mixed = np.concatenate([
            np.zeros(len(ds["val_id"]["y"]), dtype=int),
            np.ones(len(ds["conflict_dev"]["y"]), dtype=int)
        ])
        
        # Recompute anchors on the mixed set
        scores_a1 = get_mixed_scores(lambda split: (ds[split]["oracle_meta"]["c"] != ds[split]["y"]).astype(float))
        
        with torch.no_grad():
            scores_a2 = get_mixed_scores(lambda split: (
                biased_model(torch.tensor(ds[split]["X2"], dtype=torch.float32)).argmax(dim=-1).numpy() != ds[split]["y"]
            ).astype(float))
            
        auroc_a1, auprc_a1 = evaluate_metrics(y_true_mixed, scores_a1)
        auroc_a2, auprc_a2 = evaluate_metrics(y_true_mixed, scores_a2)
        
        # 6. Negative Controls
        # Shuffled evidence for A2
        np.random.shuffle(scores_a2)
        auroc_neg, _ = evaluate_metrics(y_true_mixed, scores_a2)
        
        # 7. Baselines (BAL-Q and C3)
        # BAL-Q (Q-only)
        balq_model, _ = train_bal_q(xs_train, y_train, xs_val, y_val, in_dims=in_dims, num_classes=num_classes, seed=seed, verbose=False)
        balq_model.set_quantile_thresholds(rel_train) # Pure Q-only
        
        # C3 EGSF-Real
        # For Gate 3 transfer, we use a fixed budget policy since we don't have oracle B_star in inference
        # Modality 1 (X2, Image) is restricted to 0.05 budget.
        b_policy = np.array([1.0, 0.05]) * 1.1 # kappa = 0.1
        c3_model, _ = train_c1_egsf(
            xs_train, y_train, xs_val, y_val,
            budget_bounds=b_policy, rel_train=rel_train, rel_val=rel_val,
            in_dims=in_dims, num_classes=num_classes, seed=seed, verbose=False
        )
        
        # Accuracies
        bf_conf = get_acc(bf_model, ds, "test_conflict")
        bf_id = get_acc(bf_model, ds, "test_id")
        balq_conf = get_acc(balq_model, ds, "test_conflict")
        balq_id = get_acc(balq_model, ds, "test_id")
        c3_conf = get_acc(c3_model, ds, "test_conflict")
        c3_id = get_acc(c3_model, ds, "test_id")
        
        metrics["a1_auroc"].append(auroc_a1); metrics["a1_auprc"].append(auprc_a1)
        metrics["a2_auroc"].append(auroc_a2); metrics["a2_auprc"].append(auprc_a2)
        metrics["bf_conf"].append(bf_conf); metrics["bf_id"].append(bf_id)
        metrics["balq_conf"].append(balq_conf); metrics["balq_id"].append(balq_id)
        metrics["c3_conf"].append(c3_conf); metrics["c3_id"].append(c3_id)
        metrics["a2_diff"].append(auroc_a2 - auroc_a1)
        
        print(f"  A1 Oracle AUROC: {auroc_a1:.4f}")
        print(f"  A2 Inferred AUROC: {auroc_a2:.4f}")
        print(f"  Neg-Control AUROC: {auroc_neg:.4f}")
        print(f"  BF Conflict Acc: {bf_conf:.4f} | ID Acc: {bf_id:.4f}")
        print(f"  BAL-Q Conflict Acc: {balq_conf:.4f} | ID Acc: {balq_id:.4f}")
        print(f"  C3 Conflict Acc: {c3_conf:.4f} | ID Acc: {c3_id:.4f}")

    print("\n" + "=" * 62)
    print("GATE 3 FINAL RESULTS")
    print("=" * 62)
    
    m = {k: np.mean(v) for k, v in metrics.items()}
    
    print(f"Mean A1 AUROC: {m['a1_auroc']:.4f} | AUPRC: {m['a1_auprc']:.4f}")
    print(f"Mean A2 AUROC: {m['a2_auroc']:.4f} | AUPRC: {m['a2_auprc']:.4f}")
    print(f"Mean A2 - A1 AUROC diff: {m['a2_diff']:+.4f}")
    print("-" * 62)
    print(f"Mean BF Conflict Acc: {m['bf_conf']:.4f} | ID Acc: {m['bf_id']:.4f}")
    print(f"Mean BAL-Q Conflict Acc: {m['balq_conf']:.4f} | ID Acc: {m['balq_id']:.4f}")
    print(f"Mean C3 Conflict Acc: {m['c3_conf']:.4f} | ID Acc: {m['c3_id']:.4f}")
    
    id_regression = m['bf_id'] - m['c3_id']
    conflict_improve = m['c3_conf'] - m['bf_conf']
    
    print("-" * 62)
    print(f"ID Regression: {id_regression * 100:.2f} percentage points (<= 5pp required)")
    print(f"Conflict Improvement over BF: +{conflict_improve * 100:.2f} pp")
    
    pass_a2 = (m['a2_diff'] >= -0.10)
    pass_conflict = (conflict_improve > 0)
    pass_id = (id_regression <= 0.05)
    
    print("\nCriteria Check:")
    print(f"  A2 AUROC >= A1 AUROC - 0.10: {'PASS' if pass_a2 else 'FAIL'}")
    print(f"  Conflict Benefit > 0:        {'PASS' if pass_conflict else 'FAIL'}")
    print(f"  ID Regression <= 5pp:        {'PASS' if pass_id else 'FAIL'}")
    
    if pass_a2 and pass_conflict and pass_id:
        print("\nGATE 3 FINAL CERTIFICATION: PASS")
    else:
        print("\nGATE 3 FINAL CERTIFICATION: FAIL")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run only seed 0")
    parser.add_argument("--dev", action="store_true", help="Run seeds 0-2 (dev experiment)")
    args = parser.parse_args()
    
    if args.smoke:
        seeds = [0]
    elif args.dev:
        seeds = [0, 1, 2]
    else:
        seeds = [0, 1, 2, 3, 4]
    run_experiment(seeds, dev_mode=(args.smoke or args.dev))

