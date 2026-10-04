"""
egsf/data/jdb_r.py
──────────────────
JDB-R: Simulated Real Data Benchmark — EGSF v8.0, Gate 3.

Simulates the high-dimensional embeddings of a realistic multimodal dataset
(e.g., Colored-MNIST + Spoken-Digit).

Modality 0 (X1) - Audio / Spoken-Digit:
  Contains invariant class information (spoken digit) and noise.

Modality 1 (X2) - Image / Colored-MNIST:
  Contains invariant class information (shape) AND spurious cue (color).
  In training environments, color is highly correlated with the class.

The function returns the data, explicitly separating available features
from oracle metadata (to enforce A2 realism).
"""

import numpy as np

def generate_jdbr(
    n_train: int = 10000,
    n_val: int = 2000,
    n_dev: int = 2000,
    n_test: int = 2000,
    rho_corr: float = 0.95,
    seed: int = 0
):
    rng = np.random.default_rng(seed)
    K = 4   # 4 digits to match R2
    d = 8   # embedding dimension

    # Prototypes
    proto_y_x1 = rng.normal(0, 2.0, (K, d))  # Audio invariant signal
    # NOTE: No shape proto for X2 — it is a pure shortcut modality.
    # In train: X2 = proto_c_x2[y] + noise  (highly predictive via color)
    # In conflict: X2 = proto_c_x2[c_random] + noise  (zero MI with y)
    # This exactly mirrors JDB-S R2: sigma_causal large, sigma_cue small.
    proto_c_x2 = rng.normal(0, 2.0, (K, d))  # Image color (only cue signal)

    def make_split(n, cue_broken=False, env_idx=0):
        y = rng.integers(0, K, size=n)
        
        c = np.zeros_like(y)
        if cue_broken:
            c = rng.integers(0, K, size=n)
        else:
            mask = rng.random(size=n) < rho_corr
            c[mask] = y[mask]
            off_diag = rng.integers(1, K, size=np.sum(~mask))
            c[~mask] = (y[~mask] + off_diag) % K

        # X1 is Audio (Invariant). Very high noise → BF relies on X2 (like R2)
        X1 = proto_y_x1[y] + rng.normal(0, 5.0, (n, d))
        
        # X2 is Image (Color-only shortcut). Low noise in train; random in conflict
        X2 = proto_c_x2[c] + rng.normal(0, 0.3, (n, d))

        return {
            "X1": X1.astype(np.float32),
            "X2": X2.astype(np.float32),
            "y": y,
            "oracle_meta": {
                "c": c,
                "env_idx": env_idx,
                "cue_broken": cue_broken
            }
        }

    ds = {
        "train": make_split(n_train, cue_broken=False, env_idx=0),
        "val_id": make_split(n_val, cue_broken=False, env_idx=0),
        "cal_crc": make_split(n_val, cue_broken=False, env_idx=0),
        "test_id": make_split(n_test, cue_broken=False, env_idx=0),
        "conflict_dev": make_split(n_dev, cue_broken=True, env_idx=1),
        "test_conflict": make_split(n_test, cue_broken=True, env_idx=1),
        "ground_truth": {
            # B_star is the minimum information budget. In conflict_dev (cue_broken=True),
            # the correlation of X2 drops heavily, so its budget approaches 0.
            "B_star": [1.0, 0.05]
        }
    }
    
    return ds

if __name__ == "__main__":
    ds = generate_jdbr()
    print("JDB-R generated successfully.")
