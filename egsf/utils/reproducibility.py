"""
egsf/utils/reproducibility.py
──────────────────────────────
Global reproducibility utilities.
Import seed_everything() at the top of every experiment script.
"""

import os
import random
import numpy as np
import torch
import yaml
import hashlib
import json
from pathlib import Path


def seed_everything(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops (may slow down GPU; acceptable for our small models)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return as dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    """Compute a short SHA-256 hash of a config dict for reproducibility logs."""
    serialized = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialized).hexdigest()[:12]


def experiment_id(model_tag: str, seed: int, cfg_hash: str) -> str:
    """Canonical experiment identifier string."""
    return f"{model_tag}_s{seed}_{cfg_hash}"


if __name__ == "__main__":
    # Quick self-test
    seed_everything(42)
    cfg = load_config(Path(__file__).parents[2] / "configs" / "jdb_s.yaml")
    h = config_hash(cfg)
    print(f"Config hash: {h}")
    print(f"Example experiment id: {experiment_id('bf-v1', 0, h)}")
    print("reproducibility.py OK")
