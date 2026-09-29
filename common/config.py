"""Config system: one YAML per ladder model, flags default off.

Contract (Part 4): every new feature behind `features.<name>` defaulting to false.
`config_hash()` is stored with every results table and certificate log.
"""
from __future__ import annotations
import hashlib, yaml

def load(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def config_hash(cfg: dict) -> str:
    blob = yaml.safe_dump(cfg, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
