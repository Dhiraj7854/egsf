"""Invariance test: budget ~0 for modality whose label link destroyed in one env."""
from diagnostics.budget import tier1_budget
def test_invariance():
    b = tier1_budget({"e_train": {"m1": 1.0, "m2": 1.0}, "e_broken": {"m1": 1.0, "m2": 0.0}})
    assert b["m2"] < 0.35
