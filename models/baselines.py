"""Baselines BAL-U/G/Q/A (Step 4). Owner: A. Same encoders/splits/seeds as BF.

BAL-U: KL(alpha||uniform). BAL-G: OGM-GE-style slow dominant encoder (simplified unless official code).
BAL-Q: gate from input quality Q (SNR). BAL-A: correct when attribution share > thr.
Tune each on validation, document; never on conflict-test. Label re-implementations honestly.
"""
BAL = ["BAL-U", "BAL-G", "BAL-Q", "BAL-A"]
