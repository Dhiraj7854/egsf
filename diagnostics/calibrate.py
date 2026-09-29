"""D5 Calibrated diagnosis (Step 10). Owner: B.

Fit g (isotonic/Platt) on (H, ER_LCB, anchor, U) -> D ~= P(harmful|evidence).
Strict 3-way: cal-g fits g, cal-crc sets lambda* (Step 11), test evaluates. Never reuse.
Report ECE/Brier/reliability. Prefer Platt when n small; reweight if class imbalance.
"""
def fit_calibrator(*args, **kwargs):
    raise NotImplementedError("Step 10 task")
