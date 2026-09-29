"""U-Mod unimodal predictors (Step 2). Owner: A.

Small MLP per modality; (a) per-environment preferred, (b) pooled-train fallback.
Temperature scaling on held-out part of each env. Report acc/CE/ECE per mod per env.
Info proxy I_m^e = max(0, logK - CE_cal^e) -> budget shares.
"""
def train_unimodal(*args, **kwargs):
    raise NotImplementedError("Step 2 task")
