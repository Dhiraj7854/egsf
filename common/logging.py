"""Certificate logging schema (Part 7): one row per instance per modality.

Columns:
 identity: instance_id, split, environment, seed, model_tag, config_hash
 reliance+evidence: rho, delta, S_replicates, Q, R, C, T, B_lo, B_hi
 diagnosis: ER, ER_LCB, anchor_tier, anchor, U, H, D
 decision: lambda_star, alpha_risk, flagged, abstain, alpha_before, alpha_after
 ground_truth (synthetic only): regime, planted_mechanism, B_star, label
"""
from __future__ import annotations
CERT_COLUMNS = [
 "instance_id","split","environment","seed","model_tag","config_hash",
 "modality","rho","delta","Q","R","C","T","B_lo","B_hi",
 "ER","ER_LCB","anchor_tier","anchor","U","H","D",
 "lambda_star","alpha_risk","flagged","abstain","alpha_before","alpha_after",
 "regime","planted_mechanism","B_star","label",
]

def new_certificate_row(**kw) -> dict:
    row = {k: None for k in CERT_COLUMNS}
    row.update(kw)
    return row
