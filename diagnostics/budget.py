"""Tier-1 invariant budget (Step 6, Gate 1). Owner: B.

b_m^e = I_m^e/sum_k I_k^e, B_m = min_e b_m^e. Compare vs exact SCM B*_m for budget error.
Puts rho and B on same simplex. Env splits disjoint from final test.
"""
def tier1_budget(info_per_env: dict) -> dict:
    import numpy as np
    envs = list(info_per_env)
    mods = list(info_per_env[envs[0]])
    out = {}
    for m in mods:
        out[m] = min(info_per_env[e][m] / max(1e-12, sum(info_per_env[e].values())) for e in envs)
    return out
