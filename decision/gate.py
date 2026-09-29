"""Projection gate (Step 11). Owner: C.

For flagged (D_m>lambda*): alpha'=argmin_q KL(q||alpha) s.t. q_m<=max(eps,B_{m,hi}+s).
Closed form: clip to caps + rescale rest; verify vs numerical solver. Sums to 1, respects caps/floor, identity if none flagged.
If corrected rho still > cap (weak alpha-rho link): feature-level correction / selective balance-loss fine-tune.
"""
import numpy as np
def project(alpha: np.ndarray, caps: dict, flagged: dict, eps: float = 1e-3) -> np.ndarray:
    q = alpha.copy().astype(float)
    for m, is_f in flagged.items():
        if is_f: q[m] = min(q[m], caps.get(m, 1.0))
    q = np.maximum(q, eps); return q / q.sum()
