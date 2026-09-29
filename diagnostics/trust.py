"""D2 Trust interval (Step 7). Owner: B.

T_m=(Q_m*(1-ECE_m)*(1-sigma_tilde_m))^{1/3}; B_{m,hi}=B_m+kappa*(1-T_m), kappa in {0.1,0.2,0.3}.
Choose kappa on validation only; report sensitivity.
"""
def trust_interval(B: float, Q: float, ece: float, sigma: float, kappa: float = 0.2) -> float:
    T = (max(0, Q) * max(0, 1-ece) * max(0, 1-sigma)) ** (1/3)
    return B + kappa * (1 - T)
