"""D0 Reliance Meter (Step 5). Owner: B. Post-hoc on frozen BF.

Delta_m(x)=max(0, logp(y|x)-E_{x'm}[logp(y|x_{-m},x'_m)]), rho=Delta/sum+delta.
Resample x'm from batch (S=4-8); cross-check GradxInput + 1-step IG.
Validate: monotone in rho_corr; vs planted ablation GT; resample vs zero-mask; corr(rho,alpha).
Warns Step 11 if corr(rho,alpha) weak -> feature-level correction needed.
"""
def reliance(*args, S: int = 6, **kwargs):
    raise NotImplementedError("Step 5 task")
