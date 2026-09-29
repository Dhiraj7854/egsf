"""D3 Anchor + Harmfulness (Step 8). Owner: B.

anchor_m = clip(1-Delta^{conf}_m/(Delta_m+delta),0,1); H_m = anchor_m * ER_m.
Negative controls: random anchor (no help), wrong-modality (hurt), shuffled evidence (gain removed).
Keep conflict-dev disjoint from conflict-test. Inspect R5 if anchor flags legitimate features.
A2 (Step 14, inferred cues): biased predictor + loss-cluster envs + agreement/biased vs invariant.
"""
def anchor(delta: float, delta_conf: float, delta_eps: float = 1e-8) -> float:
    return min(1.0, max(0.0, 1.0 - delta_conf / (delta + delta_eps)))

def harmfulness(a: float, er: float) -> float:
    return a * er
