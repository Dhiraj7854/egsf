"""JDB-S generator (Step 1). Owner: A.

Two versions: discrete (exact PID) + continuous (embeddings for MLPs).
Regimes R1..R5, rho_corr in {0.6,0.7,0.8,0.9}. Seeded + versioned.
TODO(A): implement SCM; unit tests in tests/test_generator.py must pass:
 - empirical cue correlation matches rho_corr
 - PID parts sum to total; XOR -> pure synergy; copy-of-X1 -> pure redundancy
"""
VERSION = "jdb-s-v0"
REGIMES = ["R1","R2","R3","R4","R5"]
RHO_CORR_LEVELS = [0.6, 0.7, 0.8, 0.9]

def generate(*, n: int, K: int = 4, regime: str = "R2", rho_corr: float = 0.9,
             seed: int = 0, continuous: bool = True):
    raise NotImplementedError("Step 1 task — see tests/test_generator.py for checks")
