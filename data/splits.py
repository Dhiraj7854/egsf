"""Split discipline (Part 4). Owner: A, enforced for all.

Train (cue correlated) -> train only. Val-ID -> early stop/temp-scale/hparams.
Conflict-dev (cue-broken) -> budgets, anchor A1, kappa/z/s tuning. NEVER final results.
Cal-G (mixed labeled) -> fit g. Cal-CRC (justified only) -> set lambda*. NEVER fit g.
Test-ID/Test-conflict -> final numbers, touched once per frozen model.
"""
SPLITS = ["train","val-id","conflict-dev","cal-g","cal-crc","test-id","test-conflict"]

def assert_disjoint(*id_sets) -> None:
    seen = set()
    for s in id_sets:
        overlap = seen & set(s)
        assert not overlap, f"split leakage: {len(overlap)} overlapping ids"
        seen |= set(s)
