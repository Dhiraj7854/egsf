## Hypothesis
Link to `notes/hypothesis_log.md` entry (date BEFORE coding): <!-- -->
Metric + pre-registered margin: <!-- e.g. AUROC gain ≥0.05, CI excl. 0, 5 paired seeds -->

## Config flag
Flag name (defaults to off): `...`
Previous model reproduces with flag off? [ ] yes (`pytest -q` output pasted)

## Paired comparison (same seeds/splits)
| model | ID acc | conflict acc | AUROC | seeds |
|---|---|---|---|---|
| off | | | | |
| on  | | | | |
CI (paired bootstrap): <!-- -->

## Split discipline
- [ ] No tuning on conflict-test / final-test
- [ ] Cal-G / Cal-CRC / test disjoint (assert passes)
- [ ] Negative controls run (where applicable)

## Checklist
- [ ] `ruff` + `pytest` pass
- [ ] Results table + config hash saved to `logs/`
- [ ] Negative result appended if margin missed
