# Contributing — EGSF (group of 3)

## Feature-adding protocol (Part 4, mandatory)

1. **Hypothesis first.** One-page entry in `notes/hypothesis_log.md` dated _before_ coding: what improves, metric, margin.
2. **Flag it.** New feature behind config switch defaulting to `off` so previous model reproduces exactly.
3. **Regression with flag off.** `pytest -q` must match previous tagged model reference numbers.
4. **Paired comparison.** Same seeds + splits, feature-on vs off; paired-bootstrap CI via `eval/stats.py`.
5. **Decide.** Keep iff margin met; else revert/park + entry in `notes/negative_results_log.md`.
6. **Tag.** `git tag D0|D1|...|C4`, store config + results table under `logs/`.
7. **Never tune on test.** Hyperparams from val / calibration splits only. See `notes/splits.md`.

## Branching

- `main` is protected (CI must pass, 1 review).
- Branches: `step/<step>-<owner>-<short>` e.g. `step/1-jdb-s-generator-aisha`.
- Never commit directly to `main`. PR template enforces: hypothesis link, paired table, split discipline checkbox.

## Definition of done (per Step)

Goal/Build/How/Check in PDF Part 3. Check must be demonstrated with a command + logged table hash.

## Style

- `ruff check .`, `pytest -q` before every push (pre-commit does it).
- Configs are the source of truth; no hardcoded hyperparams in `.py`.
- Seeds fixed everywhere via `common/seeds.py`.
