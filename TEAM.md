# TEAM — parallel development plan (3 people)

Goal: 3 people can work in parallel without merge conflicts until Gate 1, then converge.

## Ownership (minimizes overlapping files)

| Person | Stream | Owns (CODEOWNERS) | First deliverable |
|--------|--------|-------------------|-------------------|
| A — Data & Base | Steps 0–4 | `data/`, `models/`, `configs/u_mod|bf|bal_*|bf_oracle.yaml`, `experiments/e9*` | JDB-S + U-Mod + BF + Gate 0 table |
| B — Diagnostics | Steps 5–10 | `diagnostics/`, `configs/d0*|d1*|d2*|d3*|d4*|d5*`, `experiments/e1|e2|e3*` | D0 validated → E1 ladder (Gate 1) |
| C — Decision/Eval/Theory | Steps 11–12 + eval | `decision/`, `eval/`, `common/`, `configs/c1*`, `experiments/e4|e5|e6*`, `docs/theory/` | CRC + projection gate + risk-coverage |

Shared (all read, one writer per PR): `notes/`, `tests/regression.py`, `configs/*.yaml` flag names.

`common/config.py` flag registry is the contract between streams — changing a flag name requires notifying all.

## Branch / milestone map

- Milestone `gate-0` (Oct 25): A ships BF failure; B stubs D0 API; C stubs CRC/gate API. Integration PR: `step/3-gate0-integration`.
- Milestone `gate-1` (Dec 31): B ships D1; A freezes generator version; C provides AUROC/FCR helpers. No new layers until Gate 1 passes.
- Milestone `gate-2` (Apr 15): C ships C1; B freezes D5 calibration; A runs baselines E9.
- Milestone `gate-3` (Jun 20): all on JDB-R + ablations.

Suggested initial branches (create on clone):
`step/0-repo-literature`, `step/1-jdb-s-generator`, `step/2-umod`, `step/3-bf`, `step/4-baselines`, `step/5-d0-reliance`

## Merge protocol

1. Rebase on `main`, run `make check` (pytest + ruff).
2. PR must link hypothesis-log date + paired-seed table (use PR template).
3. Owner of touched area reviews (see CODEOWNERS). No self-merge.
4. Tag ladder models on merge: `git tag -a D0 -m "D0 reliance meter, config hash ..."`.

## Weekly cadence (Part 8)

- Class weeks: 1 build block + 1 run/log block + short log review. Unit-test-sized tasks only.
- Holiday weeks: gate experiments + same-day write-up + mid-week gate checkpoint.
- Every 2 weeks: update plan; fix / cut / pivot decision if gate at risk.

## Cut list (if behind — Part 6 order)

1. C8, generative anchors, drift monitoring → 2. C7,C6 → 3. C5, group-FCR → 4. VQA-CP → 5. Tier-2 continuous (keep exact discrete PID). Never cut: Q-only test, negative controls, E4, split discipline.
