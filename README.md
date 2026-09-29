# EGSF — Explanation-Guided Selective Fusion (v8.0)

> Plan only; no result here has been measured. This repo implements the **EGSF v8.0 Complete Execution Plan** step-by-step.

From first base model → full certified system: model ladder, gates, contingencies.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp configs/bf.yaml configs/local.yaml  # never commit local.yaml
python -m experiments.e1_ladder --config configs/d1_budget.yaml --seeds 0 1 2 3 4
pytest -q
```

## Model ladder (source: Part 2)

| # | Tag | Kind | File |
|---|-----|------|------|
| 1 | U-Mod | Base | `models/unimodal.py`, `configs/u_mod.yaml` |
| 2 | BF | Base fusion | `models/fusion.py`, `configs/bf.yaml` |
| 3 | BF-Oracle | Baseline | `models/oracle.py` |
| 4-7 | BAL-U/G/Q/A | Baselines | `models/baselines.py` |
| 8-13 | D0–D5 | Diagnostics (post-hoc, reuse BF) | `diagnostics/` |
| 14 | C1 EGSF-Core | Core | `decision/` |
| 15-17 | C2/C3/C4 | Extension | PID / Real / Deploy |
| 18-21 | C5–C8 | Optional/stretch | cut first if behind |

Priority: 1–14 = the project. 15–17 = publishable on real data. 18–21 = optional.

## Gates (source: Part 6)

- **Gate 0 (Step 3):** ID acc high, conflict acc drops ≥10pts at ρ_corr=0.9, α_cue rises.
- **Gate 1 (Step 6):** ER beats Q-only by ≥0.05 AUROC in ≥3/5 regimes incl. R2,R5, no regression R1/R3.
- **Gate 2 (Step 11):** FCR ≤ α over ≥1000 splits; conflict acc up; justified preserved.
- **Gate 3 (Step 14):** A2 near A1; frozen JDB-S→JDB-R transfer beats baselines.

> If Gate 1 fails after 2 fix iterations: stop, write negative result, pivot. Do not add layers to rescue failing diagnosis.

## Seven rules (Part 1)

1. One change at a time, behind a config flag defaulting to off.
2. Pre-register success (metric + margin) before running.
3. Diagnose first (D0–D5 post-hoc), correct second (C1).
4. Freeze before testing; never tune on conflict-test / final-test.
5. Keep negative results log.
6. Every model reproducible: git tag + config + seeds + results table.
7. Stop adding layers when a gate fails.

## Repo layout (Part 7)

```
configs/      # one YAML per ladder model
data/         # jdb_s (discrete+continuous), jdb_r, splits.py
models/       # encoders, fusion, unimodal, baselines, oracle
diagnostics/  # reliance, budget, excess, trust(→interval), anchor, uncertainty, calibrate
decision/     # crc.py, gate.py, abstain.py
eval/         # metrics.py, stats.py
common/       # config.py, seeds.py, logging.py (cert schema)
tests/        # unit + regression suite
experiments/  # e1..e9, each reads config, writes results table
logs/         # per-instance certificate CSV/Parquet (gitignored, keep .gitkeep)
notes/        # hypothesis log, negative log, literature memo
```

## Splits — each split has exactly one job (Part 4)

See `notes/splits.md` and `data/splits.py`. Enforced by disjoint-ID asserts in code.

## Parallel dev (3 people)

See `TEAM.md` + `CONTRIBUTING.md`. TL;DR: `main` protected; work on `step/<n>-<owner>` branches; PR + paired-seed table + CI required.

## Certificate log schema

See `common/logging.py` — one row per instance per modality.
