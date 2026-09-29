# Splits (Part 4) — each split has exactly one job

| split | env | used for | MUST NOT |
|---|---|---|---|
| train | train (cue corr) | BF, U-Mod, baselines | any cal/eval |
| val-id | train distr | early stop, temp scale, hparams | final results |
| conflict-dev | cue-broken | budgets, A1, kappa/z/s | final results |
| cal-g | mixed labeled | fit g | set lambda* |
| cal-crc | justified only | set lambda* | fit g / tuning |
| test-id/test-conflict | both | final numbers, once per frozen model | any tuning |

Enforce via `data/splits.py::assert_disjoint` in every experiment.
