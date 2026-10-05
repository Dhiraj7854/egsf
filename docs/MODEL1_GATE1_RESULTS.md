# Model 1 — Gate-1 Evaluation Results

## 1. Purpose

Gate 1 evaluates whether the explanation/reliance diagnostic provides a sufficiently strong improvement over the quality-only comparison.

The evaluation was performed after Model 1 training and was kept independent from the final-test tuning process.

---

## 2. Gate-1 Decision

### Final status

**NOT YET PASS**

### Detailed status

**POSITIVE_BUT_MARGIN_FAIL**

The observed effect was positive, but it did not reach the predefined required margin.

---

## 3. Main Gate-1 Result

The independent SCM evaluation compared:

- ER: Explanation/Reliance-based measure
- Q-only: Quality-only measure

For the structurally evaluable R5 condition:

**Mean paired ER − Q-only difference:**

+0.023456

**95% confidence interval:**

[0.007247, 0.043937]

**Number of paired seeds:**

5

**Predefined required margin:**

0.05

---

## 4. Margin Check

The Gate-1 criterion required:

ER − Q-only >= 0.05

Observed:

ER − Q-only = 0.023456

Therefore:

0.023456 < 0.05

The predefined margin was not reached.

---

## 5. Confidence Interval

The 95% confidence interval was:

[0.007247, 0.043937]

The confidence interval excludes zero.

This indicates that the observed difference is positive in the evaluated paired-seed analysis.

However, the entire interval remains below the required margin of 0.05.

Therefore, the result is statistically positive but does not satisfy the predefined practical/evidence margin.

---

## 6. Interpretation

The Gate-1 result should not be interpreted as saying that Model 1 failed.

Model 1 itself was successfully:

- implemented,
- trained,
- evaluated,
- audited,
- and frozen.

Instead, the result means that the current diagnostic signal was promising but not strong enough to satisfy the predefined Gate-1 criterion.

This provides evidence for continuing the EGSF research with improved diagnostic and selective-intervention methods.

---

## 7. Independent SCM Evaluation

The evaluation used independent SCM-based labels rather than labels generated from the model's own reliance scores.

This was done to reduce circularity in the evaluation.

Protocol checks confirmed:

- Independent SCM labels: YES
- Labels dependent on model-derived reliance: NO
- Labels dependent on ER: NO
- Labels dependent on QER: NO

Only R5 provided defined AUROC comparisons because R1-R4 had structural one-class labels for the corresponding SCM evaluation.

---

## 8. R5 ER vs Q-only Differences

| rho | ER-Q-only difference |
|---|---:|
| 0.6 | +0.080512 |
| 0.7 | +0.070856 |
| 0.8 | +0.032746 |
| 0.9 | -0.027039 |

The paired-seed bootstrap aggregation across the five seeds produced:

**Mean difference = +0.023456**

with:

**95% CI = [0.007247, 0.043937]**

---

## 9. Protocol Integrity

The following protocol conditions were verified:

| Check | Result |
|---|---|
| Independent SCM labels | PASS |
| Labels depend on model reliance | NO |
| Labels depend on ER | NO |
| Labels depend on QER | NO |
| Final-test tuning | NO |
| Threshold changed after evaluation | NO |
| Models retrained after evaluation | NO |
| Bootstrap margin changed | NO |
| R5 removed | NO |

---

## 10. Model Freeze

After Gate-1 evaluation:

- No post-evaluation retraining was performed.
- The Gate-1 threshold was not changed.
- The evaluation criterion was not modified.
- Model 1 remained frozen.

### Model status

**FROZEN**

---

## 11. Final Gate-1 Conclusion

Gate 1 produced a positive diagnostic signal:

**ER − Q-only = +0.023456**

However, the predefined required margin was:

**0.05**

Because:

**0.023456 < 0.05**

the Gate-1 criterion was not satisfied.

### Final result

**GATE 1: NOT YET PASS**

**Status: POSITIVE_BUT_MARGIN_FAIL**

The frozen Model 1 baseline will therefore serve as the reference point for the next EGSF research stage.
