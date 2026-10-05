# EGSF Model 1 — Corrected BaseFusion Baseline

## 1. Overview

Model 1 is the frozen baseline model used in the EGSF research project.

The purpose of Model 1 is to provide a fixed reference point for evaluating the future explanation-guided and selective-fusion methods.

Model 1 is a corrected Gate-1 BaseFusion baseline trained on the corrected D1 experimental dataset.

---

## 2. Model Architecture

Model 1 uses two modality-specific encoders followed by a learned fusion gate.

### Modality 1 encoder

- Input dimension: 16
- Hidden dimension: 64
- Two fully connected layers
- ReLU activation

### Modality 2 encoder

- Input dimension: 16
- Hidden dimension: 64
- Two fully connected layers
- ReLU activation

### Fusion gate

The two encoded representations are concatenated and passed through a gate.

The gate produces two logits, which are converted into modality weights using softmax.

The fused representation is:

alpha_1 * h_1 + alpha_2 * h_2

where:

- h_1 = representation from modality 1
- h_2 = representation from modality 2
- alpha_1 + alpha_2 = 1

### Classifier

- Input dimension: 64
- Output classes: 2

### Total trainable parameters

**19,012**

---

## 3. Training Configuration

| Setting | Value |
|---|---|
| Model | Corrected BaseFusion |
| Input dimension | 16 + 16 |
| Hidden dimension | 64 |
| Number of classes | 2 |
| Epochs | 12 |
| Batch size | 256 |
| Learning rate | 0.001 |
| Correlation settings | 0.6, 0.7, 0.8, 0.9 |
| Seeds | 42, 43, 44, 45, 46 |
| Total configurations | 20 |

Therefore:

4 correlation settings × 5 seeds = **20 configurations**

All 20 configurations were successfully trained and audited.

---

## 4. Validation Performance

The corrected Model 1 achieved:

- Mean validation accuracy: **0.7789**
- Minimum validation accuracy: **0.6870**
- Maximum validation accuracy: **0.8188**
- Configurations evaluated: **20/20**

The model architecture audit passed with exactly **19,012 parameters**.

---

## 5. Gate 0 Result

Gate 0 checks whether the controlled experimental setup behaves as expected.

At correlation setting rho = 0.9:

- Mean conflict drop: **14.5767 percentage points**
- Required threshold: **10 percentage points**
- All five seeds exceeded the threshold.

### Gate 0 status

**PASS**

This confirmed that the controlled experimental setup was suitable for continuing the research pipeline.

---

## 6. Gate 1 Evaluation

Gate 1 evaluates whether the explanation/reliance diagnostic provides a sufficiently strong improvement over the quality-only comparison.

The independent SCM evaluation was used to avoid circular evaluation.

Only R5 provided defined AUROC values because R1-R4 have structural one-class labels for the corresponding SCM evaluation.

For R5:

### ER versus Q-only

Mean paired difference:

**+0.023456**

95% confidence interval:

**[0.007247, 0.043937]**

Number of paired seeds:

**5**

Predefined required margin:

**0.05**

Since:

0.023456 < 0.05

the predefined margin was not reached.

### Gate 1 status

**NOT YET PASS**

The detailed status was:

**POSITIVE_BUT_MARGIN_FAIL**

This means the observed effect was positive, but it was smaller than the predefined required margin.

---

## 7. Protocol Integrity

The final Model 1 evaluation maintained the predefined evaluation protocol.

- Independent SCM labels: YES
- Labels dependent on model-derived reliance: NO
- Labels dependent on ER: NO
- Labels dependent on QER: NO
- Final-test tuning: NO
- Threshold changed after evaluation: NO
- Models retrained after evaluation: NO
- Bootstrap margin changed: NO
- R5 removed: NO

---

## 8. Model Freeze

After the Gate 1 evaluation, Model 1 was frozen.

No post-evaluation retraining was performed.

The Gate 1 threshold was not changed after observing the result.

Therefore, Model 1 remains a fixed baseline/reference model for subsequent EGSF experiments.

### Model status

**FROZEN**

---

## 9. Why Model 1 Is Important

Model 1 provides the baseline against which future EGSF components can be compared.

The goal is not to force both modalities to contribute equally.

The research goal is to determine whether modality contribution is justified by the information available from each modality.

The future EGSF pipeline will therefore build on the frozen Model 1 baseline rather than modifying it after seeing the Gate 1 result.

---

## 10. Future Research Direction

The planned research progression is:

Model 1 — Frozen BaseFusion
↓
Modality reliance diagnosis
↓
Explanation-guided analysis
↓
Selective intervention
↓
Improved EGSF model
↓
Comparison against frozen Model 1

Future models should be evaluated relative to the frozen Model 1 baseline.

---

## 11. Checkpoint

The trained Model 1 checkpoint is stored in:

`models/EGSF_Model1_Gate1_BaseFusion_20models.pt`

The checkpoint contains the 20 trained BaseFusion models corresponding to:

- rho = 0.6, seeds 42–46
- rho = 0.7, seeds 42–46
- rho = 0.8, seeds 42–46
- rho = 0.9, seeds 42–46

The checkpoint status is:

**FROZEN**

---

## 12. Final Summary

| Item | Result |
|---|---|
| Model | Corrected BaseFusion |
| Parameters | 19,012 |
| Configurations | 20/20 |
| Mean validation accuracy | 0.7789 |
| Minimum validation accuracy | 0.6870 |
| Maximum validation accuracy | 0.8188 |
| Gate 0 | PASS |
| Gate 1 | NOT YET PASS |
| ER − Q-only | +0.023456 |
| 95% CI | [0.007247, 0.043937] |
| Required margin | 0.05 |
| Final-test tuning | NO |
| Post-evaluation retraining | NO |
| Threshold changed | NO |
| Model status | FROZEN |

---

## 13. Research Interpretation

Model 1 was successfully implemented, trained, evaluated, audited, and frozen.

The Gate 1 diagnostic produced a positive effect, but the effect did not reach the predefined margin of 0.05.

Therefore, the result should not be described as a failure of the model.

Instead, it indicates that the current diagnostic signal is promising but not yet strong enough to satisfy the Gate 1 criterion.

This provides a scientifically controlled starting point for the next EGSF research stage.
