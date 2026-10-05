# EGSF Model 1 — Frozen BaseFusion Baseline

This document describes the complete development, evaluation, validation, and freezing of **Model 1** in the EGSF research project.

Model 1 is the **corrected Gate-1 BaseFusion baseline**. It is not the final EGSF model. It is a fixed reference model that will be used to evaluate the future explanation-guided and selective-fusion methods.

---

## 1. Research Idea

The research studies **multimodal deep learning**, where a model receives information from more than one modality.

The main research question is:

> Does a multimodal model rely on each modality in a way that is justified by the information that modality actually provides?

A model can achieve good prediction accuracy while still relying too strongly on an easy or shortcut modality.

Therefore, the research does not only ask:

> How accurate is the model?

It also asks:

> Why is the model relying on a particular modality?

and:

> Is that reliance justified by the available information?

This motivates the EGSF research direction.

---

## 2. EGSF — Explanation-Guided Selective Fusion

**EGSF** stands for:

> **Explanation-Guided Selective Fusion**

The overall research idea is:

```text
Train a multimodal model
        ↓
Measure modality reliance
        ↓
Compare reliance with available information
        ↓
Identify excessive or unjustified reliance
        ↓
Explain the model behaviour
        ↓
Apply selective intervention
        ↓
Evaluate the improved fusion model
```

The important principle is:

> **I do not want to force both modalities to contribute equally. I want the model's modality contribution to be justified by the information that the modality actually provides.**

Model 1 is the first major frozen baseline for this research direction.

---

## 3. Model 1 Architecture

Model 1 uses a **BaseFusion** architecture.

Each modality has its own encoder. The encoded representations are then combined using a learned gate.

```text
              Modality 1                 Modality 2
              16 features                16 features
                   │                         │
                   ▼                         ▼
              Encoder 1                 Encoder 2
                   │                         │
                   ▼                         ▼
                h1 (64)                  h2 (64)
                   │                         │
                   └──────────┬──────────────┘
                              │
                              ▼
                         Concatenate
                              │
                              ▼
                            Gate
                              │
                              ▼
                          α1, α2
                              │
                              ▼
                      Weighted Fusion
                              │
                              ▼
                         Classifier
                              │
                              ▼
                          2 Classes
```

The two modality encoders transform the original 16-dimensional inputs into 64-dimensional representations.

The gate receives the concatenated representations and produces two fusion weights:

```text
α1 = weight for modality 1
α2 = weight for modality 2
```

The weighted representations are then fused and passed to the classifier.

### Architecture Configuration

| Component | Configuration |
|---|---:|
| Modality 1 input | 16 |
| Modality 2 input | 16 |
| Encoder hidden dimension | 64 |
| Gate hidden dimension | 64 |
| Number of classes | 2 |
| Total parameters | 19,012 |

---

## 4. Research Dataset and Experimental Setup

Model 1 was evaluated using the corrected D1 experimental dataset.

The main experimental variables were:

- Correlation levels:
  - 0.6
  - 0.7
  - 0.8
  - 0.9
- Random seeds:
  - 42
  - 43
  - 44
  - 45
  - 46

Therefore:

```text
4 correlation settings × 5 random seeds = 20 configurations
```

The same experimental structure was maintained across the configurations so that differences between results could be interpreted consistently.

---

## 5. Why Multiple Correlation Levels Were Used

The correlation parameter controls how strongly the experimental modalities are related to the underlying prediction structure.

Using multiple correlation levels allows the model to be tested under different controlled conditions.

The four settings were:

```text
rho = 0.6
rho = 0.7
rho = 0.8
rho = 0.9
```

This prevents the baseline from being evaluated under only one artificial condition.

A stronger correlation does not automatically mean that one modality should receive more attention. The purpose is to study how the model behaves as the relationship between modalities changes.

---

## 6. Why Five Random Seeds Were Used

A single random seed can produce a result that is unusually good or unusually poor.

Therefore, Model 1 was trained using five independent seeds:

```text
42
43
44
45
46
```

This gives:

```text
20 total experiments
= 4 correlation levels × 5 seeds
```

The use of multiple seeds makes the reported baseline more reliable than relying on one training run.

---

## 7. Gate 0: Controlled Experimental Validation

Before evaluating Model 1, the controlled experimental environment was checked using Gate 0.

Gate 0 was designed to verify that the experimental setup could produce the intended conflict behaviour.

The important criterion was the conflict reduction at:

```text
rho = 0.9
```

The observed mean conflict reduction was:

```text
14.5767 percentage points
```

The required threshold was:

```text
10 percentage points
```

Therefore:

```text
14.5767 > 10
```

and all five seeds passed the required condition.

### Gate 0 Result

```text
GATE 0 = PASS
```

This gave confidence that the controlled experimental environment was behaving as intended before proceeding with the corrected Model 1 baseline.

---

## 8. An Important Dataset Correction

During the research process, an issue was discovered in the original D1 dataset generation procedure.

The original procedure created a new generator separately for different split/environment combinations.

This could result in different underlying prototypes or structural relationships being used across parts of the dataset.

That would make comparisons less reliable.

Therefore, the D1 generation procedure was corrected.

### Corrected approach

For each `(rho, seed)` configuration:

```text
Create ONE generator
        ↓
Reuse the same generator
        ↓
Train split
Validation split
Test environments
Different regimes
```

This ensured that the structural relationships remained consistent across the relevant dataset components.

---

## 9. Why the Dataset Was Regenerated

Because the dataset-generation issue could affect downstream experiments, the affected D1 pipeline was corrected and regenerated.

The corrected datasets were then audited for:

- train/validation alignment
- regime consistency
- numerical validity
- structural consistency
- provenance
- final-test protection

The corrected D1 pipeline passed the required checks.

The important principle was:

> Once a data-generation issue was discovered, affected downstream experiments were rerun rather than continuing with potentially inconsistent data.

---

## 10. Model 1 Training

After the corrected D1 dataset was available, Model 1 was trained again from scratch for all 20 configurations.

The training configuration was:

| Setting | Value |
|---|---:|
| Model | Corrected BaseFusion |
| Modality 1 input | 16 |
| Modality 2 input | 16 |
| Hidden dimension | 64 |
| Classes | 2 |
| Epochs | 12 |
| Batch size | 256 |
| Learning rate | 0.001 |
| Correlations | 0.6, 0.7, 0.8, 0.9 |
| Seeds | 42, 43, 44, 45, 46 |
| Total configurations | 20 |

The resulting model contains:

```text
19,012 trainable parameters
```

---

## 11. Model 1 Validation Results

The final validation results for all 20 configurations were:

| rho | Seed | Validation Accuracy |
|---:|---:|---:|
| 0.6 | 42 | 0.6995 |
| 0.6 | 43 | 0.7762 |
| 0.6 | 44 | 0.8187 |
| 0.6 | 45 | 0.7943 |
| 0.6 | 46 | 0.8003 |
| 0.7 | 42 | 0.6870 |
| 0.7 | 43 | 0.7790 |
| 0.7 | 44 | 0.8183 |
| 0.7 | 45 | 0.7752 |
| 0.7 | 46 | 0.7985 |
| 0.8 | 42 | 0.6945 |
| 0.8 | 43 | 0.7802 |
| 0.8 | 44 | 0.8170 |
| 0.8 | 45 | 0.7975 |
| 0.8 | 46 | 0.8017 |
| 0.9 | 42 | 0.7242 |
| 0.9 | 43 | 0.7857 |
| 0.9 | 44 | 0.8157 |
| 0.9 | 45 | 0.8060 |
| 0.9 | 46 | 0.8085 |

Overall validation performance:

```text
Mean = 0.7789
Minimum = 0.6870
Maximum = 0.8188
```

Therefore, all 20 configurations successfully produced valid Model 1 results.

---

## 12. Model 1 Architecture Audit

After training, the architecture and checkpoints were audited.

The expected parameter count was:

```text
19,012 parameters
```

The actual parameter count matched the expected value.

The audit also checked:

- model coverage
- checkpoint consistency
- validation results
- loss consistency
- accuracy consistency
- gate outputs
- numerical validity
- provenance
- final-test protection

The architecture audit passed.

```text
Model 1 Architecture Audit = PASS
```

---

## 13. Modality Reliance Analysis

After training the baseline model, the next question was:

> Which modality does the trained model actually rely on?

The model contains a learned gate that produces two weights:

```text
alpha_1
alpha_2
```

These represent the model's learned preference between the two modality representations during fusion.

However, the research does not assume that a larger gate weight automatically means that the modality is genuinely justified.

Therefore, intervention-based reliance was also measured.

The purpose was to compare:

```text
What the model uses
        versus
What information the modality actually provides
```

This distinction is central to the EGSF research direction.

---

## 14. Information Budget

An information budget was calculated for the modalities.

The budget represents the amount of relevant information that the modality can provide under the controlled experimental setting.

The corrected information-budget audit contained:

```text
D1_CORRECTED_BUDGET_ENVS_DF = 160 rows
D1_CORRECTED_BUDGET_BY_SEED = 40 rows
```

The budget calculations were also checked for:

- zero-information rows
- share-sum consistency
- numerical validity
- per-seed consistency

The zero-information row count was:

```text
0
```

The information-share sum error was:

```text
0
```

This confirmed that the corrected budget construction was numerically consistent.

---

## 15. Explanation-Based Reliance

The next step was to compare model reliance against the available information budget.

A useful diagnostic idea is:

```text
Excess Reliance ≈ Model Reliance − Available Information
```

The goal is not simply to determine whether a modality is important.

The deeper question is:

> Is the model relying on a modality more than the available information justifies?

This is the diagnostic motivation behind EGSF.

The analysis produced both:

- ER-based measurements
- Q-only measurements

where Q-only represents a quality-based comparison without the full information-budget justification.

---

## 16. Why ER Was Compared With Q-Only

Quality alone does not necessarily explain whether a modality deserves the model's reliance.

For example:

```text
High quality
≠
High justified information contribution
```

Therefore, the research compares:

```text
ER-based explanation
versus
Q-only explanation
```

The intention is to determine whether information-aware diagnostics provide additional explanatory value beyond simple quality weighting.

This is important because EGSF is intended to be diagnostic-first rather than simply another quality-weighting method.

---

## 17. Independent SCM Evaluation

A major concern in explanation research is circular evaluation.

If the same model's outputs are used to create the labels that are then used to evaluate the model's explanation, the evaluation can become self-referential.

To reduce this problem, Model 1 used an independent structural causal model (SCM) reference for the final Gate-1 comparison.

The independent SCM evaluation used planted structural information rather than relying on the model's own reliance values to define the evaluation target.

The protocol explicitly checked that:

```text
Independent SCM labels = YES

Labels depend on model-derived reliance = NO

Labels depend on ER = NO

Labels depend on QER = NO
```

This provides an independent reference for evaluating the diagnostic behaviour.

---

## 18. Why Only R5 Produced a Valid AUROC Comparison

The SCM evaluation contained several experimental regimes.

However, not every regime contained both positive and negative classes required for a meaningful AUROC calculation.

Several regimes were structurally one-class under the relevant planted labels.

Therefore, a valid AUROC comparison was available only for:

```text
R5
```

This was not treated as a reason to artificially modify the labels or force an AUROC calculation where the required class structure did not exist.

The structural limitation was preserved in the evaluation.

---

## 19. Gate-1 Statistical Result

For R5, the comparison between ER and Q-only produced the following differences:

| rho | ER − Q-only |
|---:|---:|
| 0.6 | +0.080512 |
| 0.7 | +0.070856 |
| 0.8 | +0.032746 |
| 0.9 | -0.027039 |

Across the paired-seed evaluation, the overall mean difference was:

```text
ER − Q-only = +0.023456
```

The 95% confidence interval was:

```text
[0.007247, 0.043937]
```

Number of paired seeds:

```text
n = 5
```

The preregistered required margin was:

```text
0.05
```

Therefore:

```text
Observed difference = 0.023456
Required margin     = 0.05

0.023456 < 0.05
```

The confidence interval excluded zero, meaning the observed effect was positive.

However, the effect did not reach the required preregistered margin.

---

## 20. Gate-1 Final Decision

The Gate-1 decision was therefore:

```text
Gate 1 = NOT YET PASS
```

More specifically:

```text
Status = POSITIVE_BUT_MARGIN_FAIL
```

This means:

```text
Positive effect observed
        +
Confidence interval excludes zero
        +
Required margin not reached
        ↓
Gate-1 criterion not passed
```

This result is scientifically useful because it provides evidence of a positive direction without overstating the strength of the result.

---

## 21. Protocol Integrity

After the Gate-1 evaluation, the research protocol was checked to ensure that the result had not been manipulated after seeing the outcome.

The final audit confirmed:

```text
Final-test tuning                 = NO
Threshold changed after result    = NO
Labels modified after evaluation  = NO
Models retrained after evaluation = NO
Bootstrap margin changed          = NO
R5 artificially removed           = NO
```

This is important because the baseline should remain a trustworthy reference point.

The result was accepted as observed rather than changing the evaluation rules to force a pass.

---

## 22. Model 1 Freeze

Model 1 was therefore permanently frozen as the baseline/reference model.

Freezing means:

> The learned parameters of Model 1 will no longer be changed during the later EGSF development process.

The purpose is to create a fixed reference point.

Future methods can now be compared against the same baseline.

The frozen model is:

```text
Model 1
    ↓
Corrected BaseFusion
    ↓
19,012 parameters
    ↓
20 experimental configurations
    ↓
Mean validation accuracy = 0.7789
    ↓
Gate-1 = NOT YET PASS
    ↓
FROZEN
```

The complete 20-model checkpoint was also saved and preserved.

---

## 23. Future EGSF Research Direction

Model 1 is not the final EGSF model.

It is the frozen baseline from which the remaining research will proceed.

The planned research direction is:

```text
Frozen Model 1
      ↓
Diagnose modality reliance
      ↓
Measure information justification
      ↓
Identify unjustified / excessive reliance
      ↓
Generate explanations
      ↓
Apply selective intervention
      ↓
Evaluate whether the intervention improves justified fusion
      ↓
Final EGSF model
```

The central research idea is:

> I do not want to force both modalities to contribute equally. I want the model's modality contribution to be justified by the information that the modality actually provides.

The future EGSF components will therefore be evaluated relative to the frozen Model 1 baseline rather than continuously modifying the baseline itself.

The major planned directions include:

- modality dominance diagnosis
- information-aware balancing
- explanation-guided modality weighting
- attribution-based modality importance
- counterfactual sensitivity
- quality-aware modality weighting
- selective intervention
- final explanation-guided selective fusion

The important methodological principle is:

```text
Diagnosis first
      ↓
Explanation
      ↓
Selective intervention
      ↓
Improved fusion
```

Model 1 therefore serves as the fixed experimental reference against which these future components can be evaluated.

---

# Final Model 1 Summary

| Item | Final Result |
|---|---|
| Model | Corrected BaseFusion |
| Input dimensions | 16 + 16 |
| Hidden dimension | 64 |
| Classes | 2 |
| Parameters | 19,012 |
| Correlation settings | 0.6, 0.7, 0.8, 0.9 |
| Seeds | 42, 43, 44, 45, 46 |
| Configurations | 20 |
| Mean validation accuracy | 0.7789 |
| Minimum validation accuracy | 0.6870 |
| Maximum validation accuracy | 0.8188 |
| Gate 0 | PASS |
| Gate 1 | NOT YET PASS |
| Gate-1 detailed status | POSITIVE_BUT_MARGIN_FAIL |
| ER − Q-only | +0.023456 |
| 95% CI | [0.007247, 0.043937] |
| Required margin | 0.05 |
| Final-test tuning | NO |
| Post-evaluation retraining | NO |
| Threshold changed | NO |
| Labels modified after evaluation | NO |
| Model status | FROZEN |

---

# Final Research Statement

Model 1 successfully establishes a corrected and reproducible BaseFusion baseline for the EGSF research project.

The baseline demonstrates valid multimodal fusion performance and provides a fixed reference for subsequent experiments.

The Gate-1 diagnostic criterion produced a positive effect, but the observed effect did not reach the preregistered margin. Therefore, the correct scientific conclusion is not to force a pass, but to preserve the result and continue the research from the frozen baseline.

The next stage of EGSF will focus on understanding and correcting unjustified modality reliance through explanation-guided and selective interventions.

**Model 1 is therefore complete, validated, and frozen as the baseline for the next stage of the EGSF research.**
