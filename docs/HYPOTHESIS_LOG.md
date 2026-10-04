# HYPOTHESIS LOG

## Central Hypothesis
Explanation-Guided Selective Fusion (EGSF) detects when a multimodal model
relies excessively on a modality beyond what its invariant information
justifies, then selectively corrects that reliance with a risk-controlled gate.

Core chain:
  Reliance → invariant information budget → excess reliance → trust →
  spuriousness → uncertainty → calibrated diagnosis → certified selective
  correction

---

## Hypothesis Entries

### H-0: Environment Setup
**Hypothesis:** Standard PyTorch ML stack is sufficient for this project.
**Experiment:** Package import check.
**Expected:** torch, numpy, sklearn, scipy all importable.
**Observed:** All present (torch 2.14.0+cpu, numpy 2.2.6, sklearn 1.7.1, scipy 1.17.1).
**Conclusion:** CONFIRMED. Environment is adequate.

---
<!-- Add new entries below as the project progresses -->
