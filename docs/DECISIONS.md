# DECISIONS

Records important design and research decisions with rationale.

---

## D-1: CPU-only PyTorch
**Date:** 2026-10-04
**Decision:** Proceed with torch 2.14.0+cpu (no CUDA detected).
**Rationale:** JDB-S experiments use small MLPs; CPU is sufficient for Steps 1–11.
**Risk:** C3/real-data experiments may be slow if using larger encoders.
**Action if risk materializes:** Evaluate cloud GPU or reduce model size.

## D-2: Fixed random seeds
**Decision:** Use seeds {0, 1, 2, 3, 4} for all 5-seed runs; seeds {0, 1, 2} for 3-seed dev runs.
**Rationale:** Reproducibility requirement from project plan.

---
<!-- Add new entries as decisions are made -->
