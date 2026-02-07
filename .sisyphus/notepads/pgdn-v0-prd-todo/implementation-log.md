# PGDN v0 Implementation Learnings

## 2026-02-06

- Locked ACP-first contract in `PGDN_IMPLEMENTATION_PLAN.md` with mandatory `language`, `period`, `syllable_slots`, `mask` fields.
- Froze split protocol (random, temporal, low-resource 10/5/1/0%) and leakage prevention rules to reduce silent evaluation drift.
- Implemented minimal ACP data builder that generates graph/targets/splits JSON artifacts with deterministic behavior.
- Implemented PGDN v0 module chain (embedder -> pairformer-lite -> conditioned diffusion) plus ranking/confidence outputs.
- Added baseline and capped ablation automation (`no_pairformer`, `no_diffusion`, `no_constraint_loss`) with mean/std aggregation.
- Added evaluation and report publishing scripts including calibration disclaimer and reproducibility manifest.
- Environment blocker discovered: `torch` was not installed in runtime, so v0 execution path was implemented as a deterministic Python baseline that preserves module boundaries and output contracts.
- Added `scripts/cleanup_runs.py` with academic-style retention policy (keep aggregate metrics, keep top-K samples, keep latest checkpoints) and manifest-based audit trail.
