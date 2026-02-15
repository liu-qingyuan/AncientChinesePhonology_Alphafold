# Experiments Notes Index

This directory stores concise, tracked experiment notes.

Policy:
- Keep raw artifacts in `runs/` (untracked).
- Keep auto-generated local reports in `reports/` (untracked).
- Keep only short, human-readable summaries here.

Current notes:
- `docs/experiments/coupling.md`: group-coupling benchmark summary and pointers to run outputs.
- `docs/experiments/paper_grade_validation.md`: five-seed paper-grade validation summary with aggregated statistics.
- `docs/experiments/multidataset_integration_phase1.md`: Phase1 multidataset integration gate summary with per-source and pooled results.
- `docs/experiments/multidataset_integration_phase2.md`: Phase2 multidataset integration gate summary (current decision: GO from `runs/multidataset_phase2/gate_summary.json`; includes dated historical no-go note for provenance).

Suggested format for future notes:
- Goal (what question this experiment answers)
- Setup (key flags, checkpoint, split)
- Result (small table + one-paragraph conclusion)
- Evidence paths (exact `runs/...` locations)
