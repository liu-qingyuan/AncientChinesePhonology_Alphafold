# Multidataset Integration Phase 2

## Goal

Extend multi-source integration with self-distillation (pseudo-labels from v1 -> v2), AF3 alignment evaluation, and stricter OOD/calibration gates.

## Setup

- **Sidecars integrated**: ACP (韻書/廣韻), WikiHan (wikihan)
- **Data contract**: Phase2 synthetic namespace (`synth_phase2:`), ACP + wikihan records with full provenance
- **Evaluation**: Pooled + per-source + OOD holdout + calibration metrics
- **Gates enforced**:
  - Reproducibility check
  - ACP non-regression (relative degradation <= 0.05)
  - Calibration ceiling (calibration error <= 0.25)
  - OOD floor (ranking_score_mean >= 0.55)

## Results

### Pooled and Per-Source Metrics

| Source | n_records | ranking_score_mean | uncertainty_mean | constraint_penalty_mean | max_abs |
|--------|-----------|-------------------|------------------|------------------------|---------|
| acp    | 64        | 0.6718            | 0.0868           | 0.0790                 | 0.8231  |
| wikihan| 64        | 0.6864            | 0.0657           | 0.0679                 | 0.7895  |
| **pooled** | **128** | **0.6791**       | **0.0763**       | **0.0734**             | **0.8063** |

### OOD and Calibration

| Check | Metric | Threshold | Actual | Status |
|-------|--------|-----------|--------|--------|
| OOD floor | ranking_score_mean | >= 0.55 | 0.6864 | PASS |
| Calibration ceiling | calibration_error | <= 0.25 | 0.2457 | PASS |
| Reproducibility | core hashes + numeric | identical | 0.0 diff | PASS |

### Gate Summary

| Gate | Status |
|------|--------|
| Reproducibility | PASS |
| ACP non-regression | PASS |
| Calibration ceiling | PASS |
| OOD floor | PASS |

**Final decision**: `GO` (source: `runs/multidataset_phase2/gate_summary.json`)

**Historical note (2026-02-16)**: an earlier note recorded a no-go decision when ACP relative degradation was missing from the eval summary. Current gate truth is `GO` per `runs/multidataset_phase2/gate_summary.json`.

## Conclusion

Phase2 self-distillation pipeline (v1 -> pseudo-labels -> v2) completed successfully with pooled ranking_score_mean of 0.679. Calibration (0.2457), OOD floor (0.6864), reproducibility, and ACP non-regression checks passed. Current gate decision is `GO` from `runs/multidataset_phase2/gate_summary.json`.

## Deferred to Phase 3

- Additional sidecar sources beyond ACP + WikiHan
- Architecture-wide redesign (Pairformer/diffusion redesign)
- Full multi-source training at scale
- Enhanced source-aware weighting policies

## Evidence Paths

- Gate summary: `runs/multidataset_phase2/gate_summary.json`
- Eval summary JSON: `runs/multidataset_phase2/eval/summary.json`
- Eval summary CSV: `runs/multidataset_phase2/eval/summary.csv`
- Reproducibility check: `runs/multidataset_phase2/repro/repro_check.json`
- Artifact manifest: `runs/multidataset_phase2/repro/artifact_manifest.json`
- Phase2 contract validator: `scripts/validate_phase2_contract.py`
