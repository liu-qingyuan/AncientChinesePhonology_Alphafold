# Multidataset Integration Phase 1

## Goal

Enable first production-like multi-source run (ACP + new sidecars) with source-aware evaluation and strict non-regression safety gates.

## Setup

- **Sidecars integrated**: tshet-uinh (韻書/廣韻), Unihan
- **Data contract**: Phase1 fields (`record_id`, `graph_id`, `target_vector`, `mask`, `source`) with namespaced IDs (`acp:`, `mocha:`, `tshet:`, `unihan:`)
- **Evaluation**: Multi-source eval with per-source and pooled metrics
- **Gates enforced**:
  - Contract validation
  - Sidecar smoke (tshet + unihan)
  - Source-aware leakage checker
  - ACP non-regression (threshold 5%)

## Results

### Per-Source and Pooled Metrics

| Source | ranking_score_mean | constraint_penalty_mean | max_abs |
|--------|-------------------|------------------------|---------|
| acp    | 0.6647 | 0.1551 | 0.7862 |
| tshet  | 0.7500 | 0.0000 | 0.7848 |
| unihan | 0.7154 | 0.0630 | 0.7800 |
| **pooled** | **0.7100** | **0.0727** | **0.7836** |

### Gate Summary

| Gate | Status |
|------|--------|
| Contract validation | PASS |
| Sidecar artifacts | PASS |
| Source-eval (3 sources) | PASS |
| Leakage (clean) | PASS |
| Leakage (injected) | FAIL (as expected) |
| ACP non-regression | PASS (GO) |

**Final decision**: `GO` (no blockers)

## Conclusion

Phase1 successfully integrated tshet-uinh and Unihan sidecars alongside existing ACP, with source-aware evaluation producing per-source and pooled metrics. All gates passed: contract validation, sidecar smoke, leakage detection (both clean and injected fail cases), and ACP non-regression. The pooled `ranking_score_mean` of 0.71 is within expected range.

## Deferred to Phase 2

- Additional sidecar sources beyond tshet + Unihan
- Architecture-wide redesign (Pairformer/diffusion redesign)
- Full multi-source training at scale
- Enhanced source-aware weighting policies

## Evidence Paths

- Gate summary: `runs/multidataset_phase1/gate_summary.json`
- Eval summary JSON: `runs/multidataset_phase1/eval/summary.json`
- Eval summary CSV: `runs/multidataset_phase1/eval/summary.csv`
- ACP non-regression: `runs/multidataset_phase1/acp_non_regression.json`
- Phase1 contract validator: `scripts/validate_phase1_contract.py`
