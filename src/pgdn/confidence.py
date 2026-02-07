from statistics import mean
from typing import Dict, List


def confidence_bucket(uncertainty: float) -> str:
    if uncertainty < 0.08:
        return "high"
    if uncertainty < 0.18:
        return "medium"
    return "low"


def _variance(values: List[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    return sum((v - m) * (v - m) for v in values) / float(len(values))


def ranking_terms_for_record(
    sample_vectors: List[List[float]], slot_mask: Dict[str, int]
) -> Dict[str, float]:
    if not sample_vectors:
        return {
            "constraint_satisfaction": 0.0,
            "sample_consistency": 0.0,
            "penalty_impossible_combo": 0.0,
            "penalty_constraint_violation": 1.0,
            "uncertainty_mean": 1.0,
            "ranking_score": -1.0,
        }

    dim = len(sample_vectors[0])
    per_dim_var = []
    for i in range(dim):
        per_dim_var.append(_variance([s[i] for s in sample_vectors]))
    uncertainty = mean(per_dim_var)

    sample_consistency = max(0.0, 1.0 - uncertainty)

    avg_vec = [mean([s[i] for s in sample_vectors]) for i in range(dim)]
    impossible = mean(max(0.0, abs(v) - 1.0) for v in avg_vec)

    expanded_mask = []
    for slot in ("I", "M", "N", "C"):
        expanded_mask.extend([float(slot_mask.get(slot, 0))] * 8)
    violations = [abs(v) * (1.0 - expanded_mask[i]) for i, v in enumerate(avg_vec)]
    penalty_violation = mean(violations)
    constraint = max(0.0, 1.0 - penalty_violation)

    ranking_score = (
        0.45 * constraint
        + 0.30 * sample_consistency
        - 0.15 * impossible
        - 0.10 * penalty_violation
        - 0.10 * uncertainty
    )

    return {
        "constraint_satisfaction": constraint,
        "sample_consistency": sample_consistency,
        "penalty_impossible_combo": impossible,
        "penalty_constraint_violation": penalty_violation,
        "uncertainty_mean": uncertainty,
        "ranking_score": ranking_score,
    }
