import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

from .data.io import write_json


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def top_k_coverage(scores: List[float], k: int = 5) -> float:
    if not scores:
        return 0.0
    ranked = sorted(scores, reverse=True)
    threshold = ranked[min(k - 1, len(ranked) - 1)]
    return sum(1 for s in scores if s >= threshold) / float(len(scores))


def pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0.0 or deny == 0.0:
        return 0.0
    return num / (denx * deny)


def calibration_error(uncertainty: List[float], quality: List[float]) -> float:
    if not uncertainty:
        return 0.0
    return mean(abs((1.0 - u) - q) for u, q in zip(uncertainty, quality))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PGDN ranking outputs")
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/pgdn_v0/eval.json"))
    args = parser.parse_args()

    rows = load_rows(args.ranking)
    scores = [float(r["ranking_score"]) for r in rows]
    constraints = [float(r["constraint_satisfaction"]) for r in rows]
    uncertainties = [float(r["uncertainty_mean"]) for r in rows]
    penalties = [float(r["penalty_constraint_violation"]) for r in rows]

    metrics = {
        "top_k_coverage_at_5": top_k_coverage(scores, k=5),
        "ranking_constraint_correlation": pearson(scores, constraints),
        "constraint_violation_breakdown": {
            "mean_penalty_constraint_violation": mean(penalties) if penalties else 0.0,
            "std_penalty_constraint_violation": pstdev(penalties)
            if len(penalties) > 1
            else 0.0,
        },
        "calibration": {
            "ordinal_ece_proxy": calibration_error(uncertainties, constraints),
            "disclaimer": "confidence is ordinal reliability, not calibrated probability",
        },
    }
    write_json(args.out, metrics)
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
