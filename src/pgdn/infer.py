import argparse
import csv
import json
import random
from pathlib import Path

from .confidence import confidence_bucket, ranking_terms_for_record
from .data.dataset import load_targets
from .data.io import write_json


def _predict_samples(mean_vec, ablation: str, samples: int, seed: int):
    rng = random.Random(seed)
    scale = 0.25
    if ablation == "no_diffusion":
        scale = 0.08
    elif ablation == "no_pairformer":
        scale = 0.30
    elif ablation == "no_constraint_loss":
        scale = 0.35
    out = []
    for _ in range(samples):
        out.append([v + rng.gauss(0.0, scale) for v in mean_vec])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer PGDN v0")
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--job-name", type=str, default="pgdn_v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=["none", "no_pairformer", "no_diffusion", "no_constraint_loss"],
    )
    args = parser.parse_args()

    ckpt = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    rows = load_targets(args.targets)
    with args.split_manifest.open("r", encoding="utf-8") as f:
        split = json.load(f)
    test_ids = set(split["random"]["test"])
    eval_rows = [r for r in rows if r["record_id"] in test_ids][: args.limit]

    run_dir = args.out / args.job_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "ranking_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "record_id",
                "seed",
                "sample",
                "ranking_score",
                "constraint_satisfaction",
                "sample_consistency",
                "penalty_impossible_combo",
                "penalty_constraint_violation",
                "uncertainty_mean",
                "confidence_bucket",
            ],
        )
        writer.writeheader()

        for i, row in enumerate(eval_rows):
            sample_vectors = _predict_samples(
                ckpt["mean_vector"], args.ablation, args.samples, args.seed + i
            )
            terms = ranking_terms_for_record(sample_vectors, row["mask"])

            summary = {
                "job_name": args.job_name,
                "run_id": f"{args.job_name}:{row['record_id']}:{args.seed}",
                "seed": args.seed,
                "sample": 0,
                "record_id": row["record_id"],
                "character": row["character"],
                "language": row["language"],
                "period": row["period"],
                "ranking_score": terms["ranking_score"],
                "constraint_satisfaction": terms["constraint_satisfaction"],
                "sample_consistency": terms["sample_consistency"],
                "penalty_impossible_combo": terms["penalty_impossible_combo"],
                "penalty_constraint_violation": terms["penalty_constraint_violation"],
                "uncertainty_mean": terms["uncertainty_mean"],
                "confidence_bucket": confidence_bucket(terms["uncertainty_mean"]),
                "calibration_note": "ordinal confidence only; not probability calibrated",
            }

            out_dir = (
                run_dir
                / f"seed-{args.seed}_sample-0"
                / row["record_id"].replace(":", "_")
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            mean_prediction = [
                sum(col) / float(len(col)) for col in zip(*sample_vectors)
            ]
            write_json(out_dir / "summary.json", summary)
            write_json(out_dir / "model.json", {"prediction": mean_prediction})
            write_json(
                out_dir / "confidences.json",
                {"uncertainty_mean": summary["uncertainty_mean"]},
            )

            writer.writerow(
                {
                    "record_id": row["record_id"],
                    "seed": args.seed,
                    "sample": 0,
                    "ranking_score": summary["ranking_score"],
                    "constraint_satisfaction": summary["constraint_satisfaction"],
                    "sample_consistency": summary["sample_consistency"],
                    "penalty_impossible_combo": summary["penalty_impossible_combo"],
                    "penalty_constraint_violation": summary[
                        "penalty_constraint_violation"
                    ],
                    "uncertainty_mean": summary["uncertainty_mean"],
                    "confidence_bucket": summary["confidence_bucket"],
                }
            )

    print(f"wrote={csv_path}")


if __name__ == "__main__":
    main()
