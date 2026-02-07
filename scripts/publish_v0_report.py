import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def load_results(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish PGDN v0 report")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/v0"))
    args = parser.parse_args()

    results = load_results(args.results)
    with args.eval.open("r", encoding="utf-8") as f:
        eval_metrics = json.load(f)
    with args.split_manifest.open("r", encoding="utf-8") as f:
        split_manifest = json.load(f)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "PGDN_v0_report.md"
    repro_path = args.out_dir / "reproducibility_manifest.json"

    lines = []
    lines.append("# PGDN v0 Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("- ACP-first pipeline")
    lines.append("- Baseline + capped ablations (<=3)")
    lines.append("- Ranking and confidence with calibration disclaimer")
    lines.append("")
    lines.append("## Results Table")
    if results:
        lines.append("| experiment | mean_ranking_score | std_ranking_score |")
        lines.append("|---|---:|---:|")
        for row in results:
            lines.append(
                f"| {row['experiment']} | {float(row['mean_ranking_score']):.4f} | {float(row['std_ranking_score']):.4f} |"
            )
    else:
        lines.append("No experiment results found.")
    lines.append("")
    lines.append("## Evaluation")
    lines.append(f"- top_k_coverage_at_5: {eval_metrics['top_k_coverage_at_5']:.4f}")
    lines.append(
        f"- ranking_constraint_correlation: {eval_metrics['ranking_constraint_correlation']:.4f}"
    )
    lines.append(
        "- calibration: "
        f"{eval_metrics['calibration']['ordinal_ece_proxy']:.4f} "
        f"({eval_metrics['calibration']['disclaimer']})"
    )
    lines.append("")
    lines.append("## Risk Register")
    lines.append("1. Diffusion backbone adaptation mismatch (image-first assumptions)")
    lines.append("2. Split leakage in historical data organization")
    lines.append("3. Uncalibrated confidence interpreted as probability")
    lines.append("4. Scope creep into Tangut-first pipeline")
    lines.append("")
    lines.append("## Deferred to v0.2+")
    lines.append("1. Tangut-first production data pipeline")
    lines.append("2. Sim-to-Real large synthetic pretraining")
    lines.append("3. Recycling/self-distillation loops")
    lines.append("4. Advanced controversy hypothesis engine")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reproducibility = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": [7, 13, 42],
        "split_manifest": split_manifest,
        "config": {
            "target_dim": 32,
            "single_dim": 64,
            "pair_dim": 32,
            "timesteps": 100,
            "samples": 8,
        },
        "dependencies": {
            "python": "3.10+",
            "torch": "required",
            "lucidrains_denoising_diffusion_pytorch": "reference backbone",
        },
        "known_risks": [
            "confidence is ordinal and not probability-calibrated",
            "feature encoder is hashed proxy for v0 and not phonetic expert feature mapping",
        ],
    }
    repro_path.write_text(
        json.dumps(reproducibility, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote={report_path}")
    print(f"wrote={repro_path}")


if __name__ == "__main__":
    main()
