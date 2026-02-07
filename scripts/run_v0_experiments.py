import argparse
import csv
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_metric(path: Path) -> float:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    scores = [float(r["ranking_score"]) for r in rows]
    return mean(scores) if scores else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline + capped ablations")
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/v0_experiments"))
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()

    modes = ["no_diffusion", "no_pairformer", "none", "no_constraint_loss"]
    seeds = [7, 13]
    rows = []

    for mode in modes:
        scores = []
        for seed in seeds:
            ckpt_dir = args.out / "checkpoints"
            run(
                [
                    args.python,
                    "-m",
                    "pgdn.train",
                    "--targets",
                    str(args.targets),
                    "--split-manifest",
                    str(args.split_manifest),
                    "--epochs",
                    "1",
                    "--seed",
                    str(seed),
                    "--ablation",
                    mode,
                    "--out",
                    str(ckpt_dir),
                ]
            )
            ckpt = ckpt_dir / f"pgdn_v0_{mode}.json"
            run(
                [
                    args.python,
                    "-m",
                    "pgdn.infer",
                    "--targets",
                    str(args.targets),
                    "--split-manifest",
                    str(args.split_manifest),
                    "--checkpoint",
                    str(ckpt),
                    "--job-name",
                    f"{mode}_seed{seed}",
                    "--seed",
                    str(seed),
                    "--ablation",
                    mode,
                    "--out",
                    str(args.out),
                    "--limit",
                    "32",
                ]
            )
            ranking_csv = args.out / f"{mode}_seed{seed}" / "ranking_scores.csv"
            scores.append(read_metric(ranking_csv))
        rows.append(
            {
                "experiment": "baseline_transformer"
                if mode == "no_diffusion"
                else mode,
                "mean_ranking_score": mean(scores),
                "std_ranking_score": pstdev(scores) if len(scores) > 1 else 0.0,
            }
        )

    table_path = args.out / "results_table.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["experiment", "mean_ranking_score", "std_ranking_score"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote={table_path}")


if __name__ == "__main__":
    main()
