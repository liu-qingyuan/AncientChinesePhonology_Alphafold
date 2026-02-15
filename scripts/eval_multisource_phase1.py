import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import cast


REQUIRED_METRICS = [
    "ranking_score_mean",
    "uncertainty_mean",
    "constraint_penalty_mean",
    "max_abs",
]


@dataclass(frozen=True)
class SourceRow:
    source: str
    eval_path: str
    n_records: int | None
    metrics: dict[str, float]


def _parse_source_eval(text: str) -> tuple[str, Path]:
    source, sep, raw_path = text.partition("=")
    source_name = source.strip()
    path_text = raw_path.strip()
    if sep != "=" or not source_name or not path_text:
        raise argparse.ArgumentTypeError(
            f"invalid --source-eval {text!r}; expected format 'source=path/to/eval.json'"
        )
    return source_name, Path(path_text)


def _load_eval(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"missing eval file: {path}")
    obj: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"expected JSON object in {path}")
    return cast(dict[str, object], obj)


def _read_optional_n_records(eval_obj: dict[str, object]) -> int | None:
    for key in ("n_records", "num_records", "n_eval_records"):
        value = eval_obj.get(key)
        if isinstance(value, int) and value > 0:
            return int(value)
    return None


def _require_metric(eval_obj: dict[str, object], metric: str, source: str, path: Path) -> float:
    value = eval_obj.get(metric)
    if not isinstance(value, (int, float)):
        raise SystemExit(f"missing numeric metric {metric!r} for source={source!r} in {path}")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-source eval metrics into pooled multisource summary",
    )
    _ = parser.add_argument(
        "--source-eval",
        action="append",
        type=_parse_source_eval,
        required=True,
        help="Repeatable mapping: source=path/to/eval.json",
    )
    _ = parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/multidataset_phase1/eval"),
        help="Output directory for summary.json and summary.csv",
    )
    args = parser.parse_args()

    source_to_path: dict[str, Path] = {}
    source_eval_items = cast(list[tuple[str, Path]], args.source_eval)
    for source, path in source_eval_items:
        if source in source_to_path:
            raise SystemExit(f"duplicate source name in --source-eval: {source}")
        source_to_path[source] = Path(path)

    by_source: dict[str, SourceRow] = {}
    all_have_n_records = True
    for source in sorted(source_to_path):
        eval_path = source_to_path[source]
        eval_obj = _load_eval(eval_path)
        n_records = _read_optional_n_records(eval_obj)
        metrics = {
            metric: _require_metric(eval_obj, metric, source, eval_path)
            for metric in REQUIRED_METRICS
        }
        by_source[source] = SourceRow(
            source=source,
            eval_path=str(eval_path),
            n_records=n_records,
            metrics=metrics,
        )
        if n_records is None:
            all_have_n_records = False

    pooled: dict[str, object] = {}
    if all_have_n_records:
        weights: list[int] = []
        for source in sorted(by_source):
            n_records = by_source[source].n_records
            if n_records is None:
                raise RuntimeError("internal error: n_records missing in weighted branch")
            weights.append(int(n_records))
        total_weight = sum(weights)
        if total_weight <= 0:
            all_have_n_records = False
        else:
            pooled["weighting"] = "n_records"
            pooled["n_records"] = int(total_weight)
            for metric in REQUIRED_METRICS:
                numerator = 0.0
                for source in sorted(by_source):
                    row = by_source[source]
                    n_records = row.n_records
                    if n_records is None:
                        raise RuntimeError("internal error: n_records missing in weighted branch")
                    numerator += float(row.metrics[metric]) * float(n_records)
                pooled[metric] = float(numerator / float(total_weight))

    if not all_have_n_records:
        pooled["weighting"] = "simple_mean_fallback"
        pooled["n_records"] = None
        for metric in REQUIRED_METRICS:
            pooled[metric] = float(
                sum(float(by_source[source].metrics[metric]) for source in sorted(by_source)) / max(len(by_source), 1)
            )

    by_source_json = {
        source: {
            "source": row.source,
            "eval_path": row.eval_path,
            "n_records": row.n_records,
            **row.metrics,
        }
        for source, row in by_source.items()
    }

    summary = {
        "metric_version": 1,
        "pooled": pooled,
        "by_source": by_source_json,
    }

    out_dir = cast(Path, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"

    _ = json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = ["source", "n_records", *REQUIRED_METRICS]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for source in sorted(by_source):
            row = by_source[source]
            writer.writerow(
                {
                    "source": source,
                    "n_records": row.n_records,
                    **{metric: row.metrics[metric] for metric in REQUIRED_METRICS},
                }
            )
        writer.writerow(
            {
                "source": "pooled",
                "n_records": pooled.get("n_records"),
                **{metric: pooled.get(metric) for metric in REQUIRED_METRICS},
            }
        )

    print(json.dumps({"wrote": str(json_path), "wrote_csv": str(csv_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
