from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import cast

from pgdn.eval import calibration_error


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
    for key in ("n_records", "num_records", "n_eval_records", "limit"):
        value = eval_obj.get(key)
        if isinstance(value, int) and value > 0:
            return int(value)
    return None


def _require_metric(eval_obj: dict[str, object], metric: str, source: str, path: Path) -> float:
    value = eval_obj.get(metric)
    if not isinstance(value, (int, float)):
        raise SystemExit(f"missing numeric metric {metric!r} for source={source!r} in {path}")
    return float(value)


def _extract_relative_degradation(obj: dict[str, object], path: Path) -> float:
    value = obj.get("relative_degradation")
    if not isinstance(value, (int, float)):
        raise SystemExit(f"missing numeric metric 'relative_degradation' in {path}")
    return float(value)


def _load_optional_acp_relative_degradation(path: Path | None) -> tuple[float | None, str | None]:
    if path is not None:
        obj = _load_eval(path)
        return _extract_relative_degradation(obj, path), str(path)

    for candidate in (
        Path("runs/multidataset_phase2/acp_non_regression.json"),
        Path("runs/multidataset_phase1/acp_non_regression.json"),
    ):
        if not candidate.is_file():
            continue
        obj = _load_eval(candidate)
        return _extract_relative_degradation(obj, candidate), str(candidate)
    return None, None


def _build_pooled(by_source: dict[str, SourceRow]) -> dict[str, object]:
    all_have_n_records = all(row.n_records is not None for row in by_source.values())
    pooled: dict[str, object] = {}
    if all_have_n_records:
        total_weight = sum(int(cast(int, row.n_records)) for row in by_source.values())
        if total_weight > 0:
            pooled["weighting"] = "n_records"
            pooled["n_records"] = int(total_weight)
            for metric in REQUIRED_METRICS:
                numerator = sum(
                    float(by_source[source].metrics[metric]) * float(cast(int, by_source[source].n_records))
                    for source in sorted(by_source)
                )
                pooled[metric] = float(numerator / float(total_weight))
            return pooled

    pooled["weighting"] = "simple_mean_fallback"
    pooled["n_records"] = None
    for metric in REQUIRED_METRICS:
        pooled[metric] = float(
            sum(float(by_source[source].metrics[metric]) for source in sorted(by_source))
            / max(len(by_source), 1)
        )
    return pooled


def _build_ood_row(ood_eval_path: Path) -> SourceRow:
    source = "ood_holdout"
    eval_obj = _load_eval(ood_eval_path)
    return SourceRow(
        source=source,
        eval_path=str(ood_eval_path),
        n_records=_read_optional_n_records(eval_obj),
        metrics={
            metric: _require_metric(eval_obj, metric, source, ood_eval_path)
            for metric in REQUIRED_METRICS
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate phase2 multisource eval metrics with pooled, OOD, and calibration blocks",
    )
    _ = parser.add_argument(
        "--source-eval",
        action="append",
        type=_parse_source_eval,
        required=True,
        help="Repeatable mapping: source=path/to/eval.json",
    )
    _ = parser.add_argument(
        "--ood-eval",
        type=Path,
        required=True,
        help="Path to OOD holdout eval.json",
    )
    _ = parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/multidataset_phase2/eval"),
        help="Output directory for summary.json and summary.csv",
    )
    _ = parser.add_argument(
        "--acp-non-regression",
        type=Path,
        default=None,
        help=(
            "Optional path to ACP non-regression JSON containing relative_degradation; "
            "if omitted, auto-discovers runs/multidataset_phase2/acp_non_regression.json "
            "then runs/multidataset_phase1/acp_non_regression.json"
        ),
    )
    args = parser.parse_args()

    source_to_path: dict[str, Path] = {}
    source_eval_items = cast(list[tuple[str, Path]], args.source_eval)
    for source, path in source_eval_items:
        if source in source_to_path:
            raise SystemExit(f"duplicate source name in --source-eval: {source}")
        source_to_path[source] = Path(path)

    by_source_rows: dict[str, SourceRow] = {}
    for source in sorted(source_to_path):
        eval_path = source_to_path[source]
        eval_obj = _load_eval(eval_path)
        by_source_rows[source] = SourceRow(
            source=source,
            eval_path=str(eval_path),
            n_records=_read_optional_n_records(eval_obj),
            metrics={
                metric: _require_metric(eval_obj, metric, source, eval_path)
                for metric in REQUIRED_METRICS
            },
        )

    pooled = _build_pooled(by_source_rows)
    ood_row = _build_ood_row(Path(args.ood_eval))
    acp_relative_degradation, acp_non_regression_path = _load_optional_acp_relative_degradation(
        cast(Path | None, args.acp_non_regression)
    )

    by_source_json: dict[str, dict[str, object]] = {
        source: {
            "source": row.source,
            "eval_path": row.eval_path,
            "n_records": row.n_records,
            **row.metrics,
        }
        for source, row in by_source_rows.items()
    }
    if acp_relative_degradation is not None and "acp" in by_source_json:
        by_source_json["acp"]["character_consistency_mean_relative_degradation"] = acp_relative_degradation
        by_source_json["acp"]["relative_degradation"] = acp_relative_degradation

    quality_for_calibration = [
        float(by_source_rows[source].metrics["ranking_score_mean"]) for source in sorted(by_source_rows)
    ]
    uncertainty_for_calibration = [
        float(by_source_rows[source].metrics["uncertainty_mean"]) for source in sorted(by_source_rows)
    ]
    quality_for_calibration.append(float(ood_row.metrics["ranking_score_mean"]))
    uncertainty_for_calibration.append(float(ood_row.metrics["uncertainty_mean"]))

    calibration = {
        "method": "source_level_ordinal_proxy",
        "uncertainty_metric": "uncertainty_mean",
        "quality_metric": "ranking_score_mean",
        "includes_ood": True,
        "n_points": int(len(quality_for_calibration)),
        "calibration_error": float(
            calibration_error(uncertainty_for_calibration, quality_for_calibration)
        ),
        "disclaimer": "source-level proxy; confidence is ordinal reliability, not calibrated probability",
    }

    summary = {
        "metric_version": 2,
        "pooled": pooled,
        "by_source": by_source_json,
        "ood": {
            "source": ood_row.source,
            "eval_path": ood_row.eval_path,
            "n_records": ood_row.n_records,
            **ood_row.metrics,
        },
        "calibration": calibration,
    }
    if acp_relative_degradation is not None:
        summary["acp_non_regression"] = {
            "relative_degradation": acp_relative_degradation,
            "path": acp_non_regression_path,
        }

    out_dir = cast(Path, args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"

    _ = json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = ["source", "row_type", "is_ood", "n_records", *REQUIRED_METRICS]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for source in sorted(by_source_rows):
            row = by_source_rows[source]
            writer.writerow(
                {
                    "source": source,
                    "row_type": "source",
                    "is_ood": 0,
                    "n_records": row.n_records,
                    **{metric: row.metrics[metric] for metric in REQUIRED_METRICS},
                }
            )
        writer.writerow(
            {
                "source": "pooled",
                "row_type": "pooled",
                "is_ood": 0,
                "n_records": pooled.get("n_records"),
                **{metric: pooled.get(metric) for metric in REQUIRED_METRICS},
            }
        )
        writer.writerow(
            {
                "source": "__OOD_MARKER__",
                "row_type": "ood_marker",
                "is_ood": 1,
                "n_records": None,
                **{metric: None for metric in REQUIRED_METRICS},
            }
        )
        writer.writerow(
            {
                "source": ood_row.source,
                "row_type": "ood",
                "is_ood": 1,
                "n_records": ood_row.n_records,
                **{metric: ood_row.metrics[metric] for metric in REQUIRED_METRICS},
            }
        )

    print(json.dumps({"wrote": str(json_path), "wrote_csv": str(csv_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
