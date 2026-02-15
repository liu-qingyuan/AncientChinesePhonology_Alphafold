from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CORE_ARTIFACTS = [
    "eval/summary.json",
    "eval/summary.csv",
    "pretrain/checkpoint_none.pt",
    "pretrain/train_metrics.jsonl",
    "pretrain/pretrain_run_meta.json",
    "self_distill/comparison.json",
    "self_distill/pseudo_label_manifest.json",
    "self_distill/v2/checkpoint_none.pt",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"expected JSON object in {path}")
    return obj


def _flatten_numbers(value: Any, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = value[key]
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_numbers(child, child_prefix))
        return out
    if isinstance(value, list):
        for i, child in enumerate(value):
            child_prefix = f"{prefix}[{i}]"
            out.update(_flatten_numbers(child, child_prefix))
        return out
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)
    return out


def _artifact_manifest(run_dir: Path) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    missing: dict[str, str] = {}
    for rel in CORE_ARTIFACTS:
        path = run_dir / rel
        if not path.is_file():
            missing[rel] = str(path)
            continue
        artifacts[rel] = {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
    return {
        "run_dir": str(run_dir),
        "core_artifact_count": len(CORE_ARTIFACTS),
        "hashed_artifact_count": len(artifacts),
        "missing_artifacts": missing,
        "artifacts": artifacts,
    }


def _numeric_payload(run_dir: Path) -> dict[str, float]:
    eval_summary = _load_json_object(run_dir / "eval" / "summary.json")
    distill_comparison = _load_json_object(run_dir / "self_distill" / "comparison.json")

    selected = {
        "eval": {
            "pooled": eval_summary.get("pooled"),
            "by_source": eval_summary.get("by_source"),
            "ood": eval_summary.get("ood"),
            "calibration": eval_summary.get("calibration"),
        },
        "self_distill": {
            "pseudo_selection": distill_comparison.get("pseudo_selection"),
            "v1": distill_comparison.get("v1"),
            "v2": distill_comparison.get("v2"),
            "delta_v2_minus_v1": distill_comparison.get("delta_v2_minus_v1"),
        },
    }
    return _flatten_numbers(selected, "")


def _compare_numbers(
    a_values: dict[str, float],
    b_values: dict[str, float],
    tolerance: float,
) -> tuple[bool, dict[str, Any]]:
    keys = sorted(set(a_values) | set(b_values))
    missing_in_a = [k for k in keys if k not in a_values]
    missing_in_b = [k for k in keys if k not in b_values]

    mismatches: list[dict[str, Any]] = []
    compared_count = 0
    max_abs_diff = 0.0
    for key in keys:
        if key not in a_values or key not in b_values:
            continue
        a = a_values[key]
        b = b_values[key]
        abs_diff = abs(a - b)
        compared_count += 1
        max_abs_diff = max(max_abs_diff, abs_diff)
        if math.isnan(abs_diff) or abs_diff > tolerance:
            mismatches.append(
                {
                    "metric": key,
                    "run_a": a,
                    "run_b": b,
                    "abs_diff": abs_diff,
                    "tolerance": tolerance,
                }
            )

    ok = not missing_in_a and not missing_in_b and not mismatches
    return ok, {
        "ok": ok,
        "tolerance": float(tolerance),
        "compared_metric_count": int(compared_count),
        "max_abs_diff": float(max_abs_diff),
        "missing_metric_in_run_a": missing_in_a,
        "missing_metric_in_run_b": missing_in_b,
        "mismatches": mismatches,
    }


def _check_run_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"missing {label} path: {path}")
    if not path.is_dir():
        raise SystemExit(f"{label} must be a directory: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase2 reproducibility and deterministic rerun gate")
    _ = parser.add_argument("--run-a", type=Path, required=True, help="First run directory")
    _ = parser.add_argument("--run-b", type=Path, required=True, help="Second run directory")
    _ = parser.add_argument("--tolerance", type=float, default=1e-6, help="Absolute tolerance for numeric comparison")
    _ = parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("runs/multidataset_phase2/repro/artifact_manifest.json"),
        help="Output JSON path for hash manifest",
    )
    _ = parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/multidataset_phase2/repro/repro_check.json"),
        help="Output JSON path for repro check",
    )
    args = parser.parse_args()

    if float(args.tolerance) < 0:
        raise SystemExit(f"tolerance must be >= 0, got: {args.tolerance}")

    _check_run_dir(args.run_a, "run-a")
    _check_run_dir(args.run_b, "run-b")

    run_a_manifest = _artifact_manifest(args.run_a)
    run_b_manifest = _artifact_manifest(args.run_b)
    manifest = {
        "manifest_version": 1,
        "run_a": run_a_manifest,
        "run_b": run_b_manifest,
    }

    core_hashes_ok = (
        run_a_manifest["hashed_artifact_count"] == len(CORE_ARTIFACTS)
        and run_b_manifest["hashed_artifact_count"] == len(CORE_ARTIFACTS)
    )

    run_a_numbers = _numeric_payload(args.run_a)
    run_b_numbers = _numeric_payload(args.run_b)
    metrics_ok, metrics_detail = _compare_numbers(run_a_numbers, run_b_numbers, float(args.tolerance))

    status = "PASS" if core_hashes_ok and metrics_ok else "FAIL"
    result = {
        "gate": "phase2_reproducibility",
        "status": status,
        "run_a": str(args.run_a),
        "run_b": str(args.run_b),
        "core_hashes_ok": bool(core_hashes_ok),
        "core_artifacts": list(CORE_ARTIFACTS),
        "compared_artifacts": [
            "eval/summary.json",
            "self_distill/comparison.json",
        ],
        "numeric_comparison": metrics_detail,
        "artifact_manifest": str(args.manifest_out),
    }

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"repro_status={status}")
    print(f"core_hashes_ok={core_hashes_ok}")
    print(f"compared_metric_count={metrics_detail['compared_metric_count']}")
    print(f"artifact_manifest={args.manifest_out}")
    print(f"repro_check={args.out}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
