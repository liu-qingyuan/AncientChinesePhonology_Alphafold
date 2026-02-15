from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _as_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _dig(obj: object, keys: tuple[str, ...]) -> object:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _load_thresholds(repo_root: Path) -> dict[str, float]:
    defaults = {
        "acp_non_regression_gate": 0.05,
        "calibration_ceiling_gate": 0.25,
        "ood_floor_gate": 0.55,
    }
    registry_path = repo_root / "runs/multidataset_phase2/contracts/metric_registry.json"
    if not registry_path.exists():
        return defaults

    try:
        registry = _load_json(registry_path)
    except Exception:
        return defaults

    hierarchy = registry.get("hierarchy")
    if not isinstance(hierarchy, dict):
        return defaults
    primary = hierarchy.get("primary_ship_gates")
    if not isinstance(primary, list):
        return defaults

    thresholds = dict(defaults)
    for item in primary:
        if not isinstance(item, dict):
            continue
        gate_id = item.get("gate_id")
        threshold = _as_float(item.get("threshold"))
        if isinstance(gate_id, str) and threshold is not None and gate_id in thresholds:
            thresholds[gate_id] = threshold
    return thresholds


def _extract_acp_relative_degradation(summary: dict[str, object]) -> tuple[float | None, str | None]:
    candidates: list[tuple[tuple[str, ...], str]] = [
        (("acp_non_regression", "relative_degradation"), "acp_non_regression.relative_degradation"),
        (("acp_non_regression_gate", "relative_degradation"), "acp_non_regression_gate.relative_degradation"),
        (("gates", "acp_non_regression", "relative_degradation"), "gates.acp_non_regression.relative_degradation"),
        (("acp", "relative_degradation"), "acp.relative_degradation"),
        (
            ("by_source", "acp", "character_consistency_mean_relative_degradation"),
            "by_source.acp.character_consistency_mean_relative_degradation",
        ),
        (("by_source", "acp", "relative_degradation"), "by_source.acp.relative_degradation"),
        (
            ("by_source", "acp", "character_consistency_relative_degradation"),
            "by_source.acp.character_consistency_relative_degradation",
        ),
    ]
    for keys, label in candidates:
        value = _as_float(_dig(summary, keys))
        if value is not None:
            return value, label
    return None, None


def _check_eval_thresholds(
    repo_root: Path, thresholds: dict[str, float]
) -> tuple[dict[str, object], dict[str, object], dict[str, object], list[str]]:
    summary_path = repo_root / "runs/multidataset_phase2/eval/summary.json"
    if not summary_path.exists():
        reason = "missing runs/multidataset_phase2/eval/summary.json"
        missing: dict[str, object] = {
            "ok": False,
            "reason": reason,
            "summary_path": str(summary_path),
        }
        return missing, missing, missing, [
            f"acp_non_regression_gate:{reason}",
            f"calibration_ceiling_gate:{reason}",
            f"ood_floor_gate:{reason}",
        ]

    summary = _load_json(summary_path)
    blockers: list[str] = []

    acp_threshold = float(thresholds["acp_non_regression_gate"])
    acp_value, acp_source = _extract_acp_relative_degradation(summary)
    acp_ok = acp_value is not None and acp_value <= acp_threshold
    acp_reason = "ok"
    if acp_value is None:
        acp_reason = "missing relative_degradation metric"
    elif acp_value > acp_threshold:
        acp_reason = "threshold_exceeded"
    if not acp_ok:
        blockers.append(f"acp_non_regression_gate:{acp_reason}")
    acp_detail: dict[str, object] = {
        "ok": acp_ok,
        "metric_key": "character_consistency_mean_relative_degradation",
        "comparator": "<=",
        "threshold": acp_threshold,
        "actual": acp_value,
        "actual_source": acp_source,
        "summary_path": str(summary_path),
        "reason": acp_reason,
    }

    calibration_threshold = float(thresholds["calibration_ceiling_gate"])
    calibration_value = _as_float(_dig(summary, ("calibration", "calibration_error")))
    calibration_ok = calibration_value is not None and calibration_value <= calibration_threshold
    calibration_reason = "ok"
    if calibration_value is None:
        calibration_reason = "missing calibration.calibration_error"
    elif calibration_value > calibration_threshold:
        calibration_reason = "threshold_exceeded"
    if not calibration_ok:
        blockers.append(f"calibration_ceiling_gate:{calibration_reason}")
    calibration_detail: dict[str, object] = {
        "ok": calibration_ok,
        "metric_key": "calibration_error",
        "comparator": "<=",
        "threshold": calibration_threshold,
        "actual": calibration_value,
        "summary_path": str(summary_path),
        "reason": calibration_reason,
    }

    ood_threshold = float(thresholds["ood_floor_gate"])
    ood_value = _as_float(_dig(summary, ("ood", "ranking_score_mean")))
    ood_ok = ood_value is not None and ood_value >= ood_threshold
    ood_reason = "ok"
    if ood_value is None:
        ood_reason = "missing ood.ranking_score_mean"
    elif ood_value < ood_threshold:
        ood_reason = "below_threshold"
    if not ood_ok:
        blockers.append(f"ood_floor_gate:{ood_reason}")
    ood_detail: dict[str, object] = {
        "ok": ood_ok,
        "metric_key": "ranking_score_mean",
        "comparator": ">=",
        "threshold": ood_threshold,
        "actual": ood_value,
        "summary_path": str(summary_path),
        "reason": ood_reason,
    }

    return acp_detail, calibration_detail, ood_detail, blockers


def _check_repro(repo_root: Path) -> tuple[dict[str, object], list[str]]:
    repro_path = repo_root / "runs/multidataset_phase2/repro/repro_check.json"
    if not repro_path.exists():
        reason = "missing runs/multidataset_phase2/repro/repro_check.json"
        return {
            "ok": False,
            "repro_path": str(repro_path),
            "reason": reason,
        }, [f"reproducibility_gate:{reason}"]

    repro = _load_json(repro_path)
    status = repro.get("status")
    core_hashes_ok = repro.get("core_hashes_ok")
    artifact_manifest = repro.get("artifact_manifest")

    manifest_exists = False
    manifest_path = None
    if isinstance(artifact_manifest, str):
        manifest_path = repo_root / artifact_manifest
        manifest_exists = manifest_path.exists()

    ok = status == "PASS" and core_hashes_ok is True and manifest_exists
    reason = "ok"
    if status != "PASS":
        reason = "status_not_pass"
    elif core_hashes_ok is not True:
        reason = "core_hashes_not_ok"
    elif not manifest_exists:
        reason = "missing_artifact_manifest"

    blockers = [] if ok else [f"reproducibility_gate:{reason}"]
    detail = {
        "ok": ok,
        "repro_path": str(repro_path),
        "status": status,
        "core_hashes_ok": core_hashes_ok,
        "artifact_manifest": artifact_manifest,
        "artifact_manifest_exists": manifest_exists,
        "reason": reason,
    }
    return detail, blockers


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble multidataset phase2 final gate summary")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/multidataset_phase2/gate_summary.json"),
        help="output gate summary path",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="exit non-zero when any blocking check fails",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    thresholds = _load_thresholds(repo_root)

    checks: dict[str, dict[str, object]] = {}
    blockers: list[str] = []

    repro_detail, repro_blockers = _check_repro(repo_root)
    checks["reproducibility_gate"] = repro_detail
    blockers.extend(repro_blockers)

    acp_detail, calibration_detail, ood_detail, eval_blockers = _check_eval_thresholds(repo_root, thresholds)
    checks["acp_non_regression_gate"] = acp_detail
    checks["calibration_ceiling_gate"] = calibration_detail
    checks["ood_floor_gate"] = ood_detail
    blockers.extend(eval_blockers)

    decision = "GO" if not blockers else "NO_GO"
    gate_summary = {
        "decision": decision,
        "blockers": blockers,
        "checks": checks,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gate_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"gate_decision={decision}")
    print(f"blocker_count={len(blockers)}")
    print(f"gate_summary={args.out}")

    if args.require_all and blockers:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
