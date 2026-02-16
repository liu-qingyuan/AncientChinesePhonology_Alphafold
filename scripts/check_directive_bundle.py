#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

FREEZE_POLICY_PATH = ROOT / "runs" / "policy" / "dataset_scope_freeze_policy.json"
FREEZE_CHECK_PATH = ROOT / "runs" / "policy" / "dataset_scope_freeze_check.json"
MAPPING_PATH = ROOT / "runs" / "directive_bundle" / "directive_mapping.json"
REPRO_GATE_PATH = ROOT / "runs" / "directive_bundle" / "repro_gate_check.json"
BUNDLE_PATH = ROOT / "runs" / "directive_bundle" / "directive_execution_bundle.json"
TABLE_PATH = ROOT / "runs" / "directive_bundle" / "directive_execution_table.csv"
LOCK_VERIFY_PATH = ROOT / "runs" / "next_phase_lock" / "lock_verify.json"

EXPECTED_BLOCKED_SOURCES = {"STEDT", "CogNet"}
EXPECTED_TABLE_COLUMNS = {
    "task_id",
    "artifact_path",
    "status",
    "verification_status",
    "blocked_sources",
    "referenced_paths_count",
    "paths_exist",
}


def _load_json_object(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"expected JSON object in {path}")
    return obj


def _must(condition: bool, checks: list[dict[str, Any]], violations: list[str], name: str, detail: str) -> None:
    status = "PASS" if condition else "FAIL"
    checks.append({"name": name, "status": status, "detail": detail})
    if not condition:
        violations.append(f"{name}: {detail}")


def _parse_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"missing CSV header in {path}")
        rows = [row for row in reader]
        return list(reader.fieldnames), rows


def _check_freeze_policy(
    freeze_policy: dict[str, Any],
    freeze_check: dict[str, Any],
    mapping: dict[str, Any],
    bundle: dict[str, Any],
    checks: list[dict[str, Any]],
    violations: list[str],
) -> None:
    blocked_sources = freeze_policy.get("blocked_sources")
    blocked_set = set(blocked_sources) if isinstance(blocked_sources, list) else set()
    _must(
        blocked_set == EXPECTED_BLOCKED_SOURCES,
        checks,
        violations,
        "freeze_policy.blocked_sources",
        "dataset_scope_freeze_policy.json blocked_sources must be exactly STEDT/CogNet",
    )

    _must(
        freeze_check.get("status") == "PASS"
        and freeze_check.get("blocked_sources_present") is True,
        checks,
        violations,
        "freeze_check.status",
        "dataset_scope_freeze_check.json must PASS with blocked_sources_present=true",
    )

    blocked_execution = mapping.get("blocked_execution")
    mapping_block_ok = True
    if not isinstance(blocked_execution, dict):
        mapping_block_ok = False
    else:
        for source in sorted(EXPECTED_BLOCKED_SOURCES):
            entry = blocked_execution.get(source)
            if not isinstance(entry, dict) or entry.get("status") != "blocked_by_freeze":
                mapping_block_ok = False
                break
    _must(
        mapping_block_ok,
        checks,
        violations,
        "mapping.blocked_execution",
        "directive_mapping.json blocked_execution must keep STEDT/CogNet blocked_by_freeze",
    )

    summary = bundle.get("summary")
    summary_ok = (
        isinstance(summary, dict)
        and summary.get("scope_freeze_active") is True
        and summary.get("STEDT_status") == "blocked_placeholder"
        and summary.get("CogNet_status") == "blocked_placeholder"
    )
    _must(
        summary_ok,
        checks,
        violations,
        "bundle.summary.blocked_placeholder",
        "directive_execution_bundle.json summary must retain blocked placeholder statuses for STEDT/CogNet",
    )


def _check_mapping_coverage(mapping: dict[str, Any], checks: list[dict[str, Any]], violations: list[str]) -> None:
    directives = mapping.get("directives")
    mapping_status = mapping.get("mapping_status")
    if not isinstance(directives, list) or not isinstance(mapping_status, dict):
        _must(False, checks, violations, "mapping.coverage", "directives list and mapping_status must exist")
        return

    directive_ids = {d.get("id") for d in directives if isinstance(d, dict)}
    _must(
        directive_ids == {"directive_1", "directive_2", "directive_3"},
        checks,
        violations,
        "mapping.directive_ids",
        "directive_mapping.json must include directive_1, directive_2, directive_3",
    )

    in_scope_count = sum(1 for d in directives if isinstance(d, dict) and d.get("in_scope") is True)
    deferred_count = sum(1 for d in directives if isinstance(d, dict) and d.get("deferred") is True)

    _must(
        int(mapping_status.get("total_directives", -1)) == len(directives)
        and int(mapping_status.get("in_scope_count", -1)) == in_scope_count
        and int(mapping_status.get("deferred_count", -1)) == deferred_count,
        checks,
        violations,
        "mapping.count_consistency",
        "mapping_status counts must match directives payload",
    )


def _check_repro_gate(repro_gate: dict[str, Any], checks: list[dict[str, Any]], violations: list[str]) -> None:
    _must(
        repro_gate.get("status") == "PASS"
        and repro_gate.get("determinism_pass") is True
        and repro_gate.get("accounting_consistent") is True,
        checks,
        violations,
        "repro_gate.status",
        "repro_gate_check.json must PASS with determinism_pass=true and accounting_consistent=true",
    )

    comparisons = repro_gate.get("checksum_comparisons")
    all_equal = False
    if isinstance(comparisons, list) and len(comparisons) > 0:
        all_equal = True
        for item in comparisons:
            if not isinstance(item, dict) or item.get("equal") is not True:
                all_equal = False
                break
    _must(
        all_equal,
        checks,
        violations,
        "repro_gate.checksum_comparisons",
        "all checksum_comparisons entries must report equal=true",
    )


def _check_bundle_and_table(
    bundle: dict[str, Any],
    table_header: list[str],
    table_rows: list[dict[str, str]],
    checks: list[dict[str, Any]],
    violations: list[str],
) -> None:
    tasks = bundle.get("tasks")
    _must(
        isinstance(tasks, dict),
        checks,
        violations,
        "bundle.tasks",
        "directive_execution_bundle.json must contain tasks object",
    )
    if not isinstance(tasks, dict):
        return

    task5 = tasks.get("task5_directive_bundle")
    referenced_paths: list[str] = []
    if isinstance(task5, dict):
        raw_paths = task5.get("referenced_paths")
        if isinstance(raw_paths, list):
            referenced_paths = [str(p) for p in raw_paths]

    task5_ok = (
        isinstance(task5, dict)
        and task5.get("verification_status") == "PASS"
        and len(referenced_paths) >= 2
        and "runs/directive_bundle/directive_execution_bundle.json" in referenced_paths
        and "runs/directive_bundle/directive_execution_table.csv" in referenced_paths
    )
    _must(
        task5_ok,
        checks,
        violations,
        "bundle.task5.paths",
        "task5_directive_bundle must reference bundle/table paths with PASS verification status",
    )

    scope_note = bundle.get("scope_note")
    scope_text = scope_note if isinstance(scope_note, str) else ""
    scope_note_ok = (
        "STEDT" in scope_text
        and "CogNet" in scope_text
        and "out-of-scope" in scope_text
        and "placeholder" in scope_text
    )
    _must(
        scope_note_ok,
        checks,
        violations,
        "bundle.scope_note.contract",
        "scope_note must explicitly include STEDT/CogNet as out-of-scope placeholders",
    )

    _must(
        EXPECTED_TABLE_COLUMNS.issubset(set(table_header)),
        checks,
        violations,
        "table.header.contract",
        "directive_execution_table.csv header must contain required contract columns",
    )

    rows_by_task = {
        row.get("task_id", ""): row
        for row in table_rows
        if isinstance(row, dict) and row.get("task_id")
    }
    task3 = rows_by_task.get("directive_3_open_dataset_prep")
    task3_ok = (
        isinstance(task3, dict)
        and str(task3.get("status", "")).strip().lower() == "deferred"
        and str(task3.get("blocked_sources", "")).replace(" ", "") in {"STEDT,CogNet", "CogNet,STEDT"}
    )
    _must(
        task3_ok,
        checks,
        violations,
        "table.directive_3.blocked",
        "directive_3_open_dataset_prep row must be deferred with STEDT,CogNet blocked_sources",
    )


def _check_lock(lock_verify: dict[str, Any], checks: list[dict[str, Any]], violations: list[str]) -> None:
    _must(
        lock_verify.get("status") == "PASS" and int(lock_verify.get("drift_count", -1)) == 0,
        checks,
        violations,
        "lock_verify.status",
        "lock_verify.json must PASS with drift_count=0",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Directive bundle strict gate checker")
    _ = parser.add_argument("--strict", action="store_true", help="Fail on any violation")
    _ = parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Output JSON report path",
    )
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []
    violations: list[str] = []

    freeze_policy = _load_json_object(FREEZE_POLICY_PATH)
    freeze_check = _load_json_object(FREEZE_CHECK_PATH)
    mapping = _load_json_object(MAPPING_PATH)
    repro_gate = _load_json_object(REPRO_GATE_PATH)
    bundle = _load_json_object(BUNDLE_PATH)
    lock_verify = _load_json_object(LOCK_VERIFY_PATH)
    table_header, table_rows = _parse_table(TABLE_PATH)

    _check_freeze_policy(freeze_policy, freeze_check, mapping, bundle, checks, violations)
    _check_mapping_coverage(mapping, checks, violations)
    _check_repro_gate(repro_gate, checks, violations)
    _check_bundle_and_table(bundle, table_header, table_rows, checks, violations)
    _check_lock(lock_verify, checks, violations)

    has_violation = len(violations) > 0
    status = "PASS" if not has_violation else ("FAIL" if args.strict else "WARN")

    report = {
        "status": status,
        "checks": checks,
        "violations": violations,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"status={status}")
    print(f"check_count={len(checks)}")
    print(f"violation_count={len(violations)}")
    print(f"report={args.report}")

    if has_violation and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
