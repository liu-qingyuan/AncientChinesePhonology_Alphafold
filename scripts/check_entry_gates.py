#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import cast


ROOT = Path(__file__).resolve().parents[1]

DIRECTIVE2_APPROVAL_PATH = ROOT / "runs" / "approvals" / "directive2_redesign_approval.json"
DIRECTIVE3_APPROVAL_PATH = ROOT / "runs" / "approvals" / "directive3_freeze_exception_approval.json"
STRICT_GATE_REPORT_PATH = ROOT / "runs" / "next_phase_gate" / "gate_check_report.json"


def _load_json_object(path: Path) -> dict[str, object]:
    raw_obj = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw_obj, dict):
        raise SystemExit(f"expected JSON object in {path}")
    return cast(dict[str, object], raw_obj)


def _check(condition: bool, name: str, detail: str, checks: list[dict[str, object]], violations: list[str]) -> None:
    status = "PASS" if condition else "FAIL"
    checks.append({"name": name, "status": status, "detail": detail})
    if not condition:
        violations.append(f"{name}: {detail}")


def _is_approved(obj: dict[str, object]) -> bool:
    return str(obj.get("status", "")).upper() == "APPROVED" and bool(obj.get("approved")) is True


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent D2/D3 entry gate checker")
    _ = parser.add_argument(
        "--decision-out",
        type=Path,
        default=ROOT / "runs" / "next_phase_entry" / "entry_gate_decision.json",
        help="Output decision JSON path",
    )
    _ = parser.add_argument(
        "--report-out",
        type=Path,
        default=ROOT / "runs" / "next_phase_gate" / "entry_gates_report.json",
        help="Output gate report JSON path",
    )
    _ = parser.add_argument("--strict", action="store_true", help="Return non-zero on any gate violation")
    args = parser.parse_args()
    decision_out = cast(Path, args.decision_out)
    report_out = cast(Path, args.report_out)
    strict_mode = cast(bool, args.strict)

    checks: list[dict[str, object]] = []
    violations: list[str] = []

    d2_approval = _load_json_object(DIRECTIVE2_APPROVAL_PATH)
    d3_approval = _load_json_object(DIRECTIVE3_APPROVAL_PATH)
    strict_gate = _load_json_object(STRICT_GATE_REPORT_PATH)

    strict_gate_ok = str(strict_gate.get("status", "")).upper() == "PASS"
    d2_approved = _is_approved(d2_approval)
    d3_exception_approved = _is_approved(d3_approval)

    _check(
        strict_gate_ok,
        "strict_gate.status",
        "gate_check_report.json status must be PASS before directive entry",
        checks,
        violations,
    )
    _check(
        d2_approved,
        "directive2.redesign_approval",
        "directive2_redesign_approval.json must be APPROVED with approved=true",
        checks,
        violations,
    )
    _check(
        d3_exception_approved,
        "directive3.freeze_exception_approval",
        "directive3_freeze_exception_approval.json must be APPROVED with approved=true for activation",
        checks,
        violations,
    )

    directive2_reasons: list[str] = []
    directive3_reasons: list[str] = []

    if not strict_gate_ok:
        directive2_reasons.append("strict_gate_not_pass")
        directive3_reasons.append("strict_gate_not_pass")
    if not d2_approved:
        directive2_reasons.append("redesign_approval_required")
    if not d3_exception_approved:
        directive3_reasons.append("freeze_exception_approval_required")

    directive2_entry_ready = strict_gate_ok and d2_approved
    directive3_entry_ready = strict_gate_ok and d3_exception_approved

    checked_at_utc = datetime.now(timezone.utc).isoformat()
    has_violation = len(violations) > 0
    report_status = "PASS" if not has_violation else ("FAIL" if strict_mode else "WARN")

    decision = {
        "artifact": "entry_gate_decision_v1",
        "directive2_entry_ready": directive2_entry_ready,
        "directive3_entry_ready": directive3_entry_ready,
        "gate_status": "READY" if (directive2_entry_ready or directive3_entry_ready) else "BLOCKED",
        "reasons": {
            "directive2": directive2_reasons,
            "directive3": directive3_reasons,
        },
        "checked_at_utc": checked_at_utc,
        "references": {
            "directive2_approval": "runs/approvals/directive2_redesign_approval.json",
            "directive3_approval": "runs/approvals/directive3_freeze_exception_approval.json",
            "strict_gate_report": "runs/next_phase_gate/gate_check_report.json",
        },
    }

    report = {
        "status": report_status,
        "directive2_entry_ready": directive2_entry_ready,
        "directive3_entry_ready": directive3_entry_ready,
        "checks": checks,
        "violations": violations,
        "checked_at_utc": checked_at_utc,
    }

    _ = decision_out.parent.mkdir(parents=True, exist_ok=True)
    _ = report_out.parent.mkdir(parents=True, exist_ok=True)
    _ = decision_out.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _ = report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"status={report_status}")
    print(f"directive2_entry_ready={directive2_entry_ready}")
    print(f"directive3_entry_ready={directive3_entry_ready}")
    print(f"report={report_out}")
    print(f"decision={decision_out}")

    if has_violation and strict_mode:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
