from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_contract(repo_root: Path) -> tuple[bool, dict[str, object]]:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "validate_phase1_contract.py"),
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        proc.returncode == 0,
        {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "command": " ".join(cmd),
        },
    )


def _check_sidecars(repo_root: Path) -> tuple[bool, dict[str, object]]:
    required = {
        "tshet_builder_meta": repo_root / "runs/multidataset_phase1/sidecars/tshet_uinh_builder_meta.json",
        "unihan_builder_meta": repo_root / "runs/multidataset_phase1/sidecars/unihan_builder_meta.json",
        "tshet_smoke_checkpoint": repo_root / "runs/multidataset_phase1/tshet_smoke_train/checkpoint_none.pt",
        "unihan_smoke_checkpoint": repo_root / "runs/multidataset_phase1/unihan_smoke_train/checkpoint_none.pt",
    }
    existence = {k: p.exists() for k, p in required.items()}
    ok = all(existence.values())
    return ok, {"ok": ok, "artifacts": existence}


def _check_source_eval(repo_root: Path) -> tuple[bool, dict[str, object]]:
    summary_path = repo_root / "runs/multidataset_phase1/eval/summary.json"
    if not summary_path.exists():
        return False, {"ok": False, "summary_path": str(summary_path), "reason": "missing"}

    summary = _load_json(summary_path)
    pooled = summary.get("pooled")
    by_source = summary.get("by_source")
    has_required = isinstance(pooled, dict) and isinstance(by_source, dict)
    source_count = len(by_source) if isinstance(by_source, dict) else 0
    ranking = None
    if isinstance(pooled, dict):
        ranking = pooled.get("ranking_score_mean")
    ok = bool(has_required and source_count >= 3)
    return (
        ok,
        {
            "ok": ok,
            "summary_path": str(summary_path),
            "has_pooled": isinstance(pooled, dict),
            "by_source_count": source_count,
            "pooled_ranking_score_mean": ranking,
        },
    )


def _check_leakage(repo_root: Path) -> tuple[bool, dict[str, object]]:
    clean_path = (
        repo_root
        / ".sisyphus/notepads/multidataset-integration-phase1/leakage_summary_clean.json"
    )
    injected_path = (
        repo_root
        / ".sisyphus/notepads/multidataset-integration-phase1/leakage_summary_injected.json"
    )
    if not clean_path.exists() or not injected_path.exists():
        return (
            False,
            {
                "ok": False,
                "clean_exists": clean_path.exists(),
                "injected_exists": injected_path.exists(),
            },
        )

    clean = _load_json(clean_path)
    injected = _load_json(injected_path)
    clean_status = clean.get("status")
    injected_status = injected.get("status")
    collisions = clean.get("record_id_collisions") if isinstance(clean, dict) else None
    record_id_collision_count = None
    if isinstance(collisions, dict):
        record_id_collision_count = collisions.get("count")
    ok = clean_status == "PASS" and injected_status == "FAIL"
    return (
        ok,
        {
            "ok": ok,
            "clean_summary_path": str(clean_path),
            "injected_summary_path": str(injected_path),
            "clean_status": clean_status,
            "injected_status": injected_status,
            "clean_record_id_collision_count": record_id_collision_count,
        },
    )


def _check_acp_non_regression(repo_root: Path) -> tuple[bool, dict[str, object]]:
    path = repo_root / "runs/multidataset_phase1/acp_non_regression.json"
    if not path.exists():
        return False, {"ok": False, "path": str(path), "reason": "missing"}

    payload = _load_json(path)
    gate_decision = payload.get("gate_decision")
    relative_degradation = payload.get("relative_degradation")
    threshold = payload.get("threshold")
    ok = gate_decision == "GO"
    return (
        ok,
        {
            "ok": ok,
            "path": str(path),
            "gate_decision": gate_decision,
            "relative_degradation": relative_degradation,
            "threshold": threshold,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble multidataset phase1 gate summary")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/multidataset_phase1/gate_summary.json"),
        help="output gate summary path",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    checks: dict[str, dict[str, object]] = {}
    blockers: list[str] = []

    for name, fn in (
        ("contract_gate", _check_contract),
        ("sidecar_evidence", _check_sidecars),
        ("source_eval_summary", _check_source_eval),
        ("leakage_gate", _check_leakage),
        ("acp_non_regression_gate", _check_acp_non_regression),
    ):
        ok, detail = fn(repo_root)
        checks[name] = detail
        if not ok:
            blockers.append(name)

    decision = "GO" if not blockers else "NO_GO"
    gate_summary = {
        "decision": decision,
        "blockers": blockers,
        "checks": checks,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gate_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"gate_decision={decision}")
    print(f"gate_summary={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
