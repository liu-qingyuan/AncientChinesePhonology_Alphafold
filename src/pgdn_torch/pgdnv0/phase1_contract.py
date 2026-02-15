from __future__ import annotations

from pathlib import Path


PHASE1_REQUIRED_FIELDS: tuple[str, ...] = (
    "record_id",
    "graph_id",
    "target_vector",
    "mask",
    "source",
)

ACP_LEGACY_REQUIRED_FIELDS: tuple[str, ...] = (
    "record_id",
    "graph_id",
    "target_vector",
    "mask",
)

PHASE1_RECORD_ID_PREFIXES: tuple[str, ...] = (
    "acp:",
    "mocha:",
    "tshet:",
    "unihan:",
)


def _is_acp_legacy_record_id(record_id: object) -> bool:
    if not isinstance(record_id, str):
        return False
    if record_id.count(":") != 1:
        return False
    left, right = record_id.split(":", 1)
    if not left or not right:
        return False
    return not any(record_id.startswith(p) for p in PHASE1_RECORD_ID_PREFIXES)


def is_acp_legacy_row(row: object) -> bool:
    if not isinstance(row, dict):
        return False
    source = row.get("source")
    if source not in (None, ""):
        return False
    return _is_acp_legacy_record_id(row.get("record_id"))


def validate_phase1_target_row(row: object, *, allow_acp_legacy: bool = True) -> list[str]:
    errors: list[str] = []
    if not isinstance(row, dict):
        return ["row must be a JSON object"]

    if allow_acp_legacy and is_acp_legacy_row(row):
        missing = [k for k in ACP_LEGACY_REQUIRED_FIELDS if k not in row]
        if missing:
            errors.append(f"missing required keys={missing}")
        return errors

    missing = [k for k in PHASE1_REQUIRED_FIELDS if k not in row]
    if missing:
        errors.append(f"missing required keys={missing}")

    rid = row.get("record_id")
    if not isinstance(rid, str) or not rid:
        errors.append("record_id must be a non-empty string")
    elif not any(rid.startswith(p) for p in PHASE1_RECORD_ID_PREFIXES):
        errors.append(
            f"record_id must start with one of {PHASE1_RECORD_ID_PREFIXES}: {rid!r}"
        )

    source = row.get("source")
    if not isinstance(source, str) or not source:
        errors.append("source must be a non-empty string")

    return errors


def default_phase1_target_candidates(repo_root: Path) -> list[Path]:
    canonical = [
        repo_root / "data" / "targets" / "acp_targets.jsonl",
        repo_root / "data" / "targets" / "mocha_sidecar_targets.jsonl",
        repo_root / "data" / "targets" / "tshet_uinh_targets.jsonl",
        repo_root / "data" / "targets" / "unihan_targets.jsonl",
    ]
    discovered = sorted((repo_root / "runs").glob("**/*_targets.jsonl"))

    out: list[Path] = []
    seen: set[Path] = set()
    for p in canonical + discovered:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def default_phase1_split_candidates(repo_root: Path) -> list[Path]:
    canonical = [
        repo_root / "data" / "splits" / "manifest.json",
        repo_root / "data" / "splits" / "mocha_sidecar_splits.json",
        repo_root / "data" / "splits" / "tshet_uinh_splits.json",
        repo_root / "data" / "splits" / "unihan_splits.json",
    ]
    discovered = sorted((repo_root / "runs").glob("**/*_splits.json"))

    out: list[Path] = []
    seen: set[Path] = set()
    for p in canonical + discovered:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out
