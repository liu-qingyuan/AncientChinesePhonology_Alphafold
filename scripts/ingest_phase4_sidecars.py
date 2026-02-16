#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STATUS_SUCCESS = "success"
STATUS_BLOCKED_ACCESS = "blocked_access"
STATUS_MISSING_SOURCE = "missing_source"
ALLOWED_STATUSES = {STATUS_SUCCESS, STATUS_BLOCKED_ACCESS, STATUS_MISSING_SOURCE}

BLOCKED_MARKERS = (
    "ACCESS_BLOCKED",
    "BLOCKED_ACCESS",
    "LICENSE_REQUIRED",
    "RESTRICTED",
)

TABULAR_SUFFIXES = {".csv", ".tsv", ".txt"}
JSONL_SUFFIXES = {".jsonl"}

CHAR_KEYS = (
    "character",
    "word",
    "concept",
    "form",
    "value",
    "orthography",
)
IPA_KEYS = (
    "ipa",
    "phonetic",
    "pronunciation",
    "segments",
    "transcription",
    "form",
    "value",
)
LANG_KEYS = (
    "language",
    "lang",
    "doculect",
    "iso",
    "iso_code",
    "lect",
)


@dataclass(frozen=True)
class SourceSpec:
    key: str
    label: str
    source_url: str
    license_note: str
    access_note: str


@dataclass(frozen=True)
class ExtractedRow:
    source_file: str
    source_line: int
    character: str
    ipa: str
    language: str


SOURCES = (
    SourceSpec(
        key="cldf",
        label="CLDF",
        source_url="https://cldf.clld.org/",
        license_note="Varies by CLDF dataset; preserve upstream metadata/license files.",
        access_note="Open format, but access/licensing remain dataset-specific.",
    ),
    SourceSpec(
        key="stedt",
        label="STEDT",
        source_url="https://stedt.berkeley.edu/",
        license_note="STEDT materials may have usage restrictions; confirm upstream terms.",
        access_note="Some resources may require explicit permission or controlled access.",
    ),
    SourceSpec(
        key="cognet",
        label="CogNet",
        source_url="https://aclanthology.org/2023.acl-long.712/",
        license_note="Check the specific release terms for the local CogNet snapshot.",
        access_note="Access varies by release channel and local availability.",
    ),
)


def _hash_to_unit_floats(key: str, n: int) -> list[float]:
    out: list[float] = []
    counter = 0
    while len(out) < n:
        digest = hashlib.sha256(f"{key}|{counter}".encode("utf-8")).digest()
        for b in digest:
            out.append((float(b) / 127.5) - 1.0)
            if len(out) >= n:
                break
        counter += 1
    return out


def _slot_vector(value: str) -> list[float]:
    key = value.strip()
    if not key:
        return [0.0] * 8
    return _hash_to_unit_floats(key, 8)


def _normalize_key(k: str) -> str:
    return k.strip().lower().replace(" ", "_")


def _pick_first(row: dict[str, str], candidates: tuple[str, ...]) -> str:
    for key in candidates:
        value = row.get(key, "")
        if value.strip():
            return value.strip()
    return ""


def _iter_candidate_files(source_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in TABULAR_SUFFIXES or suffix in JSONL_SUFFIXES:
            files.append(path)
    return sorted(files, key=lambda p: str(p.relative_to(source_root)))


def _extract_from_tabular(path: Path, source_root: Path, remaining: int) -> list[ExtractedRow]:
    out: list[ExtractedRow] = []
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            return out
        norm_names = [_normalize_key(name) for name in fieldnames]
        for row_idx, raw in enumerate(reader, start=2):
            if remaining <= 0:
                break
            norm_row: dict[str, str] = {}
            for src_name, norm_name in zip(fieldnames, norm_names):
                value = raw.get(src_name)
                norm_row[norm_name] = "" if value is None else str(value).strip()

            character = _pick_first(norm_row, CHAR_KEYS)
            ipa = _pick_first(norm_row, IPA_KEYS)
            language = _pick_first(norm_row, LANG_KEYS)
            if not character and not ipa:
                continue

            out.append(
                ExtractedRow(
                    source_file=str(path.relative_to(source_root)),
                    source_line=row_idx,
                    character=character,
                    ipa=ipa,
                    language=language,
                )
            )
            remaining -= 1
    return out


def _extract_from_jsonl(path: Path, source_root: Path, remaining: int) -> list[ExtractedRow]:
    out: list[ExtractedRow] = []
    with path.open("r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f, start=1):
            if remaining <= 0:
                break
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict):
                continue

            norm_row = {_normalize_key(str(k)): str(v).strip() for k, v in raw.items()}
            character = _pick_first(norm_row, CHAR_KEYS)
            ipa = _pick_first(norm_row, IPA_KEYS)
            language = _pick_first(norm_row, LANG_KEYS)
            if not character and not ipa:
                continue

            out.append(
                ExtractedRow(
                    source_file=str(path.relative_to(source_root)),
                    source_line=row_idx,
                    character=character,
                    ipa=ipa,
                    language=language,
                )
            )
            remaining -= 1
    return out


def _extract_rows(source_root: Path, limit: int) -> tuple[list[ExtractedRow], list[str]]:
    rows: list[ExtractedRow] = []
    used_files: list[str] = []
    files = _iter_candidate_files(source_root)
    remaining = max(int(limit), 0)

    for path in files:
        if remaining <= 0:
            break
        suffix = path.suffix.lower()
        if suffix in TABULAR_SUFFIXES:
            extracted = _extract_from_tabular(path, source_root=source_root, remaining=remaining)
        elif suffix in JSONL_SUFFIXES:
            extracted = _extract_from_jsonl(path, source_root=source_root, remaining=remaining)
        else:
            extracted = []
        if extracted:
            used_files.append(str(path.relative_to(source_root)))
            rows.extend(extracted)
            remaining = max(remaining - len(extracted), 0)

    return rows, sorted(set(used_files))


def _record_id(dataset_key: str, row: ExtractedRow) -> str:
    payload = "\t".join([dataset_key, row.source_file, str(row.source_line), row.character, row.ipa, row.language])
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{dataset_key}:{digest}"


def _to_sidecar_row(dataset_key: str, row: ExtractedRow) -> dict[str, object]:
    rid = _record_id(dataset_key, row)
    source_key = "|".join([row.source_file, str(row.source_line), row.language])
    target_vector = (
        _slot_vector(row.character)
        + _slot_vector(row.ipa)
        + _slot_vector(row.language)
        + _slot_vector(source_key)
    )
    mask = {
        "I": 1 if row.character else 0,
        "M": 1 if row.ipa else 0,
        "N": 1 if row.language else 0,
        "C": 1,
    }
    return {
        "record_id": rid,
        "graph_id": f"graph:{dataset_key}:{row.language or 'unknown'}",
        "source": f"{dataset_key}_sidecar_v1",
        "character": row.character,
        "ipa": row.ipa,
        "language": row.language,
        "source_file": row.source_file,
        "source_line": int(row.source_line),
        "target_vector": target_vector,
        "mask": mask,
    }


def _deterministic_splits(record_ids: list[str]) -> dict[str, dict[str, list[str]]]:
    ordered = sorted(record_ids)
    n = len(ordered)
    if n == 0:
        split = {"train": [], "dev": [], "test": []}
        return {"random": split, "temporal": split}
    if n == 1:
        split = {"train": ordered[:], "dev": [], "test": ordered[:1]}
        return {"random": split, "temporal": split}

    n_train = max(1, int(n * 0.7))
    n_dev = max(1, int(n * 0.15))
    if n_train + n_dev >= n:
        n_dev = 1
        n_train = max(1, n - 2)
    split = {
        "train": ordered[:n_train],
        "dev": ordered[n_train : n_train + n_dev],
        "test": ordered[n_train + n_dev :],
    }
    if not split["test"]:
        split["test"] = ordered[-1:]
    return {"random": split, "temporal": split}


def _validate_report_schema(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(report.get("sources"), list):
        return ["sources must be a list"]

    status_coverage = report.get("status_coverage")
    if not isinstance(status_coverage, list):
        errors.append("status_coverage must be a list")

    for i, row in enumerate(report["sources"]):
        if not isinstance(row, dict):
            errors.append(f"sources[{i}] must be an object")
            continue
        key = row.get("key")
        status = row.get("status")
        reason = row.get("reason")
        link_path = row.get("link_path")
        sample_count = row.get("sample_count")
        if not isinstance(key, str) or not key:
            errors.append(f"sources[{i}].key must be a non-empty string")
        if status not in ALLOWED_STATUSES:
            errors.append(f"sources[{i}].status must be one of {sorted(ALLOWED_STATUSES)}")
        if not isinstance(reason, str) or not reason:
            errors.append(f"sources[{i}].reason must be a non-empty string")
        if not isinstance(link_path, str) or not link_path.startswith("data/external/links/"):
            errors.append(f"sources[{i}].link_path must be under data/external/links/")
        if not isinstance(sample_count, int) or sample_count < 0:
            errors.append(f"sources[{i}].sample_count must be a non-negative int")
        if status == STATUS_SUCCESS and isinstance(sample_count, int) and sample_count <= 0:
            errors.append(f"sources[{i}] success requires sample_count > 0")

    return errors


def _resolve_link_path(spec_key: str, links_dir: Path) -> tuple[Path, dict[str, Any]]:
    preferred = links_dir / spec_key
    candidates: list[Path] = [preferred]
    policy = "direct"

    if spec_key == "cldf":
        canonical = links_dir / "clts"
        alias = links_dir / "cldf"
        candidates = [canonical, alias]
        policy = "cldf_compat_prefer_clts"

    selected = candidates[0]
    for path in candidates:
        if path.exists() or path.is_symlink():
            selected = path
            break

    return selected, {
        "policy": policy,
        "candidates": [str(path) for path in candidates],
        "selected": str(selected),
    }


def _ingest_source(
    spec: SourceSpec,
    links_dir: Path,
    sidecars_dir: Path,
    limit: int,
) -> dict[str, Any]:
    link_path, resolution_meta = _resolve_link_path(spec.key, links_dir)
    if link_path.is_absolute():
        try:
            link_path_rel = str(link_path.relative_to(Path.cwd().resolve()))
        except ValueError:
            link_path_rel = str(link_path)
    else:
        link_path_rel = str(link_path)
    row: dict[str, Any] = {
        "key": spec.key,
        "label": spec.label,
        "status": STATUS_MISSING_SOURCE,
        "reason": "source link missing",
        "link_path": link_path_rel,
        "resolved_source_path": None,
        "sample_count": 0,
        "outputs": None,
        "resolution": resolution_meta,
        "provenance": {
            "source_url": spec.source_url,
            "license_note": spec.license_note,
            "access_note": spec.access_note,
        },
    }

    if not link_path.exists() and not link_path.is_symlink():
        if spec.key == "cldf":
            row["reason"] = "source link missing (checked canonical clts then cldf alias)"
        return row

    if spec.key == "cldf" and resolution_meta.get("policy") == "cldf_compat_prefer_clts":
        selected = str(resolution_meta.get("selected", ""))
        canonical = str(links_dir / "clts")
        alias = str(links_dir / "cldf")
        if selected == canonical:
            row["reason"] = "resolved via canonical clts link (preferred over cldf alias)"
        elif selected == alias:
            row["reason"] = "resolved via cldf alias link (canonical clts unavailable)"

    try:
        source_root = link_path.resolve(strict=True)
    except FileNotFoundError:
        row["status"] = STATUS_BLOCKED_ACCESS
        row["reason"] = "link exists but target is unavailable"
        return row

    row["resolved_source_path"] = str(source_root)
    if not os.access(source_root, os.R_OK):
        row["status"] = STATUS_BLOCKED_ACCESS
        row["reason"] = "source path is not readable"
        return row

    for marker in BLOCKED_MARKERS:
        if (source_root / marker).exists():
            row["status"] = STATUS_BLOCKED_ACCESS
            row["reason"] = f"blocked marker present: {marker}"
            return row

    extracted, used_files = _extract_rows(source_root=source_root, limit=limit)
    if not extracted:
        row["status"] = STATUS_BLOCKED_ACCESS
        row["reason"] = "no supported lexical records found"
        return row

    sidecars_dir.mkdir(parents=True, exist_ok=True)
    targets_path = sidecars_dir / f"{spec.key}_targets.jsonl"
    splits_path = sidecars_dir / f"{spec.key}_splits.json"
    meta_path = sidecars_dir / f"{spec.key}_meta.json"

    sidecar_rows = [_to_sidecar_row(spec.key, item) for item in extracted]
    with targets_path.open("w", encoding="utf-8") as f:
        for sidecar_row in sidecar_rows:
            f.write(json.dumps(sidecar_row, ensure_ascii=False) + "\n")

    record_ids = [str(item["record_id"]) for item in sidecar_rows]
    splits_obj = _deterministic_splits(record_ids)
    splits_path.write_text(json.dumps(splits_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta_obj = {
        "dataset": spec.key,
        "source_root": str(source_root),
        "link_path": link_path_rel,
        "rows_emitted": len(sidecar_rows),
        "files_used": used_files,
        "limit": int(limit),
        "license_note": spec.license_note,
        "access_note": spec.access_note,
    }
    meta_path.write_text(json.dumps(meta_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    row["status"] = STATUS_SUCCESS
    if spec.key == "cldf" and resolution_meta.get("policy") == "cldf_compat_prefer_clts":
        selected = str(resolution_meta.get("selected", ""))
        canonical = str(links_dir / "clts")
        if selected == canonical:
            row["reason"] = "ingestion completed via canonical clts link"
        else:
            row["reason"] = "ingestion completed via cldf alias link"
    else:
        row["reason"] = "ingestion completed"
    row["sample_count"] = int(len(sidecar_rows))
    row["outputs"] = {
        "targets": str(targets_path),
        "splits": str(splits_path),
        "meta": str(meta_path),
    }
    return row


def build_ingestion_report(
    links_dir: Path,
    out_dir: Path,
    per_source_limit: int,
) -> dict[str, Any]:
    sidecars_dir = out_dir / "sidecars"
    source_rows = [
        _ingest_source(
            spec=spec,
            links_dir=links_dir,
            sidecars_dir=sidecars_dir,
            limit=per_source_limit,
        )
        for spec in SOURCES
    ]
    status_coverage = sorted({str(row["status"]) for row in source_rows})
    report: dict[str, Any] = {
        "phase": "paper_gap_phase4",
        "policy": {
            "dataset_entrypoint": "data/external/links/",
            "allowed_statuses": sorted(ALLOWED_STATUSES),
            "families": [spec.key for spec in SOURCES],
        },
        "sources": source_rows,
        "status_coverage": status_coverage,
    }
    errors = _validate_report_schema(report)
    report["schema_validation"] = {"ok": not errors, "errors": errors}
    return report


def run_loader_smoke(report: dict[str, Any]) -> dict[str, Any]:
    try:
        module = importlib.import_module("pgdn_torch.pgdnv0.data")
        ACPJsonlDataset = getattr(module, "ACPJsonlDataset")
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"failed_import_loader: {exc}",
            "successful_sources": [],
            "total_samples": 0,
            "details": [],
        }

    details: list[dict[str, Any]] = []
    successful_sources: list[str] = []
    total_samples = 0

    source_rows = report.get("sources")
    if not isinstance(source_rows, list):
        return {
            "ok": False,
            "reason": "invalid ingestion report: sources missing",
            "successful_sources": [],
            "total_samples": 0,
            "details": [],
        }

    for src in source_rows:
        if not isinstance(src, dict):
            continue
        key = str(src.get("key", ""))
        status = str(src.get("status", ""))
        if status != STATUS_SUCCESS:
            continue
        outputs = src.get("outputs")
        if not isinstance(outputs, dict):
            details.append({"key": key, "ok": False, "reason": "missing outputs for successful source"})
            continue
        targets_path_raw = outputs.get("targets")
        if not isinstance(targets_path_raw, str):
            details.append({"key": key, "ok": False, "reason": "missing targets path"})
            continue
        targets_path = Path(targets_path_raw)
        if not targets_path.exists():
            details.append({"key": key, "ok": False, "reason": f"targets not found: {targets_path}"})
            continue

        ds = ACPJsonlDataset(path=targets_path)
        sample_count = len(ds)
        total_samples += int(sample_count)
        ok = sample_count > 0
        details.append(
            {
                "key": key,
                "status": status,
                "targets": str(targets_path),
                "sample_count": int(sample_count),
                "ok": bool(ok),
            }
        )
        successful_sources.append(key)

    if successful_sources:
        all_ok = all(bool(item.get("ok")) for item in details)
        reason = "loader smoke passed" if all_ok and total_samples > 0 else "one or more successful sidecars failed loader smoke"
        return {
            "ok": bool(all_ok and total_samples > 0),
            "reason": reason,
            "successful_sources": sorted(successful_sources),
            "total_samples": int(total_samples),
            "details": details,
        }

    return {
        "ok": True,
        "reason": "no successful sidecars to smoke",
        "successful_sources": [],
        "total_samples": 0,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Task4 ingestion adapters for CLDF/STEDT/CogNet. "
            "Writes ingestion_report.json and loader_smoke.json under runs/paper_gap_phase4."
        )
    )
    parser.add_argument("--links-dir", type=Path, default=Path("data/external/links"))
    parser.add_argument("--out", type=Path, default=Path("runs/paper_gap_phase4"))
    parser.add_argument("--per-source-limit", type=int, default=256)
    parser.add_argument(
        "--skip-loader-smoke",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip loader smoke generation (default: run loader smoke)",
    )
    args = parser.parse_args()

    links_dir = Path(args.links_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_ingestion_report(
        links_dir=links_dir,
        out_dir=out_dir,
        per_source_limit=max(int(args.per_source_limit), 0),
    )
    report_path = out_dir / "ingestion_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    loader_smoke_path = out_dir / "loader_smoke.json"
    if args.skip_loader_smoke:
        loader_smoke = {
            "ok": True,
            "reason": "skipped by --skip-loader-smoke",
            "successful_sources": [],
            "total_samples": 0,
            "details": [],
        }
    else:
        loader_smoke = run_loader_smoke(report)
    loader_smoke_path.write_text(json.dumps(loader_smoke, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "ingestion_report": str(report_path),
                "loader_smoke": str(loader_smoke_path),
                "schema_ok": bool(report.get("schema_validation", {}).get("ok", False)),
                "loader_smoke_ok": bool(loader_smoke.get("ok", False)),
            },
            ensure_ascii=False,
        )
    )
    if not bool(report.get("schema_validation", {}).get("ok", False)):
        return 1
    if not bool(loader_smoke.get("ok", False)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
