from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import json
import sys
from pathlib import Path

from pgdn_torch.pgdnv0.phase1_contract import (
    ACP_LEGACY_REQUIRED_FIELDS,
    PHASE1_REQUIRED_FIELDS,
    PHASE1_RECORD_ID_PREFIXES,
    default_phase1_split_candidates,
    default_phase1_target_candidates,
    validate_phase1_target_row,
)
from pgdn_torch.pgdnv0.splits import load_split_ids


def _iter_sampled_rows(path: Path, sample_size: int) -> list[tuple[int, object]]:
    rows: list[tuple[int, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i > sample_size:
                break
            line = line.strip()
            if not line:
                continue
            rows.append((i, json.loads(line)))
    return rows


def _validate_target_file(path: Path, sample_size: int, acp_legacy_compat: bool) -> list[str]:
    errors: list[str] = []
    rows = _iter_sampled_rows(path, sample_size=sample_size)
    if not rows:
        return [f"{path}: no non-empty JSONL rows sampled"]

    for line_no, row in rows:
        row_errors = validate_phase1_target_row(row, allow_acp_legacy=acp_legacy_compat)
        for e in row_errors:
            errors.append(f"{path}:{line_no}: {e}")
    return errors


def _validate_split_manifest(path: Path) -> list[str]:
    errors: list[str] = []
    for strategy in ("random", "temporal"):
        for split in ("train", "dev", "test"):
            try:
                load_split_ids(path, strategy=strategy, split=split)
            except Exception as exc:
                errors.append(f"{path}: invalid split schema for {strategy}/{split}: {exc}")
    return errors


def _existing(paths: list[Path]) -> list[Path]:
    return [p for p in paths if p.exists()]


def _missing(paths: list[Path]) -> list[Path]:
    return [p for p in paths if not p.exists()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate frozen Phase1 contract and namespace")
    parser.add_argument(
        "--targets",
        action="append",
        default=None,
        help="target JSONL path (repeatable). default: discovered ACP/Mocha/sidecar targets",
    )
    parser.add_argument(
        "--split-manifests",
        action="append",
        default=None,
        help="split manifest JSON path (repeatable). default: discovered ACP/Mocha/sidecar manifests",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=32,
        help="rows sampled per target file (default: 32)",
    )
    parser.add_argument(
        "--acp-legacy-compat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="allow ACP legacy rows without `source` and `acp:` prefix (default: enabled)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    target_candidates = (
        [Path(p) for p in args.targets]
        if args.targets is not None
        else default_phase1_target_candidates(repo_root)
    )
    split_candidates = (
        [Path(p) for p in args.split_manifests]
        if args.split_manifests is not None
        else default_phase1_split_candidates(repo_root)
    )

    existing_targets = _existing(target_candidates)
    existing_splits = _existing(split_candidates)

    print(f"required_fields={PHASE1_REQUIRED_FIELDS}")
    print(f"record_id_prefixes={PHASE1_RECORD_ID_PREFIXES}")
    print(f"acp_legacy_required_fields={ACP_LEGACY_REQUIRED_FIELDS}")
    print(f"acp_legacy_compat={args.acp_legacy_compat}")

    for p in _missing(target_candidates):
        print(f"SKIP target missing: {p}")
    for p in _missing(split_candidates):
        print(f"SKIP split manifest missing: {p}")

    if not existing_targets:
        print("ERROR: no target files found to validate", file=sys.stderr)
        return 2

    errors: list[str] = []
    for p in existing_targets:
        print(f"CHECK target: {p}")
        errors.extend(
            _validate_target_file(
                p,
                sample_size=max(int(args.sample_size), 1),
                acp_legacy_compat=bool(args.acp_legacy_compat),
            )
        )

    for p in existing_splits:
        print(f"CHECK split manifest: {p}")
        errors.extend(_validate_split_manifest(p))

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print(f"contract_status=FAIL errors={len(errors)}", file=sys.stderr)
        return 1

    print(
        f"contract_status=PASS targets_checked={len(existing_targets)} "
        f"split_manifests_checked={len(existing_splits)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
