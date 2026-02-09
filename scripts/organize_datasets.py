#!/usr/bin/env python3
"""Create a unified external dataset index and symlink hub.

This keeps large third-party datasets in their existing locations, while
exposing a clean, stable entrypoint under data/external/links.
"""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


IGNORE_DIRS = {".git", "__pycache__"}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    source_candidates: tuple[str, ...]
    notes: str


DATASETS = (
    DatasetSpec(
        key="phoible",
        label="PHOIBLE",
        source_candidates=("phoible",),
        notes="Official PHOIBLE repo clone.",
    ),
    DatasetSpec(
        key="clts",
        label="CLTS",
        source_candidates=("clts", "CLTS"),
        notes="Not found in current workspace; check download location.",
    ),
    DatasetSpec(
        key="wikihan",
        label="WikiHan",
        source_candidates=("wikihan",),
        notes="Comparative Sinitic dataset clone.",
    ),
    DatasetSpec(
        key="wikipron",
        label="WikiPron",
        source_candidates=("wikipron",),
        notes="WikiPron repo clone with scraped resources.",
    ),
    DatasetSpec(
        key="mocha_timit",
        label="Mocha-TIMIT",
        source_candidates=("data/external/mocha_timit", "mocha_timit"),
        notes="Planned acquisition target path.",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def walk_stats(path: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for root, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for filename in filenames:
            file_path = Path(root) / filename
            try:
                stat = file_path.stat()
            except OSError:
                continue
            file_count += 1
            total_bytes += stat.st_size
    return file_count, total_bytes


def resolve_existing(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for rel in candidates:
        candidate = (root / rel).resolve()
        if candidate.exists():
            return candidate
    return None


def ensure_symlink(link_path: Path, target_path: Path, dry_run: bool) -> None:
    rel_target = Path(os.path.relpath(target_path, start=link_path.parent))
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() and Path(os.readlink(link_path)) == rel_target:
            return
        if dry_run:
            return
        if link_path.is_dir() and not link_path.is_symlink():
            raise RuntimeError(f"Refusing to replace real directory: {link_path}")
        link_path.unlink()
    if dry_run:
        return
    link_path.symlink_to(rel_target)


def remove_link_if_exists(link_path: Path, dry_run: bool) -> None:
    if not link_path.exists() and not link_path.is_symlink():
        return
    if dry_run:
        return
    if link_path.is_dir() and not link_path.is_symlink():
        raise RuntimeError(f"Refusing to remove real directory: {link_path}")
    link_path.unlink()


def _is_clts_root(path: Path) -> bool:
    # Heuristic signature check to reduce false positives.
    # CLTS is typically a repo/checkout that contains CLDF metadata plus core TSVs.
    return (
        (path / "cldf-metadata.json").exists()
        and (path / "data" / "sounds.tsv").exists()
        and (path / "data" / "graphemes.tsv").exists()
    )


def resolve_existing_dataset(
    root: Path, key: str, candidates: tuple[str, ...]
) -> Path | None:
    # Default behavior: first existing candidate path wins.
    # Dataset-specific tweaks can be added here.
    for rel in candidates:
        candidate = (root / rel).resolve()
        if not candidate.exists():
            continue
        if key == "clts" and not _is_clts_root(candidate):
            continue
        return candidate
    return None


def main() -> int:
    class Args(Namespace):
        root: str = "."
        external_dir: str = "data/external"
        dry_run: bool = False
        clts_path: list[str] = []

    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--root",
        default=".",
        help="Workspace root path (default: current directory).",
    )
    _ = parser.add_argument(
        "--external-dir",
        default="data/external",
        help="External dataset directory (default: data/external).",
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute outputs without writing files or links.",
    )
    _ = parser.add_argument(
        "--clts-path",
        action="append",
        default=[],
        help="Additional CLTS candidate path(s). Can be repeated.",
    )
    args = parser.parse_args(namespace=Args())

    root = Path(args.root).resolve()
    external_dir = (root / args.external_dir).resolve()
    links_dir = external_dir / "links"
    registry_path = external_dir / "dataset_registry.json"

    if not args.dry_run:
        links_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for spec in DATASETS:
        candidates = spec.source_candidates
        if spec.key == "clts":
            extra: list[str] = []
            env_path = os.environ.get("CLTS_PATH")
            if env_path:
                extra.append(env_path)
            extra.extend(args.clts_path)
            # Prepend user-provided candidates.
            candidates = tuple(extra) + candidates

        src = resolve_existing_dataset(root, spec.key, candidates)
        link_path = links_dir / spec.key

        if src is None:
            remove_link_if_exists(link_path, args.dry_run)
            rows.append(
                {
                    "key": spec.key,
                    "label": spec.label,
                    "status": "missing",
                    "source_candidates": list(candidates),
                    "source_path": None,
                    "link_path": None,
                    "file_count": 0,
                    "total_bytes": 0,
                    "notes": spec.notes,
                }
            )
            continue

        try:
            source_path = str(src.relative_to(root))
        except ValueError:
            # User-provided dataset location can be outside the workspace root.
            source_path = str(src)

        ensure_symlink(link_path, src, args.dry_run)
        file_count, total_bytes = walk_stats(src)
        rows.append(
            {
                "key": spec.key,
                "label": spec.label,
                "status": "present",
                "source_candidates": list(candidates),
                "source_path": source_path,
                "link_path": str(link_path.relative_to(root)),
                "file_count": file_count,
                "total_bytes": total_bytes,
                "notes": spec.notes,
            }
        )

    result = {
        "generated_at_utc": utc_now(),
        "workspace_root": str(root),
        "external_dir": str(external_dir.relative_to(root)),
        "datasets": rows,
    }

    if not args.dry_run:
        external_dir.mkdir(parents=True, exist_ok=True)
        _ = registry_path.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
