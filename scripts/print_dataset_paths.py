#!/usr/bin/env python3
"""Print the unified dataset hub mapping.

This script is a convenience wrapper around:
  - `python3 scripts/organize_datasets.py`

It prints a compact table of dataset key -> link -> target -> status.

Rationale:
  - Treat `data/external/links/` as the ONLY stable entrypoint for code.
  - Keep submodules where they are; keep downloaded datasets under `data/external/`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_link_target(link: Path) -> str | None:
    if not link.exists() and not link.is_symlink():
        return None
    if not link.is_symlink():
        return f"<not-a-symlink:{link}>"
    # readlink returns the stored path (usually relative)
    try:
        raw = os.readlink(link)
    except OSError:
        return f"<unreadable-symlink:{link}>"
    try:
        resolved = str(link.resolve())
    except OSError:
        resolved = "<unresolvable>"
    return f"{raw} => {resolved}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/external/dataset_registry.json"),
        help="Path to dataset_registry.json (default: data/external/dataset_registry.json)",
    )
    args = parser.parse_args()

    reg_path: Path = args.registry
    if not reg_path.exists():
        raise SystemExit(
            f"missing registry: {reg_path} (run: python3 scripts/organize_datasets.py)"
        )

    reg = _read_json(reg_path)
    datasets = reg.get("datasets", [])
    if not isinstance(datasets, list):
        raise SystemExit(f"unexpected registry format: datasets is {type(datasets)}")

    print("key\tstatus\tlink\ttarget")
    for row in datasets:
        if not isinstance(row, dict):
            continue
        key = str(row.get("key"))
        status = str(row.get("status"))
        link_path = row.get("link_path")
        link = Path(link_path) if isinstance(link_path, str) else None
        target = _resolve_link_target(link) if link else None
        print(f"{key}\t{status}\t{link_path or '-'}\t{target or '-'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
