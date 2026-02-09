from __future__ import annotations

import json
from pathlib import Path


def load_split_ids(manifest_path: Path, strategy: str, split: str) -> set[str]:
    """Load record IDs for a given split from ACP build manifest.

    Expected manifest schema (from `src/pgdn/data/build_acp.py`):
      {
        "random": {"train": [...], "dev": [...], "test": [...]},
        "temporal": {"train": [...], "dev": [...], "test": [...]},
        ...
      }
    """

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict):
        raise ValueError(f"split manifest must be a JSON object: {manifest_path}")

    strategy_obj = manifest.get(strategy)
    if not isinstance(strategy_obj, dict):
        raise KeyError(
            f"split manifest missing strategy={strategy!r} dict at: {manifest_path}"
        )

    ids = strategy_obj.get(split)
    if not isinstance(ids, list):
        raise KeyError(
            f"split manifest missing split={split!r} list under strategy={strategy!r}: {manifest_path}"
        )

    out: set[str] = set()
    for x in ids:
        if x is None:
            continue
        out.add(str(x))
    return out
