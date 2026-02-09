from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HubDatasetRef:
    key: str
    root: Path


def resolve_hub_dataset(
    key: str, hub_dir: Path = Path("data/external/links")
) -> HubDatasetRef:
    root = (hub_dir / key).resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"dataset link not found: {hub_dir / key} (run: python3 scripts/organize_datasets.py)"
        )
    return HubDatasetRef(key=key, root=root)
