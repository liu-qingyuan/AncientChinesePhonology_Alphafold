from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


def _is_ignored_dir(name: str) -> bool:
    return name in {".git", "__pycache__"}


def _stable_hash32(s: str) -> int:
    # Simple FNV-1a 32-bit hash (deterministic across processes).
    h = 2166136261
    for ch in s.encode("utf-8", errors="ignore"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h


class SimpleFileDataset(Dataset[dict[str, object]]):
    """Toy dataset backed by a directory tree.

    Purpose:
    - Demonstrate reading from `data/external/links/<dataset>`
    - Provide a fast, GPU-capable training skeleton without depending on
      domain-specific feature engineering yet

    Each sample = one file under `root`, converted into a small numeric feature
    vector. Labels are deterministic hashes (classification), solely for smoke
    testing the training loop.
    """

    def __init__(self, root: Path, max_files: int = 4096) -> None:
        self.root = root
        self.max_files = max_files

        files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not _is_ignored_dir(d)]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.is_file():
                    files.append(p)
                    if len(files) >= max_files:
                        break
            if len(files) >= max_files:
                break

        self.files = sorted(files)
        if not self.files:
            raise RuntimeError(f"no files found under dataset root: {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, object]:
        p = self.files[idx]
        try:
            st = p.stat()
            size = float(st.st_size)
        except OSError:
            size = 0.0

        rel = str(p.relative_to(self.root))
        ext = p.suffix.lower().lstrip(".")
        ext_h = float(_stable_hash32(ext) % 997) / 997.0
        name_h = float(_stable_hash32(rel) % 1009) / 1009.0

        # Normalize size into a bounded range.
        size_norm = min(size / 1_000_000.0, 10.0) / 10.0

        x = torch.tensor([size_norm, ext_h, name_h], dtype=torch.float32)
        y = torch.tensor(_stable_hash32(rel) % 16, dtype=torch.long)
        return {"path": rel, "x": x, "y": y}
