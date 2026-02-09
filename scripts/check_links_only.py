#!/usr/bin/env python3
"""Guardrail: enforce using the dataset hub links in repo-owned code.

This is intentionally lightweight and conservative:

- It checks only code owned by this repo (under `src/pgdn_torch/`).
- It does NOT scan submodules.
- It does NOT forbid acquisition/index scripts from referencing `data/external/*`.

Fail condition:
- A Python file under `src/pgdn_torch/` contains an obvious hardcoded raw dataset
  root like `"phoible"`, `"wikihan"`, `"wikipron"`, etc. as a top-level path.

If you need a dataset, access it via `data/external/links/<key>`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = [ROOT / "src" / "pgdn_torch"]


FORBIDDEN_PATTERNS = [
    # hardcoded repo-root dataset dirs
    re.compile(r"(?m)(^|[^\w])phoible/"),
    re.compile(r"(?m)(^|[^\w])wikihan/"),
    re.compile(r"(?m)(^|[^\w])wikipron/"),
    re.compile(r"(?m)(^|[^\w])mocha_timit/"),
    re.compile(r"(?m)(^|[^\w])clts/"),
    re.compile(r"(?m)(^|[^\w])CLTS/"),
    # raw external dataset roots (should still go through links)
    re.compile(r"data/external/mocha_timit"),
    re.compile(r"data/external/clts"),
    re.compile(r"data/external/CLTS"),
]


def main() -> int:
    violations: list[str] = []
    for base in SCAN_DIRS:
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for pat in FORBIDDEN_PATTERNS:
                if pat.search(text):
                    violations.append(f"{path.relative_to(ROOT)}: matches {pat.pattern}")
                    break

    if violations:
        print("Found hardcoded dataset paths. Use data/external/links/<key> instead.")
        for v in violations:
            print(f"- {v}")
        return 2

    print("ok: no hardcoded raw dataset paths found in src/pgdn_torch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
