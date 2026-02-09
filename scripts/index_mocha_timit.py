#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pgdn_torch.data.mocha_timit import index_mocha_timit


def main() -> int:
    parser = argparse.ArgumentParser(description="Index Mocha-TIMIT extracted files")
    parser.add_argument("--root", type=Path, default=Path("data/external/mocha_timit"))
    parser.add_argument("--speakers", type=str, default="fsew0,msak0")
    args = parser.parse_args()

    speakers = [s.strip().lower() for s in args.speakers.split(",") if s.strip()]
    rows = index_mocha_timit(args.root, speakers)

    summary = {
        "root": str(args.root),
        "speakers": speakers,
        "utterances": len(rows),
        "with_wav": sum(1 for r in rows if r.wav is not None),
        "with_ema": sum(1 for r in rows if r.ema is not None),
        "with_lab": sum(1 for r in rows if r.lab is not None),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
