from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "pgdn_torch.train.pgdnv0_train",
        "--targets",
        str(args.targets),
        "--split-manifest",
        str(args.split_manifest),
        "--split-strategy",
        str(args.split_strategy),
        "--split",
        str(args.split),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--out",
        str(args.out),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.batching:
        cmd.extend(["--batching", str(args.batching)])
    if args.extra_args:
        cmd.extend(args.extra_args)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Phase2 synthetic pretrain checkpoint")
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("data/targets/synth_phase2_targets.jsonl"),
        help="Synthetic phase2 targets JSONL",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/splits/synth_phase2_splits.json"),
        help="Synthetic phase2 split manifest",
    )
    parser.add_argument("--split-strategy", type=str, default="random")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batching", type=str, default="graph")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/multidataset_phase2/pretrain"),
        help="Output run directory for phase2 pretrain",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to pgdn_torch.train.pgdnv0_train",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cmd = _build_command(args)
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    metadata = {
        "script": "scripts/train_phase2_pretrain.py",
        "seed": int(args.seed),
        "command": cmd,
        "targets": str(args.targets),
        "split_manifest": str(args.split_manifest),
        "out": str(args.out),
        "exit_code": int(result.returncode),
        "seconds": time.time() - t0,
    }
    meta_path = args.out / "pretrain_run_meta.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
