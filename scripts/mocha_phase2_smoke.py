#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pgdn_torch.data.mocha_timit import build_mocha_sidecar


def _run_cmd(cmd: list[str], env: dict[str, str], cwd: Path) -> dict[str, object]:
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=str(cwd), check=False)
    return {
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "seconds": round(time.time() - t0, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Mocha Phase 2 sidecar smoke: build sidecar targets/splits, run tiny "
            "torch train->infer->eval chain, and emit machine-readable artifacts."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("data/external/mocha_timit"))
    parser.add_argument("--speakers", type=str, default="fsew0,msak0")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs/mocha_phase2_smoke"))
    args = parser.parse_args()

    speakers = [s.strip().lower() for s in str(args.speakers).split(",") if s.strip()]
    sidecar_dir = args.out / "sidecar"
    train_dir = args.out / "train"
    infer_dir = args.out / "infer"
    eval_dir = args.out / "eval"
    args.out.mkdir(parents=True, exist_ok=True)

    sidecar = build_mocha_sidecar(
        root=Path(args.root),
        speakers=speakers,
        out_dir=sidecar_dir,
        limit=int(args.limit),
    )
    targets = Path(str(sidecar["targets"]))
    splits = Path(str(sidecar["splits"]))
    ckpt = train_dir / "checkpoint_none.pt"
    infer_meta = infer_dir / "infer_meta.json"
    eval_json = eval_dir / "eval.json"

    env = dict(os.environ)
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"src:{py_path}" if py_path else "src"

    commands: list[list[str]] = [
        [
            sys.executable,
            "-m",
            "pgdn_torch.train.pgdnv0_train",
            "--targets",
            str(targets),
            "--split-manifest",
            str(splits),
            "--split-strategy",
            "random",
            "--split",
            "train",
            "--epochs",
            str(int(args.epochs)),
            "--batch-size",
            str(int(args.batch_size)),
            "--batching",
            "flat",
            "--num-workers",
            "0",
            "--seed",
            str(int(args.seed)),
            "--out",
            str(train_dir),
        ],
        [
            sys.executable,
            "-m",
            "pgdn_torch.infer.pgdnv0_infer",
            "--checkpoint",
            str(ckpt),
            "--targets",
            str(targets),
            "--split-manifest",
            str(splits),
            "--split-strategy",
            "random",
            "--split",
            "dev",
            "--batch-size",
            str(int(args.batch_size)),
            "--batching",
            "flat",
            "--samples",
            str(int(args.samples)),
            "--num-steps",
            str(int(args.num_steps)),
            "--seed",
            str(int(args.seed)),
            "--out",
            str(infer_dir),
        ],
        [
            sys.executable,
            "scripts/eval_torch_pgdnv0.py",
            "--checkpoint",
            str(ckpt),
            "--targets",
            str(targets),
            "--split-manifest",
            str(splits),
            "--split-strategy",
            "random",
            "--split",
            "test",
            "--batch-size",
            str(int(args.batch_size)),
            "--batching",
            "flat",
            "--samples",
            str(int(args.samples)),
            "--num-steps",
            str(int(args.num_steps)),
            "--seed",
            str(int(args.seed)),
            "--out",
            str(eval_dir),
        ],
    ]

    command_rows: list[dict[str, object]] = []
    for cmd in commands:
        row = _run_cmd(cmd, env=env, cwd=Path("."))
        command_rows.append(row)
        if int(row["returncode"]) != 0:
            break

    summary = {
        "out": str(args.out),
        "source_root": str(args.root),
        "speakers": speakers,
        "sidecar_rows": int(sidecar["rows"]),
        "targets": str(targets),
        "splits": str(splits),
        "checkpoint": str(ckpt),
        "infer_meta": str(infer_meta),
        "eval_json": str(eval_json),
        "commands": command_rows,
        "ok": bool(command_rows and all(int(r["returncode"]) == 0 for r in command_rows)),
    }
    summary_path = args.out / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    commands_csv = args.out / "commands.csv"
    with commands_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["command", "returncode", "seconds"])
        writer.writeheader()
        writer.writerows(command_rows)

    print(json.dumps({"summary": str(summary_path), "ok": bool(summary["ok"])}, ensure_ascii=False))
    if not bool(summary["ok"]):
        return 1
    if not infer_meta.exists() or not eval_json.exists():
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
