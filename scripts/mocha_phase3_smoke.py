#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast


def _run_cmd(cmd: list[str], env: dict[str, str], cwd: Path) -> dict[str, object]:
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=str(cwd), check=False)
    return {
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "seconds": round(time.time() - t0, 3),
    }


def _strip_articulatory_keys(src_targets: Path, dst_targets: Path) -> int:
    rows = 0
    dst_targets.parent.mkdir(parents=True, exist_ok=True)
    with src_targets.open("r", encoding="utf-8") as fin, dst_targets.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            row.pop("articulatory_vector", None)
            row.pop("articulatory_mask", None)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
    return rows


def _load_last_metrics_row(train_dir: Path) -> dict[str, object]:
    metrics_path = train_dir / "train_metrics.jsonl"
    if not metrics_path.exists():
        return {}
    last: dict[str, object] = {}
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            last = json.loads(line)
    return last


def _load_checkpoint_meta(ckpt_path: Path) -> dict[str, object]:
    if not ckpt_path.exists():
        return {}
    import torch

    obj = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(obj, dict):
        return {}
    keys = [
        "artic_loss_weight",
        "artic_constraint_mean",
        "articulatory_batches_seen",
        "articulatory_mask_sum",
        "articulatory_mask_nonzero_batches",
    ]
    return {k: obj.get(k) for k in keys}


def _run_train_case(
    case_name: str,
    targets: Path,
    splits: Path,
    out_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    artic_loss_weight: float,
) -> dict[str, object]:
    train_dir = out_dir / "train"
    ckpt = train_dir / "checkpoint_none.pt"

    env = dict(os.environ)
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"src:{py_path}" if py_path else "src"

    cmd = [
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
        str(int(epochs)),
        "--batch-size",
        str(int(batch_size)),
        "--batching",
        "flat",
        "--num-workers",
        "0",
        "--seed",
        str(int(seed)),
        "--artic-loss-weight",
        str(float(artic_loss_weight)),
        "--out",
        str(train_dir),
    ]

    cmd_row = _run_cmd(cmd, env=env, cwd=Path("."))
    last_metrics = _load_last_metrics_row(train_dir)
    ckpt_meta = _load_checkpoint_meta(ckpt)

    metrics_has_artic_term = "artic_constraint_mean" in last_metrics
    metrics_has_batch_flag = "batch_has_articulatory" in last_metrics

    returncode = cmd_row.get("returncode")
    ok = isinstance(returncode, int) and returncode == 0
    out: dict[str, object] = {
        "case": case_name,
        "targets": str(targets),
        "split_manifest": str(splits),
        "command": cmd_row,
        "checkpoint": str(ckpt),
        "last_metrics": last_metrics,
        "checkpoint_meta": ckpt_meta,
        "metrics_has_artic_term": bool(metrics_has_artic_term),
        "metrics_has_batch_flag": bool(metrics_has_batch_flag),
        "ok": bool(ok),
    }
    return out


def main() -> int:
    mocha_module = importlib.import_module("pgdn_torch.data.mocha_timit")
    build_mocha_sidecar = cast(Any, mocha_module).build_mocha_sidecar

    parser = argparse.ArgumentParser(
        description=(
            "Mocha Phase 3 smoke: build articulatory sidecar and run enabled/disabled "
            "train smokes with machine-readable evidence."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("data/external/mocha_timit"))
    parser.add_argument("--speakers", type=str, default="fsew0,msak0")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--enabled-artic-loss-weight", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=Path("runs/paper_gap_phase3"))
    args = parser.parse_args()

    speakers = [s.strip().lower() for s in str(args.speakers).split(",") if s.strip()]
    sidecar_dir = args.out / "sidecar"
    enabled_dir = args.out / "enabled"
    disabled_dir = args.out / "disabled"
    args.out.mkdir(parents=True, exist_ok=True)

    sidecar = build_mocha_sidecar(
        root=Path(args.root),
        speakers=speakers,
        out_dir=sidecar_dir,
        limit=int(args.limit),
    )
    targets = Path(str(sidecar["targets"]))
    splits = Path(str(sidecar["splits"]))
    disabled_targets = disabled_dir / "targets_no_articulatory.jsonl"
    stripped_rows = _strip_articulatory_keys(targets, disabled_targets)

    enabled_meta = _run_train_case(
        case_name="enabled",
        targets=targets,
        splits=splits,
        out_dir=enabled_dir,
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        artic_loss_weight=float(args.enabled_artic_loss_weight),
    )
    disabled_meta = _run_train_case(
        case_name="disabled",
        targets=disabled_targets,
        splits=splits,
        out_dir=disabled_dir,
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        artic_loss_weight=0.0,
    )

    enabled_meta_path = args.out / "train_smoke_enabled_meta.json"
    disabled_meta_path = args.out / "train_smoke_disabled_meta.json"
    enabled_meta_path.write_text(json.dumps(enabled_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    disabled_meta_path.write_text(json.dumps(disabled_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "out": str(args.out),
        "source_root": str(args.root),
        "speakers": speakers,
        "sidecar_rows": int(cast(int, sidecar["rows"])),
        "disabled_stripped_rows": int(stripped_rows),
        "enabled_meta": str(enabled_meta_path),
        "disabled_meta": str(disabled_meta_path),
        "ok": bool(enabled_meta["ok"] and disabled_meta["ok"]),
    }
    summary_path = args.out / "phase3_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"summary": str(summary_path), "ok": bool(summary["ok"])}, ensure_ascii=False))
    if not bool(summary["ok"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
