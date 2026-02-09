import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _require_file(path: Path, hint: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or empty file: {path}\n\nHint: {hint}\n")


def _load_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark group coupling modes (none/shared_noise/shared_denoise) with infer + eval",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="random",
        choices=["random", "temporal"],
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["train", "dev", "test"],
    )
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batching", type=str, default="graph", choices=["flat", "graph"])
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.5,
        help="CFG scale for shared_denoise (must be != 1.0 to be effective).",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", action="store_true", help="Also write summary.csv")
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()

    ckpt_hint = (
        "Run the plan smoke training, for example:\n"
        "  PYTHONPATH=src python3 -m pgdn_torch.train.pgdnv0_train \\\n+"
        "    --targets data/targets/acp_targets.jsonl \\\n+"
        "    --split-manifest data/splits/manifest.json \\\n+"
        "    --split-strategy random \\\n+"
        "    --split train \\\n+"
        "    --limit 2048 --epochs 1 --batch-size 32 --num-workers 0 --seed 42 \\\n+"
        "    --out runs/pgdnv0_paper_align_smoke\n"
    )
    _require_file(Path(args.checkpoint), hint=ckpt_hint)

    acp_hint = (
        "Build ACP artifacts, for example:\n"
        "  PYTHONPATH=src python3 -m pgdn.data.build_acp \\\n+"
        "    --csv Ancient-Chinese-Phonology/dataset/ancient_chinese_phonology.csv \\\n+"
        "    --out data --seed 42\n"
    )
    _require_file(Path(args.targets), hint=acp_hint)
    _require_file(Path(args.split_manifest), hint=acp_hint)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    modes = ["none", "shared_noise", "shared_denoise"]
    by_mode: dict[str, dict[str, object]] = {}

    for mode in modes:
        infer_dir = out_root / mode / "infer"
        eval_dir = out_root / mode / "eval"
        infer_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        cfg_scale = float(args.cfg_scale) if mode == "shared_denoise" else 1.0

        infer_cmd = [
            args.python,
            "-m",
            "pgdn_torch.infer.pgdnv0_infer",
            "--checkpoint",
            str(args.checkpoint),
            "--targets",
            str(args.targets),
            "--split-manifest",
            str(args.split_manifest),
            "--split-strategy",
            str(args.split_strategy),
            "--split",
            str(args.split),
            "--limit",
            str(args.limit),
            "--batch-size",
            str(args.batch_size),
            "--batching",
            str(args.batching),
            "--samples",
            str(args.samples),
            "--num-steps",
            str(args.num_steps),
            "--seed",
            str(args.seed),
            "--cfg-scale",
            str(cfg_scale),
            "--group-coupling",
            str(mode),
            "--group-key",
            "character",
            "--out",
            str(infer_dir),
        ]
        _run(["env", "PYTHONPATH=src", *infer_cmd])

        eval_cmd = [
            args.python,
            "scripts/eval_torch_pgdnv0.py",
            "--checkpoint",
            str(args.checkpoint),
            "--targets",
            str(args.targets),
            "--split-manifest",
            str(args.split_manifest),
            "--split-strategy",
            str(args.split_strategy),
            "--split",
            str(args.split),
            "--limit",
            str(args.limit),
            "--samples",
            str(args.samples),
            "--num-steps",
            str(args.num_steps),
            "--batch-size",
            str(args.batch_size),
            "--batching",
            str(args.batching),
            "--seed",
            str(args.seed),
            "--cfg-scale",
            str(cfg_scale),
            "--out",
            str(eval_dir),
        ]
        _run(["env", "PYTHONPATH=src", *eval_cmd])

        infer_meta_path = infer_dir / "infer_meta.json"
        eval_path = eval_dir / "eval.json"
        _require_file(infer_meta_path, hint="Infer did not produce infer_meta.json; see command output above.")
        _require_file(eval_path, hint="Eval did not produce eval.json; see command output above.")

        infer_meta = _load_json(infer_meta_path)
        eval_obj = _load_json(eval_path)

        by_mode[mode] = {
            "mode": mode,
            "infer_dir": str(infer_dir),
            "eval_dir": str(eval_dir),
            "infer_meta": infer_meta,
            "eval": eval_obj,
        }

    summary = {
        "checkpoint": str(args.checkpoint),
        "targets": str(args.targets),
        "split_manifest": str(args.split_manifest),
        "split_strategy": str(args.split_strategy),
        "split": str(args.split),
        "limit": int(args.limit),
        "batching": str(args.batching),
        "batch_size": int(args.batch_size),
        "samples": int(args.samples),
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "cfg_scale": float(args.cfg_scale),
        "modes": modes,
        "by_mode": by_mode,
    }

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(summary_path)}, ensure_ascii=False))

    if args.csv:
        csv_path = out_root / "summary.csv"
        fieldnames = [
            "mode",
            "infer_dir",
            "eval_dir",
            "max_abs",
            "frac_abs_gt_1",
            "constraint_penalty_mean",
            "ranking_score_mean",
            "coupling_shared_noise_enabled",
            "coupling_shared_noise_mismatch_count",
            "coupling_shared_denoise_enabled",
            "coupling_shared_denoise_effective",
            "coupling_shared_denoise_mismatch_count",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for mode in modes:
                m = by_mode[mode]
                infer_meta = m.get("infer_meta")
                eval_obj = m.get("eval")
                coupling = infer_meta.get("coupling", {}) if isinstance(infer_meta, dict) else {}
                sn = coupling.get("shared_noise", {}) if isinstance(coupling, dict) else {}
                sd = coupling.get("shared_denoise", {}) if isinstance(coupling, dict) else {}
                row = {
                    "mode": mode,
                    "infer_dir": m.get("infer_dir"),
                    "eval_dir": m.get("eval_dir"),
                    "max_abs": eval_obj.get("max_abs") if isinstance(eval_obj, dict) else None,
                    "frac_abs_gt_1": eval_obj.get("frac_abs_gt_1") if isinstance(eval_obj, dict) else None,
                    "constraint_penalty_mean": eval_obj.get("constraint_penalty_mean")
                    if isinstance(eval_obj, dict)
                    else None,
                    "ranking_score_mean": eval_obj.get("ranking_score_mean") if isinstance(eval_obj, dict) else None,
                    "coupling_shared_noise_enabled": sn.get("enabled") if isinstance(sn, dict) else None,
                    "coupling_shared_noise_mismatch_count": sn.get("mismatch_count") if isinstance(sn, dict) else None,
                    "coupling_shared_denoise_enabled": sd.get("enabled") if isinstance(sd, dict) else None,
                    "coupling_shared_denoise_effective": sd.get("effective") if isinstance(sd, dict) else None,
                    "coupling_shared_denoise_mismatch_count": sd.get("mismatch_count") if isinstance(sd, dict) else None,
                }
                w.writerow(row)
        print(json.dumps({"wrote": str(csv_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
