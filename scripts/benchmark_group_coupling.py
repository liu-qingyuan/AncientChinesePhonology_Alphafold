import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import torch


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


def _percentile_nearest_rank(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    if p <= 0:
        return float(min(xs))
    if p >= 1:
        return float(max(xs))
    ys = sorted(float(x) for x in xs)
    idx = int(math.ceil(float(p) * len(ys))) - 1
    idx = max(0, min(idx, len(ys) - 1))
    return float(ys[idx])


def _character_consistency_from_infer_dir(infer_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    samples_dir = infer_dir / "samples"
    sample_paths = sorted(samples_dir.glob("sample_*.pt"))
    if not sample_paths:
        raise SystemExit(
            "no infer sample files found for character_consistency\n\n"
            f"Expected at least one file like: {samples_dir}/sample_000.pt\n"
            "Hint: ensure inference ran with --samples >= 1 and wrote samples/ artifacts.\n"
        )

    record_ids0: list[str] | None = None
    sum_vectors: torch.Tensor | None = None
    vector_dim: int | None = None

    for si, p in enumerate(sample_paths):
        obj = torch.load(p, map_location="cpu")
        if not isinstance(obj, dict) or "record_id" not in obj or "vector" not in obj:
            raise SystemExit(f"unexpected sample format: {p} (expected dict with keys 'record_id' and 'vector')")
        record_ids = obj["record_id"]
        vec = obj["vector"]
        if not isinstance(record_ids, list) or not all(isinstance(x, str) for x in record_ids):
            raise SystemExit(f"unexpected sample format: {p} (record_id must be list[str])")
        if not isinstance(vec, torch.Tensor) or vec.ndim != 2:
            raise SystemExit(f"unexpected sample format: {p} (vector must be a rank-2 torch.Tensor)")
        if len(record_ids) != int(vec.shape[0]):
            raise SystemExit(
                f"schema mismatch in {p}: len(record_id)={len(record_ids)} != vector.shape[0]={int(vec.shape[0])}"
            )
        if vector_dim is None:
            vector_dim = int(vec.shape[1])
        elif int(vec.shape[1]) != int(vector_dim):
            raise SystemExit(
                f"schema mismatch in {p}: vector.shape[1]={int(vec.shape[1])} != expected {int(vector_dim)}"
            )

        vec_cpu = vec.detach().double().cpu().contiguous()
        if si == 0:
            record_ids0 = list(record_ids)
            sum_vectors = vec_cpu
        else:
            assert record_ids0 is not None
            assert sum_vectors is not None
            if record_ids != record_ids0:
                raise SystemExit(
                    "schema mismatch across sample files: record_id sequence differs across samples\n\n"
                    f"First sample: {sample_paths[0].name}\n"
                    f"Mismatched sample: {p.name}\n"
                    "Hint: rerun the benchmark to regenerate infer artifacts deterministically.\n"
                )
            sum_vectors += vec_cpu

    assert record_ids0 is not None
    assert sum_vectors is not None
    n_samples = len(sample_paths)
    mean_vectors = sum_vectors / float(max(n_samples, 1))

    # Group by character.
    by_character_vectors: dict[str, list[torch.Tensor]] = {}
    n_records_skipped_zero_norm = 0
    for rid, v in zip(record_ids0, mean_vectors, strict=True):
        character = rid.split(":", 1)[0] if ":" in rid else rid
        norm = float(torch.linalg.norm(v).item())
        if norm <= 0.0:
            n_records_skipped_zero_norm += 1
            continue
        by_character_vectors.setdefault(character, []).append(v)

    characters_total = sorted({rid.split(":", 1)[0] if ":" in rid else rid for rid in record_ids0})
    per_char: dict[str, dict[str, object]] = {}
    distances: list[float] = []
    n_records_used = 0
    n_characters_used = 0
    n_characters_excluded_small = 0

    for ch in characters_total:
        vs = by_character_vectors.get(ch, [])
        k = len(vs)
        if k < 2:
            n_characters_excluded_small += 1
            continue
        mat = torch.stack(vs, dim=0).double()
        norms = torch.linalg.norm(mat, dim=1)
        if bool((norms <= 0).any().item()):
            raise SystemExit(f"internal error: encountered zero-norm vector after filtering for character={ch!r}")
        u = mat / norms.unsqueeze(1)
        s = torch.sum(u, dim=0)
        sum_pair_dot = float(((s @ s) - float(k)) / 2.0)
        denom = float(k * (k - 1) / 2)
        mean_sim = sum_pair_dot / max(denom, 1.0)
        mean_sim = max(-1.0, min(1.0, float(mean_sim)))
        mean_dist = 1.0 - float(mean_sim)

        per_char[ch] = {
            "n_records": int(k),
            "mean_distance": float(mean_dist),
        }
        distances.append(float(mean_dist))
        n_records_used += int(k)
        n_characters_used += 1

    mean_val = float(sum(distances) / len(distances)) if distances else None
    median_val = float(statistics.median(distances)) if distances else None
    p90_val = _percentile_nearest_rank(distances, 0.9)

    overall = {
        "metric_version": 1,
        "mean": mean_val,
        "median": median_val,
        "p90": p90_val,
        "n_characters_total": int(len(characters_total)),
        "n_characters_used": int(n_characters_used),
        "n_characters_excluded_small": int(n_characters_excluded_small),
        "n_records_used": int(n_records_used),
        "n_records_skipped_zero_norm": int(n_records_skipped_zero_norm),
        "n_samples": int(n_samples),
        "vector_dim": int(vector_dim or 0),
    }

    detail = {
        "metric_version": 1,
        "by_character": {k: per_char[k] for k in sorted(per_char)},
        "n_characters_total": int(len(characters_total)),
        "n_characters_used": int(n_characters_used),
        "n_characters_excluded_small": int(n_characters_excluded_small),
        "n_records_used": int(n_records_used),
        "n_records_skipped_zero_norm": int(n_records_skipped_zero_norm),
        "n_samples": int(n_samples),
        "vector_dim": int(vector_dim or 0),
        "sample_files": [p.name for p in sample_paths],
    }
    return overall, detail


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
    parser.add_argument(
        "--skip-character-consistency",
        action="store_true",
        help="Skip computing character_consistency from infer samples (Task 11).",
    )
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

        cc_overall: dict[str, object] | None = None
        if not bool(args.skip_character_consistency):
            cc_overall, cc_detail = _character_consistency_from_infer_dir(infer_dir)
            cc_path = infer_dir / "metrics" / "character_consistency" / "by_character.json"
            cc_path.parent.mkdir(parents=True, exist_ok=True)
            cc_path.write_text(
                json.dumps(cc_detail, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"wrote": str(cc_path)}, ensure_ascii=False))

        by_mode[mode] = {
            "mode": mode,
            "infer_dir": str(infer_dir),
            "eval_dir": str(eval_dir),
            "infer_meta": infer_meta,
            "eval": eval_obj,
            "character_consistency": cc_overall,
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
            "character_consistency_mean",
            "character_consistency_median",
            "character_consistency_p90",
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
                cc_obj = m.get("character_consistency")
                cc = cc_obj if isinstance(cc_obj, dict) else None
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
                    "character_consistency_mean": cc.get("mean") if isinstance(cc, dict) else None,
                    "character_consistency_median": cc.get("median") if isinstance(cc, dict) else None,
                    "character_consistency_p90": cc.get("p90") if isinstance(cc, dict) else None,
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
