from __future__ import annotations

# pyright: reportMissingImports=false, reportArgumentType=false

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from pgdn.confidence import confidence_bucket, ranking_terms_for_record
from pgdn_torch.pgdnv0.splits import load_split_ids


def _require_nonempty_file(path: Path, hint: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or empty file: {path}\n\nHint: {hint}\n")


def _load_json(path: Path) -> dict[str, object]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"expected JSON object in {path}")
    return obj


def _read_jsonl_rows(
    path: Path,
    include_ids: set[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row_obj = json.loads(line)
            if not isinstance(row_obj, dict):
                raise SystemExit(f"invalid JSONL row in {path}: expected object")
            row = dict(row_obj)
            rid = row.get("record_id")
            rid_str = str(rid) if rid is not None else ""
            if include_ids is not None and rid_str not in include_ids:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_cmd(cmd: list[str], env: dict[str, str], cwd: Path) -> dict[str, object]:
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=str(cwd), check=False)
    return {
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "seconds": round(time.time() - t0, 3),
    }


def _load_infer_samples(samples_dir: Path) -> tuple[list[str], dict[str, list[list[float]]]]:
    sample_paths = sorted(samples_dir.glob("sample_*.pt"))
    if not sample_paths:
        raise SystemExit(f"no sample files found in {samples_dir}")

    record_ids0: list[str] | None = None
    by_record: dict[str, list[list[float]]] = {}
    for p in sample_paths:
        obj = torch.load(p, map_location="cpu")
        if not isinstance(obj, dict) or "record_id" not in obj or "vector" not in obj:
            raise SystemExit(f"unexpected sample format: {p}")
        record_ids = obj["record_id"]
        vec = obj["vector"]
        if not isinstance(record_ids, list) or not all(isinstance(x, str) for x in record_ids):
            raise SystemExit(f"unexpected sample format: {p} (record_id must be list[str])")
        if not isinstance(vec, torch.Tensor) or vec.ndim != 2:
            raise SystemExit(f"unexpected sample format: {p} (vector must be rank-2 tensor)")
        if len(record_ids) != int(vec.shape[0]):
            raise SystemExit(
                f"schema mismatch in {p}: len(record_id)={len(record_ids)} != vector.shape[0]={int(vec.shape[0])}"
            )

        if record_ids0 is None:
            record_ids0 = list(record_ids)
            by_record = {rid: [] for rid in record_ids0}
        elif record_ids != record_ids0:
            raise SystemExit(
                "schema mismatch across sample files: record_id sequence differs across samples"
            )

        vec_list = vec.detach().cpu().tolist()
        for rid, sample_vec in zip(record_ids, vec_list, strict=True):
            if rid not in by_record:
                raise SystemExit(f"internal error: missing record id while loading samples: {rid}")
            by_record[rid].append([float(x) for x in sample_vec])

    assert record_ids0 is not None
    return record_ids0, by_record


def _mean_vector(sample_vectors: list[list[float]]) -> list[float]:
    if not sample_vectors:
        return []
    return [
        float(sum(col) / float(max(len(col), 1)))
        for col in zip(*sample_vectors, strict=True)
    ]


def _mask_obj(row: dict[str, object], record_id: str) -> dict[str, int]:
    raw_mask = row.get("mask")
    if not isinstance(raw_mask, dict):
        raise SystemExit(f"row missing object mask for record_id={record_id}")
    return {
        "I": int(raw_mask.get("I", 0)),
        "M": int(raw_mask.get("M", 0)),
        "N": int(raw_mask.get("N", 0)),
        "C": int(raw_mask.get("C", 0)),
    }


def _float_metric(obj: dict[str, object], key: str) -> float | None:
    value = obj.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run phase2 self-distillation (v1 -> pseudo-labels -> v2)")
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        required=True,
        help="v1 checkpoint path (typically runs/multidataset_phase2/pretrain/checkpoint_none.pt)",
    )
    parser.add_argument("--targets", type=Path, default=Path("data/targets/acp_targets.jsonl"))
    parser.add_argument("--split-manifest", type=Path, default=Path("data/splits/manifest.json"))
    parser.add_argument("--split-strategy", type=str, default="random", choices=["random", "temporal"])
    parser.add_argument("--pseudo-split", type=str, default="train", choices=["train", "dev", "test"])
    parser.add_argument("--eval-split", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--eval-num-steps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/multidataset_phase2/self_distill"),
    )
    args = parser.parse_args()

    if int(args.samples) < 2:
        parser.error("--samples must be >= 2 for uncertainty-based confidence gating")
    if not (0.0 <= float(args.confidence_threshold) <= 1.0):
        parser.error("--confidence-threshold must be in [0, 1]")

    hint = "Use a valid v1 checkpoint, e.g., runs/multidataset_phase2/pretrain/checkpoint_none.pt"
    _require_nonempty_file(Path(args.base_checkpoint), hint=hint)
    _require_nonempty_file(Path(args.targets), hint="Provide a targets JSONL file")
    _require_nonempty_file(Path(args.split_manifest), hint="Provide a split manifest JSON")

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pseudo_jsonl = out_dir / "pseudo_labels.jsonl"
    pseudo_manifest_json = out_dir / "pseudo_label_manifest.json"
    merged_targets_jsonl = out_dir / "merged_train_targets.jsonl"
    comparison_json = out_dir / "comparison.json"

    v1_infer_dir = out_dir / "v1_infer"
    v1_eval_dir = out_dir / "v1_eval"
    v2_dir = out_dir / "v2"
    v2_eval_dir = out_dir / "v2_eval"

    env = dict(os.environ)
    src_path = str(repo_root / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}:{existing_pythonpath}" if existing_pythonpath else src_path

    command_rows: list[dict[str, object]] = []

    infer_cmd = [
        sys.executable,
        "-m",
        "pgdn_torch.infer.pgdnv0_infer",
        "--checkpoint",
        str(args.base_checkpoint),
        "--targets",
        str(args.targets),
        "--split-manifest",
        str(args.split_manifest),
        "--split-strategy",
        str(args.split_strategy),
        "--split",
        str(args.pseudo_split),
        "--samples",
        str(int(args.samples)),
        "--num-steps",
        str(int(args.num_steps)),
        "--batch-size",
        str(int(args.batch_size)),
        "--seed",
        str(int(args.seed)),
        "--out",
        str(v1_infer_dir),
    ]
    if args.limit is not None:
        infer_cmd.extend(["--limit", str(int(args.limit))])
    infer_row = _run_cmd(infer_cmd, env=env, cwd=repo_root)
    command_rows.append(infer_row)
    if int(infer_row["returncode"]) != 0:
        comparison_json.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "failed_step": "v1_infer",
                    "commands": command_rows,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return int(infer_row["returncode"])

    record_ids, sample_vectors = _load_infer_samples(v1_infer_dir / "samples")

    include_ids = load_split_ids(
        Path(args.split_manifest),
        strategy=str(args.split_strategy),
        split=str(args.pseudo_split),
    )
    real_rows = _read_jsonl_rows(Path(args.targets), include_ids=include_ids, limit=args.limit)
    real_rows_by_id = {
        str(row.get("record_id")): row
        for row in real_rows
        if isinstance(row.get("record_id"), str)
    }

    selected_rows: list[dict[str, object]] = []
    selected_records: list[dict[str, object]] = []
    excluded_records: list[dict[str, object]] = []

    for rid in record_ids:
        source_row = real_rows_by_id.get(rid)
        if source_row is None:
            raise SystemExit(f"missing source row for record_id from infer output: {rid}")
        vectors = sample_vectors.get(rid)
        if not vectors:
            raise SystemExit(f"missing sample vectors for record_id={rid}")

        mask = _mask_obj(source_row, record_id=rid)
        terms = ranking_terms_for_record(vectors, mask)
        uncertainty = float(terms["uncertainty_mean"])
        confidence = max(0.0, min(1.0, 1.0 - uncertainty))
        is_selected = confidence >= float(args.confidence_threshold)

        selection_row = {
            "record_id": rid,
            "confidence": float(confidence),
            "uncertainty_mean": float(uncertainty),
            "ranking_score": float(terms["ranking_score"]),
            "confidence_bucket": confidence_bucket(float(uncertainty)),
            "selected": bool(is_selected),
        }

        if is_selected:
            pseudo_row: dict[str, object] = {
                "record_id": f"pseudo_phase2:v1:{rid}",
                "graph_id": str(source_row.get("graph_id", "graph:pseudo_phase2:v1")),
                "source": "pseudo_phase2_v1",
                "generated_flag": True,
                "pseudo_from_record_id": rid,
                "target_vector": _mean_vector(vectors),
                "mask": mask,
                "confidence": float(confidence),
                "uncertainty_mean": float(uncertainty),
                "ranking_score": float(terms["ranking_score"]),
                "confidence_bucket": confidence_bucket(float(uncertainty)),
            }
            for key in ("character", "language", "period"):
                value = source_row.get(key)
                if isinstance(value, str):
                    pseudo_row[key] = value
            selected_rows.append(pseudo_row)
            selected_records.append(selection_row)
        else:
            excluded_records.append(selection_row)

    if any(float(r["confidence"]) < float(args.confidence_threshold) for r in selected_rows):
        raise SystemExit("internal error: pseudo label with confidence below threshold was selected")

    _write_jsonl(pseudo_jsonl, selected_rows)

    selected_confidences = [float(r["confidence"]) for r in selected_records]
    excluded_confidences = [float(r["confidence"]) for r in excluded_records]

    pseudo_manifest = {
        "metric_version": 1,
        "base_checkpoint": str(args.base_checkpoint),
        "targets": str(args.targets),
        "split_manifest": str(args.split_manifest),
        "split_strategy": str(args.split_strategy),
        "pseudo_split": str(args.pseudo_split),
        "limit": int(args.limit) if args.limit is not None else None,
        "samples": int(args.samples),
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "confidence_signal": "1.0 - uncertainty_mean",
        "confidence_field": "confidence",
        "uncertainty_field": "uncertainty_mean",
        "confidence_threshold": float(args.confidence_threshold),
        "selection": {
            "total_candidates": int(len(record_ids)),
            "selected_count": int(len(selected_rows)),
            "excluded_low_confidence_count": int(len(record_ids) - len(selected_rows)),
            "selection_rate": float(len(selected_rows) / max(len(record_ids), 1)),
            "selected_confidence_min": float(min(selected_confidences)) if selected_confidences else None,
            "selected_confidence_max": float(max(selected_confidences)) if selected_confidences else None,
            "excluded_confidence_max": float(max(excluded_confidences)) if excluded_confidences else None,
        },
        "artifacts": {
            "pseudo_labels_jsonl": str(pseudo_jsonl),
            "v1_infer_dir": str(v1_infer_dir),
        },
        "selection_preview": {
            "selected": selected_records[:20],
            "excluded_low_confidence": excluded_records[:20],
        },
    }
    pseudo_manifest_json.write_text(
        json.dumps(pseudo_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    merged_rows = [*real_rows, *selected_rows]
    _write_jsonl(merged_targets_jsonl, merged_rows)

    train_cmd = [
        sys.executable,
        "-m",
        "pgdn_torch.train.pgdnv0_train",
        "--targets",
        str(merged_targets_jsonl),
        "--epochs",
        str(int(args.epochs)),
        "--batch-size",
        str(int(args.batch_size)),
        "--batching",
        "graph",
        "--num-workers",
        "0",
        "--seed",
        str(int(args.seed)),
        "--out",
        str(v2_dir),
    ]
    train_row = _run_cmd(train_cmd, env=env, cwd=repo_root)
    command_rows.append(train_row)
    if int(train_row["returncode"]) != 0:
        comparison_json.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "failed_step": "v2_train",
                    "commands": command_rows,
                    "pseudo_manifest": str(pseudo_manifest_json),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return int(train_row["returncode"])

    v2_checkpoint = v2_dir / "checkpoint_none.pt"
    _require_nonempty_file(v2_checkpoint, hint="training should produce checkpoint_none.pt")

    eval_common = [
        "--targets",
        str(args.targets),
        "--split-manifest",
        str(args.split_manifest),
        "--split-strategy",
        str(args.split_strategy),
        "--split",
        str(args.eval_split),
        "--batch-size",
        str(int(args.batch_size)),
        "--batching",
        "graph",
        "--samples",
        str(int(args.eval_samples)),
        "--num-steps",
        str(int(args.eval_num_steps)),
        "--seed",
        str(int(args.seed)),
    ]
    if args.limit is not None:
        eval_common.extend(["--limit", str(int(args.limit))])

    eval_v1_cmd = [
        sys.executable,
        "scripts/eval_torch_pgdnv0.py",
        "--checkpoint",
        str(args.base_checkpoint),
        *eval_common,
        "--out",
        str(v1_eval_dir),
    ]
    eval_v1_row = _run_cmd(eval_v1_cmd, env=env, cwd=repo_root)
    command_rows.append(eval_v1_row)
    if int(eval_v1_row["returncode"]) != 0:
        comparison_json.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "failed_step": "v1_eval",
                    "commands": command_rows,
                    "pseudo_manifest": str(pseudo_manifest_json),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return int(eval_v1_row["returncode"])

    eval_v2_cmd = [
        sys.executable,
        "scripts/eval_torch_pgdnv0.py",
        "--checkpoint",
        str(v2_checkpoint),
        *eval_common,
        "--out",
        str(v2_eval_dir),
    ]
    eval_v2_row = _run_cmd(eval_v2_cmd, env=env, cwd=repo_root)
    command_rows.append(eval_v2_row)
    if int(eval_v2_row["returncode"]) != 0:
        comparison_json.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "failed_step": "v2_eval",
                    "commands": command_rows,
                    "pseudo_manifest": str(pseudo_manifest_json),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return int(eval_v2_row["returncode"])

    v1_metrics = _load_json(v1_eval_dir / "eval.json")
    v2_metrics = _load_json(v2_eval_dir / "eval.json")
    tracked_metrics = [
        "ranking_score_mean",
        "uncertainty_mean",
        "constraint_penalty_mean",
        "max_abs",
        "frac_abs_gt_1",
        "impossible_penalty_mean",
    ]
    delta: dict[str, float] = {}
    for key in tracked_metrics:
        v1_value = _float_metric(v1_metrics, key)
        v2_value = _float_metric(v2_metrics, key)
        if v1_value is None or v2_value is None:
            continue
        delta[key] = float(v2_value - v1_value)

    training_mode = (
        "merged_real_plus_pseudo"
        if selected_rows
        else "real_only_fallback_empty_pseudo_after_threshold"
    )

    report = {
        "metric_version": 1,
        "status": "ok",
        "base_checkpoint": str(args.base_checkpoint),
        "v2_checkpoint": str(v2_checkpoint),
        "pseudo_label_manifest": str(pseudo_manifest_json),
        "pseudo_labels": str(pseudo_jsonl),
        "merged_targets": str(merged_targets_jsonl),
        "config": {
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "samples": int(args.samples),
            "num_steps": int(args.num_steps),
            "eval_samples": int(args.eval_samples),
            "eval_num_steps": int(args.eval_num_steps),
            "confidence_threshold": float(args.confidence_threshold),
            "targets": str(args.targets),
            "split_manifest": str(args.split_manifest),
            "split_strategy": str(args.split_strategy),
            "pseudo_split": str(args.pseudo_split),
            "eval_split": str(args.eval_split),
            "limit": int(args.limit) if args.limit is not None else None,
        },
        "pseudo_selection": pseudo_manifest["selection"],
        "training_dataset_mode": training_mode,
        "training_rows": {
            "real_rows": int(len(real_rows)),
            "pseudo_rows": int(len(selected_rows)),
            "merged_rows": int(len(merged_rows)),
        },
        "v1": v1_metrics,
        "v2": v2_metrics,
        "delta_v2_minus_v1": delta,
        "commands": command_rows,
    }
    comparison_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"comparison": str(comparison_json), "v2_checkpoint": str(v2_checkpoint)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
