from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch


def _require_file(path: Path, hint: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or empty file: {path}\n\nHint: {hint}\n")


def _load_json_object(path: Path) -> dict[str, object]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"expected JSON object in {path}")
    return obj


def _extract_baseline_value(summary: dict[str, object], mode: str, metric: str) -> float:
    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, dict):
        raise SystemExit("baseline summary missing object key: aggregate")
    mode_obj = aggregate.get(mode)
    if not isinstance(mode_obj, dict):
        raise SystemExit(f"baseline summary missing mode under aggregate: {mode}")
    metric_obj = mode_obj.get(metric)
    if not isinstance(metric_obj, dict):
        raise SystemExit(f"baseline summary missing metric under aggregate.{mode}: {metric}")
    mean_obj = metric_obj.get("mean")
    if not isinstance(mean_obj, (int, float)):
        raise SystemExit(f"baseline summary missing numeric mean under aggregate.{mode}.{metric}")
    return float(mean_obj)


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


def _character_consistency_from_infer_dir(infer_dir: Path) -> dict[str, object]:
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
    distances: list[float] = []

    for ch in characters_total:
        vs = by_character_vectors.get(ch, [])
        k = len(vs)
        if k < 2:
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
        distances.append(1.0 - float(mean_sim))

    return {
        "metric_version": 1,
        "mean": float(sum(distances) / len(distances)) if distances else None,
        "median": float(statistics.median(distances)) if distances else None,
        "p90": _percentile_nearest_rank(distances, 0.9),
        "n_characters_total": int(len(characters_total)),
        "n_characters_used": int(len(distances)),
        "n_records_skipped_zero_norm": int(n_records_skipped_zero_norm),
        "n_samples": int(n_samples),
        "vector_dim": int(vector_dim or 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ACP non-regression gate on character_consistency_mean")
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("runs/paper_grade_closure/aggregate/summary.json"),
    )
    parser.add_argument("--baseline-mode", type=str, default="none")
    parser.add_argument("--metric", type=str, default="character_consistency_mean")
    parser.add_argument("--infer-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/multidataset_phase1/acp_non_regression.json"),
    )
    args = parser.parse_args()

    baseline_hint = "Use paper-grade aggregate summary at runs/paper_grade_closure/aggregate/summary.json"
    _require_file(args.baseline_summary, hint=baseline_hint)

    baseline_summary = _load_json_object(args.baseline_summary)
    baseline_value = _extract_baseline_value(baseline_summary, mode=str(args.baseline_mode), metric=str(args.metric))

    current_metric = _character_consistency_from_infer_dir(args.infer_dir)
    current_mean = current_metric.get("mean")
    if not isinstance(current_mean, (int, float)):
        raise SystemExit("current character consistency mean is unavailable; inference output is insufficient")
    current_value = float(current_mean)

    relative_degradation = (current_value - baseline_value) / max(abs(baseline_value), 1e-12)
    decision = "FAIL" if relative_degradation > float(args.threshold) else "PASS"

    result = {
        "gate": "acp_non_regression",
        "metric": str(args.metric),
        "lower_is_better": True,
        "baseline_summary": str(args.baseline_summary),
        "baseline_mode": str(args.baseline_mode),
        "baseline_value": float(baseline_value),
        "current_infer_dir": str(args.infer_dir),
        "current_value": float(current_value),
        "relative_degradation": float(relative_degradation),
        "threshold": float(args.threshold),
        "decision": decision,
        "gate_decision": "NO_GO" if decision == "FAIL" else "GO",
        "current_metric_detail": current_metric,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"acp_non_regression={decision}")
    print(f"baseline_value={baseline_value:.12f}")
    print(f"current_value={current_value:.12f}")
    print(f"relative_degradation={relative_degradation:.6f}")
    print(f"threshold={float(args.threshold):.6f}")
    print(f"summary_json={args.out}")

    return 1 if decision == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
