from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from statistics import mean, pstdev
from typing import cast


VECTOR_DIM = 32


def _as_bool_third_class(value: object) -> bool | None:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        if int(value) == 3:
            return True
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"3", "iii", "third", "third_class", "grade3", "class3"}:
            return True
        if text in {"false", "0", "no", "none"}:
            return False
    return None


def _is_third_class_row(row: dict[str, object]) -> bool:
    candidate_keys = (
        "third_class",
        "is_third_class",
        "grade",
        "division",
        "deng",
        "rhyme_class",
        "yundeng",
    )
    for key in candidate_keys:
        if key in row:
            flag = _as_bool_third_class(row.get(key))
            if flag is not None:
                return flag

    for parent_key in ("meta", "metadata", "attrs", "features"):
        parent = row.get(parent_key)
        if not isinstance(parent, dict):
            continue
        parent_obj = cast(dict[str, object], parent)
        for key in candidate_keys:
            if key in parent_obj:
                flag = _as_bool_third_class(parent_obj.get(key))
                if flag is not None:
                    return flag

    return False


def _load_targets(path: Path, limit: int | None) -> list[dict[str, object]]:
    if not path.is_file():
        raise SystemExit(f"missing target file: {path}")

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            vec = obj.get("target_vector")
            if not isinstance(vec, list) or len(vec) != VECTOR_DIM:
                continue
            rows.append(cast(dict[str, object], obj))
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def _synthetic_rows(n: int, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(max(int(n), 1)):
        rng = random.Random(int(seed) + idx)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(VECTOR_DIM)]
        rows.append(
            {
                "record_id": f"synthetic:{idx}",
                "target_vector": vec,
                "third_class": True,
            }
        )
    return rows


def _energy_for_vector(vector: list[float], target_high: float, target_front: float, high_idx: int, front_idx: int) -> float:
    high = float(vector[high_idx])
    front = float(vector[front_idx])
    return float(0.5 * (((high - target_high) ** 2) + ((front - target_front) ** 2)))


def _summarize(energies: list[float], label: str, high_target: float, front_target: float) -> dict[str, object]:
    if not energies:
        raise SystemExit("no energies computed; check subset selection")
    return {
        "label": label,
        "hypothesis": {
            "high": float(high_target),
            "front": float(front_target),
        },
        "n_records": int(len(energies)),
        "energy_mean": float(mean(energies)),
        "energy_std": float(pstdev(energies)) if len(energies) > 1 else 0.0,
        "energy_min": float(min(energies)),
        "energy_max": float(max(energies)),
        "energy_sum": float(sum(energies)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare mask A/B hypothesis energies and emit structured JSON",
    )
    _ = parser.add_argument(
        "--targets",
        type=Path,
        default=None,
        help="Optional target JSONL path (expects target_vector length 32)",
    )
    _ = parser.add_argument("--limit", type=int, default=None, help="Optional cap on loaded rows")
    _ = parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic vectors instead of loading targets",
    )
    _ = parser.add_argument("--synthetic-n", type=int, default=256, help="Synthetic record count")
    _ = parser.add_argument("--seed", type=int, default=7, help="Deterministic seed")
    _ = parser.add_argument(
        "--high-index",
        type=int,
        default=8,
        help="Feature index used for +high/-high hypothesis",
    )
    _ = parser.add_argument(
        "--front-index",
        type=int,
        default=9,
        help="Feature index used for +front/-front hypothesis",
    )
    _ = parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/paper_gap_phase1"),
        help="Output directory",
    )
    args = parser.parse_args()

    high_index = int(args.high_index)
    front_index = int(args.front_index)
    if high_index < 0 or high_index >= VECTOR_DIM:
        raise SystemExit(f"--high-index out of range [0,{VECTOR_DIM - 1}]: {high_index}")
    if front_index < 0 or front_index >= VECTOR_DIM:
        raise SystemExit(f"--front-index out of range [0,{VECTOR_DIM - 1}]: {front_index}")

    if bool(args.synthetic):
        rows = _synthetic_rows(n=int(args.synthetic_n), seed=int(args.seed))
        source_mode = "synthetic"
    else:
        target_path = Path(args.targets) if args.targets is not None else Path("data/targets/acp_targets.jsonl")
        rows = _load_targets(path=target_path, limit=(int(args.limit) if args.limit is not None else None))
        source_mode = "targets"
    if not rows:
        raise SystemExit("no valid rows available for mask A/B energy comparison")

    subset_rows = [row for row in rows if _is_third_class_row(row)]
    subset_selector = "third_class"
    if not subset_rows:
        subset_rows = list(rows)
        subset_selector = "all_rows_fallback"

    energies_a: list[float] = []
    energies_b: list[float] = []
    for row in subset_rows:
        vec_obj = row.get("target_vector")
        if not isinstance(vec_obj, list) or len(vec_obj) != VECTOR_DIM:
            continue
        vec = [float(x) for x in vec_obj]
        energies_a.append(
            _energy_for_vector(
                vector=vec,
                target_high=1.0,
                target_front=1.0,
                high_idx=high_index,
                front_idx=front_index,
            )
        )
        energies_b.append(
            _energy_for_vector(
                vector=vec,
                target_high=-1.0,
                target_front=-1.0,
                high_idx=high_index,
                front_idx=front_index,
            )
        )

    if not energies_a or not energies_b:
        raise SystemExit("no energies computed after parsing vectors")

    energy_sum_a = float(sum(energies_a))
    energy_sum_b = float(sum(energies_b))
    a_mean = float(mean(energies_a))
    b_mean = float(mean(energies_b))

    mask_a = _summarize(
        energies=energies_a,
        label="mask_a_hypothesis_plus_high_plus_front",
        high_target=1.0,
        front_target=1.0,
    )
    mask_b = _summarize(
        energies=energies_b,
        label="mask_b_hypothesis_minus_high_minus_front",
        high_target=-1.0,
        front_target=-1.0,
    )

    delta = {
        "energy_mean_b_minus_a": float(b_mean - a_mean),
        "energy_sum_b_minus_a": float(energy_sum_b - energy_sum_a),
        "relative_mean_gap_over_a": float((b_mean - a_mean) / max(abs(a_mean), 1e-12)),
        "preferred_mask": "mask_a" if a_mean < b_mean else "mask_b",
    }
    numeric_delta_fields = {
        k: v for k, v in delta.items() if isinstance(v, (int, float))
    }
    if any(math.isnan(float(v)) or math.isinf(float(v)) for v in numeric_delta_fields.values()):
        raise SystemExit("delta contains NaN or infinite value")

    summary = {
        "mask_a": {
            **mask_a,
            "feature_indices": {"high": high_index, "front": front_index},
            "subset_selector": subset_selector,
            "source_mode": source_mode,
        },
        "mask_b": {
            **mask_b,
            "feature_indices": {"high": high_index, "front": front_index},
            "subset_selector": subset_selector,
            "source_mode": source_mode,
        },
        "delta": delta,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mask_ab_energy_summary.json"
    _ = out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out_path), "records": len(subset_rows), "subset_selector": subset_selector}))


if __name__ == "__main__":
    main()
