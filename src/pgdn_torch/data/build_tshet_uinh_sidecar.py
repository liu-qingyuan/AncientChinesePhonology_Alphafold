from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TshetEntry:
    xiaoyun_id: str
    entry_id: str
    head_char: str
    phonological_status: str
    fanqie: str
    zhiyin: str
    rhyme: str


def _hash_to_unit_floats(key: str, n: int) -> list[float]:
    out: list[float] = []
    counter = 0
    while len(out) < n:
        digest = hashlib.sha256(f"{key}|{counter}".encode("utf-8")).digest()
        for b in digest:
            out.append((float(b) / 127.5) - 1.0)
            if len(out) >= n:
                break
        counter += 1
    return out


def _slot_vector(slot_key: str | None) -> list[float]:
    if slot_key is None:
        return [0.0] * 8
    key = slot_key.strip()
    if not key:
        return [0.0] * 8
    return _hash_to_unit_floats(key, 8)


def _split_ids(ids: list[str], seed: int) -> dict[str, list[str]]:
    ordered = list(ids)
    random.Random(seed).shuffle(ordered)
    n = len(ordered)
    if n == 1:
        return {"train": ordered[:], "dev": [], "test": ordered[:1]}

    n_train = max(1, int(n * 0.7))
    n_dev = max(1, int(n * 0.15))
    if n_train + n_dev >= n:
        n_dev = 1
        n_train = max(1, n - 2)
    out = {
        "train": ordered[:n_train],
        "dev": ordered[n_train : n_train + n_dev],
        "test": ordered[n_train + n_dev :],
    }
    if not out["test"]:
        out["test"] = ordered[-1:]
    return out


def _iter_entries(source_csv: Path, limit: int | None = None) -> tuple[list[TshetEntry], int]:
    rows: list[TshetEntry] = []
    scanned = 0
    with source_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            scanned += 1
            head_char = str(raw.get("字頭") or "").strip()
            if not head_char:
                continue
            xiaoyun_id = str(raw.get("小韻號") or "").strip()
            entry_id = str(raw.get("小韻字號") or "").strip()
            phonological_status = str(raw.get("音韻地位") or "").strip()
            fanqie = str(raw.get("反切") or "").strip()
            zhiyin = str(raw.get("直音") or "").strip()
            rhyme = str(raw.get("韻目原貌") or "").strip()
            if not xiaoyun_id or not entry_id or not phonological_status:
                continue

            rows.append(
                TshetEntry(
                    xiaoyun_id=xiaoyun_id,
                    entry_id=entry_id,
                    head_char=head_char,
                    phonological_status=phonological_status,
                    fanqie=fanqie,
                    zhiyin=zhiyin,
                    rhyme=rhyme,
                )
            )
            if limit is not None and len(rows) >= max(int(limit), 0):
                break
    return rows, scanned


def _to_row(entry: TshetEntry) -> dict[str, object]:
    record_id = f"tshet:guangyun:{entry.xiaoyun_id}:{entry.entry_id}"
    fanqie_or_zhiyin = entry.fanqie if entry.fanqie else entry.zhiyin
    context_key = f"{entry.rhyme}|{entry.xiaoyun_id}|{entry.entry_id}"
    target_vector = (
        _slot_vector(entry.head_char)
        + _slot_vector(entry.phonological_status)
        + _slot_vector(fanqie_or_zhiyin)
        + _slot_vector(context_key)
    )
    mask = {
        "I": 1 if entry.head_char else 0,
        "M": 1 if entry.phonological_status else 0,
        "N": 1 if fanqie_or_zhiyin else 0,
        "C": 1,
    }
    return {
        "record_id": record_id,
        "graph_id": f"graph:tshet:{entry.phonological_status}",
        "source": "tshet_uinh_sidecar_v1",
        "character": entry.head_char,
        "xiaoyun_id": entry.xiaoyun_id,
        "entry_id": entry.entry_id,
        "target_vector": target_vector,
        "mask": mask,
    }


def build_tshet_uinh_sidecar(
    root: Path,
    targets_path: Path,
    splits_path: Path,
    meta_path: Path,
    limit: int | None = None,
    seed: int = 7,
) -> dict[str, object]:
    source_csv = root / "韻書" / "廣韻.csv"
    if not source_csv.is_file():
        raise FileNotFoundError(f"tshet-uinh source file not found: {source_csv}")

    entries, scanned_rows = _iter_entries(source_csv, limit=limit)
    if not entries:
        raise ValueError(f"no tshet-uinh entries found in {source_csv}")

    rows = [_to_row(e) for e in entries]

    targets_path.parent.mkdir(parents=True, exist_ok=True)
    with targets_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    record_ids = [str(r["record_id"]) for r in rows]
    random_splits = _split_ids(record_ids, seed=seed)
    split_obj = {
        "random": random_splits,
        "temporal": random_splits,
    }
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(split_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta = {
        "source_root": str(root),
        "source_csv": str(source_csv),
        "input_counts": {
            "rows_scanned": int(scanned_rows),
            "rows_emitted": int(len(rows)),
            "rows_filtered": int(scanned_rows - len(rows)),
        },
        "outputs": {
            "targets_path": str(targets_path),
            "splits_path": str(splits_path),
        },
        "seed": int(seed),
        "limit": None if limit is None else int(limit),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "targets": targets_path,
        "splits": splits_path,
        "meta": meta_path,
        "rows": len(rows),
        "scanned_rows": scanned_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build tshet-uinh sidecar targets/splits for Phase1")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/external/links/tshet-uinh-data"),
        help="Path to tshet-uinh-data repository root",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("data/targets/tshet_uinh_targets.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/splits/tshet_uinh_splits.json"),
        help="Output split manifest path",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("runs/multidataset_phase1/sidecars/tshet_uinh_builder_meta.json"),
        help="Output builder metadata path",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    result = build_tshet_uinh_sidecar(
        root=Path(args.root),
        targets_path=Path(args.targets),
        splits_path=Path(args.splits),
        meta_path=Path(args.meta),
        limit=args.limit,
        seed=int(args.seed),
    )
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in result.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
