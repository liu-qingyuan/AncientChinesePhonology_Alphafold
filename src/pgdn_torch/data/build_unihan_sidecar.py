from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UnihanEntry:
    codepoint: str
    character: str
    mandarin: str
    cantonese: str
    definition: str


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


def _resolve_readings_path(root: Path) -> Path:
    candidates = [
        root / "Unihan_Readings.txt",
        root / "Unihan" / "Unihan_Readings.txt",
        root / "ucd" / "Unihan_Readings.txt",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Unihan_Readings.txt not found under root={root}")


def _codepoint_to_char(cp: str) -> str:
    return chr(int(cp[2:], 16))


def _iter_entries(readings_path: Path, limit: int | None = None) -> tuple[list[UnihanEntry], int]:
    scanned = 0
    by_cp: dict[str, dict[str, str]] = {}

    with readings_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            scanned += 1
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            cp = parts[0].strip()
            field = parts[1].strip()
            value = parts[2].strip()
            if not cp.startswith("U+"):
                continue
            if field not in {"kMandarin", "kCantonese", "kDefinition"}:
                continue

            row = by_cp.setdefault(cp, {})
            if field not in row:
                row[field] = value

    entries: list[UnihanEntry] = []
    for cp in sorted(by_cp):
        row = by_cp[cp]
        mandarin = row.get("kMandarin", "")
        cantonese = row.get("kCantonese", "")
        definition = row.get("kDefinition", "")
        if not mandarin and not cantonese and not definition:
            continue
        entries.append(
            UnihanEntry(
                codepoint=cp,
                character=_codepoint_to_char(cp),
                mandarin=mandarin,
                cantonese=cantonese,
                definition=definition,
            )
        )
        if limit is not None and len(entries) >= max(int(limit), 0):
            break
    return entries, scanned


def _to_row(entry: UnihanEntry) -> dict[str, object]:
    record_id = f"unihan:{entry.codepoint}"
    context_key = f"{entry.codepoint}|{entry.definition}"
    target_vector = (
        _slot_vector(entry.character)
        + _slot_vector(entry.mandarin)
        + _slot_vector(entry.cantonese)
        + _slot_vector(context_key)
    )
    mask = {
        "I": 1 if entry.character else 0,
        "M": 1 if entry.mandarin else 0,
        "N": 1 if entry.cantonese else 0,
        "C": 1,
    }
    block = f"{ord(entry.character) >> 8:04X}"
    return {
        "record_id": record_id,
        "graph_id": f"graph:unihan:block:{block}",
        "source": "unihan_sidecar_v1",
        "character": entry.character,
        "codepoint": entry.codepoint,
        "mandarin": entry.mandarin,
        "cantonese": entry.cantonese,
        "target_vector": target_vector,
        "mask": mask,
    }


def build_unihan_sidecar(
    root: Path,
    targets_path: Path,
    splits_path: Path,
    meta_path: Path,
    limit: int | None = None,
    seed: int = 11,
) -> dict[str, object]:
    readings_path = _resolve_readings_path(root)
    entries, scanned_rows = _iter_entries(readings_path, limit=limit)
    if not entries:
        raise ValueError(f"no Unihan entries found in {readings_path}")

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
        "readings_path": str(readings_path),
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
    parser = argparse.ArgumentParser(description="Build Unihan sidecar targets/splits for Phase1")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/external/links/unihan"),
        help="Path to Unihan extracted root",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("data/targets/unihan_targets.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/splits/unihan_splits.json"),
        help="Output split manifest path",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("runs/multidataset_phase1/sidecars/unihan_builder_meta.json"),
        help="Output builder metadata path",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    result = build_unihan_sidecar(
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
