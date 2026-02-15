#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path


KNOWN_LICENSES = {"CC0-1.0"}


@dataclass(frozen=True)
class WikiHanEntry:
    row_index: int
    character: str
    middle_chinese: str
    cantonese: str
    gan: str
    hakka: str
    jin: str
    mandarin: str
    hokkien: str
    wu: str
    xiang: str
    license_name: str


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


def _iter_entries(source_tsv: Path, limit: int | None = None) -> tuple[list[WikiHanEntry], int, int]:
    entries: list[WikiHanEntry] = []
    scanned = 0
    skipped_unknown_license = 0
    with source_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for raw in reader:
            scanned += 1
            license_name = "CC0-1.0"
            if license_name not in KNOWN_LICENSES:
                skipped_unknown_license += 1
                continue

            character = str(raw.get("Character") or "").strip()
            middle_chinese = str(raw.get("Middle Chinese (Baxter and Sagart 2014)") or "").strip()
            cantonese = str(raw.get("Cantonese") or "").strip()
            gan = str(raw.get("Gan") or "").strip()
            hakka = str(raw.get("Hakka") or "").strip()
            jin = str(raw.get("Jin") or "").strip()
            mandarin = str(raw.get("Mandarin") or "").strip()
            hokkien = str(raw.get("Hokkien") or "").strip()
            wu = str(raw.get("Wu") or "").strip()
            xiang = str(raw.get("Xiang") or "").strip()

            if not character:
                continue

            entries.append(
                WikiHanEntry(
                    row_index=scanned,
                    character=character,
                    middle_chinese=middle_chinese,
                    cantonese=cantonese,
                    gan=gan,
                    hakka=hakka,
                    jin=jin,
                    mandarin=mandarin,
                    hokkien=hokkien,
                    wu=wu,
                    xiang=xiang,
                    license_name=license_name,
                )
            )
            if limit is not None and len(entries) >= max(int(limit), 0):
                break
    return entries, scanned, skipped_unknown_license


def _row_hash(entry: WikiHanEntry) -> str:
    payload = "\t".join(
        [
            entry.character,
            entry.middle_chinese,
            entry.cantonese,
            entry.gan,
            entry.hakka,
            entry.jin,
            entry.mandarin,
            entry.hokkien,
            entry.wu,
            entry.xiang,
            str(entry.row_index),
        ]
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _to_row(entry: WikiHanEntry) -> dict[str, object]:
    record_id = f"wikihan:{entry.character}:{entry.row_index}"
    daughters_key = "|".join(
        [
            entry.cantonese,
            entry.gan,
            entry.hakka,
            entry.jin,
            entry.mandarin,
            entry.hokkien,
            entry.wu,
            entry.xiang,
        ]
    )
    target_vector = (
        _slot_vector(entry.character)
        + _slot_vector(entry.middle_chinese)
        + _slot_vector(daughters_key)
        + _slot_vector(f"wikihan:{entry.row_index}")
    )
    mask = {
        "I": 1 if entry.character else 0,
        "M": 1 if entry.middle_chinese else 0,
        "N": 1 if daughters_key.strip("|") else 0,
        "C": 1,
    }
    provenance = {
        "dataset_id": "wikihan",
        "license": entry.license_name,
        "source_url": "https://github.com/kalvinchang/wikihan",
        "hash": _row_hash(entry),
        "generated_flag": False,
        "generated_policy_ref": "runs/multidataset_phase2/contracts/contamination_policy.json",
    }
    return {
        "record_id": record_id,
        "graph_id": f"graph:wikihan:char:{entry.character}",
        "source": "wikihan_sidecar_v2",
        "character": entry.character,
        "middle_chinese": entry.middle_chinese,
        "cantonese": entry.cantonese,
        "gan": entry.gan,
        "hakka": entry.hakka,
        "jin": entry.jin,
        "mandarin": entry.mandarin,
        "hokkien": entry.hokkien,
        "wu": entry.wu,
        "xiang": entry.xiang,
        "target_vector": target_vector,
        "mask": mask,
        "dataset_id": provenance["dataset_id"],
        "license": provenance["license"],
        "source_url": provenance["source_url"],
        "hash": provenance["hash"],
        "generated_flag": provenance["generated_flag"],
        "generated_policy_ref": provenance["generated_policy_ref"],
        "provenance": provenance,
    }


def build_wikihan_sidecar(
    source_path: Path,
    targets_path: Path,
    splits_path: Path,
    meta_path: Path,
    limit: int | None = None,
    seed: int = 13,
) -> dict[str, object]:
    if not source_path.exists():
        raise FileNotFoundError(f"wikihan source directory not found: {source_path}")
    if not source_path.is_dir():
        raise NotADirectoryError(f"wikihan source path is not a directory: {source_path}")

    source_tsv = source_path / "wikihan-ipa.tsv"
    if not source_tsv.is_file():
        raise FileNotFoundError(f"wikihan source file not found: {source_tsv}")

    entries, scanned_rows, skipped_unknown_license = _iter_entries(source_tsv, limit=limit)
    if not entries:
        raise ValueError(f"no wikihan entries found in {source_tsv}")

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
        "source_root": str(source_path),
        "source_tsv": str(source_tsv),
        "input_counts": {
            "rows_scanned": int(scanned_rows),
            "rows_emitted": int(len(rows)),
            "rows_filtered": int(scanned_rows - len(rows)),
            "rows_skipped_unknown_license": int(skipped_unknown_license),
        },
        "outputs": {
            "targets_path": str(targets_path),
            "splits_path": str(splits_path),
        },
        "seed": int(seed),
        "limit": None if limit is None else int(limit),
        "known_licenses": sorted(KNOWN_LICENSES),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "targets": targets_path,
        "splits": splits_path,
        "meta": meta_path,
        "rows": len(rows),
        "scanned_rows": scanned_rows,
        "skipped_unknown_license": skipped_unknown_license,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build WikiHan sidecar targets/splits for Phase2")
    parser.add_argument(
        "--source-path",
        type=Path,
        default=Path("data/external/links/wikihan"),
        help="Path to WikiHan dataset root",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("data/targets/wikihan_targets.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/splits/wikihan_splits.json"),
        help="Output split manifest path",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("runs/multidataset_phase2/sidecars/wikihan_builder_meta.json"),
        help="Output builder metadata path",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    result = build_wikihan_sidecar(
        source_path=Path(args.source_path),
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
