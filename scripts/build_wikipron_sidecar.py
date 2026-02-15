#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path


KNOWN_LICENSES = {"CC-BY-SA-3.0"}


@dataclass(frozen=True)
class WikiPronSource:
    file_name: str
    iso_code: str
    iso_language_name: str
    wiktionary_language_name: str
    script: str
    dialect: str
    filtered: str
    transcription: str


@dataclass(frozen=True)
class WikiPronEntry:
    source: WikiPronSource
    line_no: int
    word: str
    pronunciation: str
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


def _collect_han_sources(summary_path: Path) -> list[WikiPronSource]:
    out: list[WikiPronSource] = []
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 8:
                continue
            source = WikiPronSource(
                file_name=str(row[0]).strip(),
                iso_code=str(row[1]).strip(),
                iso_language_name=str(row[2]).strip(),
                wiktionary_language_name=str(row[3]).strip(),
                script=str(row[4]).strip(),
                dialect=str(row[5]).strip(),
                filtered=str(row[6]).strip(),
                transcription=str(row[7]).strip(),
            )
            if source.script != "Han":
                continue
            if source.transcription != "Broad":
                continue
            if source.filtered != "False":
                continue
            out.append(source)
    return out


def _iter_entries(
    source_path: Path,
    limit: int | None = None,
) -> tuple[list[WikiPronEntry], int, int, int, list[str]]:
    summary_path = source_path / "data" / "scrape" / "summary.tsv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"wikipron summary file not found: {summary_path}")

    sources = _collect_han_sources(summary_path)
    if not sources:
        raise ValueError(f"no Han broad sources listed in {summary_path}")

    entries: list[WikiPronEntry] = []
    scanned = 0
    skipped_unknown_license = 0
    skipped_bad_rows = 0
    used_files: list[str] = []

    for source in sources:
        source_file = source_path / "data" / "scrape" / "tsv" / source.file_name
        if not source_file.is_file():
            raise FileNotFoundError(f"wikipron source file not found: {source_file}")

        used_files.append(str(source_file))
        with source_file.open("r", encoding="utf-8") as f:
            for i, raw_line in enumerate(f, start=1):
                line = raw_line.rstrip("\n")
                scanned += 1
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    skipped_bad_rows += 1
                    continue
                word = parts[0].strip()
                pronunciation = parts[1].strip()
                if not word or not pronunciation:
                    skipped_bad_rows += 1
                    continue

                license_name = "CC-BY-SA-3.0"
                if license_name not in KNOWN_LICENSES:
                    skipped_unknown_license += 1
                    continue

                entries.append(
                    WikiPronEntry(
                        source=source,
                        line_no=i,
                        word=word,
                        pronunciation=pronunciation,
                        license_name=license_name,
                    )
                )
                if limit is not None and len(entries) >= max(int(limit), 0):
                    return (
                        entries,
                        scanned,
                        skipped_unknown_license,
                        skipped_bad_rows,
                        used_files,
                    )

    return entries, scanned, skipped_unknown_license, skipped_bad_rows, used_files


def _row_hash(entry: WikiPronEntry) -> str:
    payload = "\t".join(
        [
            entry.source.file_name,
            entry.source.iso_code,
            entry.word,
            entry.pronunciation,
            str(entry.line_no),
        ]
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _record_id(entry: WikiPronEntry) -> str:
    payload = "\t".join([entry.source.iso_code, entry.source.file_name, entry.word, entry.pronunciation])
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"wikipron:{entry.source.iso_code}:{digest}"


def _to_row(entry: WikiPronEntry) -> dict[str, object]:
    record_id = _record_id(entry)
    context_key = "|".join(
        [
            entry.source.file_name,
            entry.source.script,
            entry.source.dialect,
            str(entry.line_no),
        ]
    )
    target_vector = (
        _slot_vector(entry.word)
        + _slot_vector(entry.pronunciation)
        + _slot_vector(f"{entry.source.iso_code}|{entry.source.script}|{entry.source.dialect}")
        + _slot_vector(context_key)
    )
    mask = {
        "I": 1 if entry.word else 0,
        "M": 1 if entry.pronunciation else 0,
        "N": 1 if entry.source.iso_code else 0,
        "C": 1,
    }
    provenance = {
        "dataset_id": "wikipron",
        "license": entry.license_name,
        "source_url": "https://en.wiktionary.org/wiki/Wiktionary:Copyrights",
        "hash": _row_hash(entry),
        "generated_flag": False,
        "generated_policy_ref": "runs/multidataset_phase2/contracts/contamination_policy.json",
    }
    return {
        "record_id": record_id,
        "graph_id": f"graph:wikipron:lang:{entry.source.iso_code}",
        "source": "wikipron_sidecar_v2",
        "character": entry.word,
        "ipa": entry.pronunciation,
        "iso_code": entry.source.iso_code,
        "iso_language_name": entry.source.iso_language_name,
        "wiktionary_language_name": entry.source.wiktionary_language_name,
        "script": entry.source.script,
        "dialect": entry.source.dialect,
        "source_file": entry.source.file_name,
        "source_line": entry.line_no,
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


def build_wikipron_sidecar(
    source_path: Path,
    targets_path: Path,
    splits_path: Path,
    meta_path: Path,
    limit: int | None = None,
    seed: int = 17,
) -> dict[str, object]:
    if not source_path.exists():
        raise FileNotFoundError(f"wikipron source directory not found: {source_path}")
    if not source_path.is_dir():
        raise NotADirectoryError(f"wikipron source path is not a directory: {source_path}")

    entries, scanned_rows, skipped_unknown_license, skipped_bad_rows, used_files = _iter_entries(
        source_path=source_path,
        limit=limit,
    )
    if not entries:
        raise ValueError(f"no wikipron entries found under {source_path}")

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
        "input_counts": {
            "rows_scanned": int(scanned_rows),
            "rows_emitted": int(len(rows)),
            "rows_filtered": int(scanned_rows - len(rows)),
            "rows_skipped_unknown_license": int(skipped_unknown_license),
            "rows_skipped_bad": int(skipped_bad_rows),
        },
        "sources": {
            "han_broad_files": sorted(used_files),
            "file_count": len(set(used_files)),
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
        "skipped_bad_rows": skipped_bad_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build WikiPron sidecar targets/splits for Phase2")
    parser.add_argument(
        "--source-path",
        type=Path,
        default=Path("data/external/links/wikipron"),
        help="Path to WikiPron dataset root",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("data/targets/wikipron_targets.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/splits/wikipron_splits.json"),
        help="Output split manifest path",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("runs/multidataset_phase2/sidecars/wikipron_builder_meta.json"),
        help="Output builder metadata path",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    result = build_wikipron_sidecar(
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
