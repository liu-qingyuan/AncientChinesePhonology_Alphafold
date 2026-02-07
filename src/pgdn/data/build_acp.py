import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .io import write_json, write_jsonl

LANGUAGES = ["MiddleTang", "LateTang", "Song", "Yuan", "MingQing", "Mandarin"]


def normalize_pron(pron: str) -> str:
    parts = [p.strip() for p in pron.split("-") if p.strip()]
    return " ".join(parts)


def split_slots(pron: str) -> Dict[str, Optional[str]]:
    parts = [p.strip() for p in pron.split("-") if p.strip()]
    slots: Dict[str, Optional[str]] = {"I": None, "M": None, "N": None, "C": None}
    if not parts:
        return slots
    if len(parts) == 1:
        slots["N"] = parts[0]
        return slots
    if len(parts) == 2:
        slots["I"] = parts[0]
        slots["N"] = parts[1]
        return slots
    slots["I"] = parts[0]
    slots["M"] = parts[1]
    slots["N"] = parts[2]
    if len(parts) >= 4:
        slots["C"] = parts[3]
    return slots


def slot_mask(slots: Dict[str, Optional[str]]) -> Dict[str, int]:
    return {k: int(v is not None and v != "") for k, v in slots.items()}


def token_vector(token: Optional[str], dim: int = 8) -> List[float]:
    if not token:
        return [0.0] * dim
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    values = []
    for i in range(dim):
        values.append((digest[i] / 255.0) * 2.0 - 1.0)
    return values


def build_target_vector(
    slots: Dict[str, Optional[str]], dim_per_slot: int = 8
) -> List[float]:
    vector = []
    for key in ("I", "M", "N", "C"):
        vector.extend(token_vector(slots.get(key), dim=dim_per_slot))
    return vector


def hash_partition(record_id: str, seed: int) -> float:
    blob = f"{record_id}|{seed}".encode("utf-8")
    digest = hashlib.sha1(blob).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def random_split_ids(record_ids: List[str], seed: int) -> Dict[str, List[str]]:
    train, dev, test = [], [], []
    for rid in record_ids:
        p = hash_partition(rid, seed)
        if p < 0.8:
            train.append(rid)
        elif p < 0.9:
            dev.append(rid)
        else:
            test.append(rid)
    return {"train": train, "dev": dev, "test": test}


def temporal_split(records: List[Dict[str, object]]) -> Dict[str, List[str]]:
    period_order = {
        "MiddleTang": 0,
        "LateTang": 1,
        "Song": 2,
        "Yuan": 3,
        "MingQing": 4,
        "Mandarin": 5,
    }
    split = {"train": [], "dev": [], "test": []}
    for rec in records:
        rid = str(rec["record_id"])
        p = str(rec["period"])
        idx = period_order[p]
        if idx <= 2:
            split["train"].append(rid)
        elif idx == 3:
            split["dev"].append(rid)
        else:
            split["test"].append(rid)
    return split


def low_resource_subsets(train_ids: List[str], seed: int) -> Dict[str, List[str]]:
    levels = [0.10, 0.05, 0.01, 0.00]
    rng = random.Random(seed)
    ids = list(train_ids)
    rng.shuffle(ids)
    out: Dict[str, List[str]] = {}
    for lv in levels:
        n = int(len(ids) * lv)
        out[f"{int(lv * 100)}pct"] = sorted(ids[:n])
    return out


def build_graphs_and_targets(
    csv_path: Path, out_dir: Path, seed: int
) -> Tuple[int, int]:
    by_language_nodes = defaultdict(list)
    by_language_pron = defaultdict(lambda: defaultdict(list))
    targets = []
    anchors = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            char = row["character"]
            for language in LANGUAGES:
                pron = (row.get(language) or "").strip()
                if not pron:
                    continue
                slots = split_slots(pron)
                mask = slot_mask(slots)
                normalized = normalize_pron(pron)
                record_id = f"{char}:{language}"
                graph_id = f"graph:{language}"
                node = {
                    "id": record_id,
                    "type": "char",
                    "features": {
                        "glyph": char,
                        "known_phon": build_target_vector(slots),
                        "time": language,
                    },
                }
                by_language_nodes[language].append(node)
                by_language_pron[language][normalized].append(record_id)
                target = {
                    "record_id": record_id,
                    "character": char,
                    "language": language,
                    "period": language,
                    "syllable_slots": slots,
                    "target_vector": build_target_vector(slots),
                    "mask": mask,
                    "graph_id": graph_id,
                    "normalized_pron": normalized,
                }
                targets.append(target)

    graphs = []
    for language, nodes in by_language_nodes.items():
        edges = []
        for pron, node_ids in by_language_pron[language].items():
            if len(node_ids) < 2:
                continue
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    edges.append(
                        {
                            "src": node_ids[i],
                            "dst": node_ids[j],
                            "type": "same_pron",
                            "features": {"pron": pron},
                            "weight": 1.0,
                        }
                    )
        graphs.append(
            {
                "graph_id": f"graph:{language}",
                "language": language,
                "nodes": nodes,
                "edges": edges,
            }
        )

    record_ids = [t["record_id"] for t in targets]
    random_split = random_split_ids(record_ids, seed=seed)
    temporal = temporal_split(targets)
    low_resource = low_resource_subsets(random_split["train"], seed=seed)

    split_manifest = {
        "seed": seed,
        "random": random_split,
        "temporal": temporal,
        "low_resource_train_anchors": low_resource,
        "leakage_rules": [
            "same (character, language, period) cannot cross splits",
            "same normalized pronunciation group cannot cross random partitions",
            "future periods not used as labels in temporal eval",
            "low-resource downsampling applied to train anchors only",
        ],
    }

    write_jsonl(out_dir / "graphs" / "acp_graphs.jsonl", graphs)
    write_jsonl(out_dir / "targets" / "acp_targets.jsonl", targets)
    write_jsonl(out_dir / "anchors" / "acp_anchors.jsonl", anchors)
    write_json(out_dir / "splits" / "manifest.json", split_manifest)
    return len(graphs), len(targets)


def validate_targets(path: Path) -> Tuple[int, int]:
    required = {
        "record_id",
        "character",
        "language",
        "period",
        "syllable_slots",
        "target_vector",
        "mask",
        "graph_id",
    }
    ok = 0
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if all(k in rec for k in required):
                ok += 1
            else:
                bad += 1
    return ok, bad


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ACP-first PGDN v0 data")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", default=Path("data"), type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    num_graphs, num_targets = build_graphs_and_targets(args.csv, args.out, args.seed)
    ok, bad = validate_targets(args.out / "targets" / "acp_targets.jsonl")
    print(f"graphs={num_graphs} targets={num_targets} valid={ok} invalid={bad}")
    if bad > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
