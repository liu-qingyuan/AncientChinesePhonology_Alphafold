from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

from pgdn_torch.pgdnv0.splits import load_split_ids


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


def _split_ids(ids: list[str], seed: int) -> dict[str, list[str]]:
    keyed = [
        (
            hashlib.sha256(f"synth_phase2|seed={seed}|record={rid}".encode("utf-8")).hexdigest(),
            rid,
        )
        for rid in ids
    ]
    ordered = [rid for _, rid in sorted(keyed)]

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


def _build_row(idx: int, seed: int) -> dict[str, object]:
    record_id = f"synth_phase2:s{seed}:{idx:07d}"
    family = int(hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:2], 16) % 64
    mask_raw = hashlib.sha256(f"mask|{record_id}".encode("utf-8")).digest()
    mask = {
        "I": 1 if (mask_raw[0] % 5) != 0 else 0,
        "M": 1 if (mask_raw[1] % 5) != 0 else 0,
        "N": 1 if (mask_raw[2] % 5) != 0 else 0,
        "C": 1,
    }

    return {
        "record_id": record_id,
        "graph_id": f"graph:synth_phase2:family:{family:02d}",
        "source": "synth_phase2_v1",
        "target_vector": _hash_to_unit_floats(f"target|{record_id}", 32),
        "mask": mask,
        "generated_flag": True,
    }


def _clamp_unit(x: float) -> float:
    if x < -1.0:
        return -1.0
    if x > 1.0:
        return 1.0
    return x


def _tree_leaf_ids() -> list[str]:
    families = [f"f{family:02d}" for family in range(8)]
    branches = ["a", "b"]
    leaves = ["x", "y"]
    out: list[str] = []
    for family in families:
        for branch in branches:
            for leaf in leaves:
                out.append(f"{family}.{branch}.{leaf}")
    return out


def _tree_target_vector(record_id: str, leaf_id: str, seed: int) -> list[float]:
    root = _hash_to_unit_floats(f"tree|root|seed={seed}", 32)
    family = leaf_id.split(".")[0]
    branch = ".".join(leaf_id.split(".")[:2])
    leaf = leaf_id
    noise = _hash_to_unit_floats(f"tree|noise|{record_id}", 32)
    family_shift = _hash_to_unit_floats(f"tree|family|seed={seed}|{family}", 32)
    branch_shift = _hash_to_unit_floats(f"tree|branch|seed={seed}|{branch}", 32)
    leaf_shift = _hash_to_unit_floats(f"tree|leaf|seed={seed}|{leaf}", 32)
    out: list[float] = []
    for i in range(32):
        x = (
            root[i]
            + 0.25 * family_shift[i]
            + 0.2 * branch_shift[i]
            + 0.15 * leaf_shift[i]
            + 0.1 * noise[i]
        )
        out.append(_clamp_unit(x))
    return out


def _build_tree_row(idx: int, seed: int, leaf_ids: list[str]) -> dict[str, object]:
    rng = random.Random(seed * 1_000_003 + idx)
    leaf_id = leaf_ids[rng.randrange(len(leaf_ids))]
    record_id = f"synth_phase2_tree:s{seed}:{leaf_id}:{idx:07d}"
    mask_raw = hashlib.sha256(f"tree_mask|{record_id}".encode("utf-8")).digest()
    mask = {
        "I": 1 if (mask_raw[0] % 5) != 0 else 0,
        "M": 1 if (mask_raw[1] % 5) != 0 else 0,
        "N": 1 if (mask_raw[2] % 5) != 0 else 0,
        "C": 1,
    }
    return {
        "record_id": record_id,
        "graph_id": f"graph:synth_phase2_tree:{leaf_id}",
        "source": "synth_phase2_tree_v1",
        "target_vector": _tree_target_vector(record_id=record_id, leaf_id=leaf_id, seed=seed),
        "mask": mask,
        "generated_flag": True,
    }


def _validate_provenance_core(provenance: Mapping[str, object]) -> None:
    required = ["dataset_id", "license", "source_url", "hash", "generated_flag"]
    for key in required:
        if key not in provenance:
            raise ValueError(f"provenance missing required field: {key}")
    if not isinstance(provenance["generated_flag"], bool):
        raise ValueError("provenance.generated_flag must be boolean")

    source_url = str(provenance["source_url"])
    parsed = urlparse(source_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"provenance.source_url must be uri-like: {source_url!r}")


def build_synth_phase2(
    out_root: Path,
    provenance_path: Path,
    n: int,
    seed: int,
    mode: str = "hash",
) -> dict[str, object]:
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    if mode not in {"hash", "tree"}:
        raise ValueError(f"unsupported mode: {mode!r}")

    if mode == "tree":
        targets_path = out_root / "targets" / "synthetic_tree_targets.jsonl"
        splits_path = out_root / "splits" / "synthetic_tree_splits.json"
        leaf_ids = _tree_leaf_ids()
        rows = [_build_tree_row(idx=i, seed=seed, leaf_ids=leaf_ids) for i in range(n)]
        topology_source = "builtin_balanced_binary_tree_v1"
        rule_set_version = "tree_rules_v1"
        source_url = "https://example.invalid/multidataset_phase2/synthetic/tree"
        source_name = "synth_phase2_tree_v1"
    else:
        targets_path = out_root / "targets" / "synth_phase2_targets.jsonl"
        splits_path = out_root / "splits" / "synth_phase2_splits.json"
        rows = [_build_row(idx=i, seed=seed) for i in range(n)]
        topology_source = "none_hash_legacy"
        rule_set_version = "hash_rules_v1"
        source_url = "https://example.invalid/multidataset_phase2/synthetic"
        source_name = "synth_phase2_v1"

    record_ids = [str(r["record_id"]) for r in rows]
    random_splits = _split_ids(record_ids, seed=seed)
    split_obj = {"random": random_splits, "temporal": random_splits}

    targets_path.parent.mkdir(parents=True, exist_ok=True)
    with targets_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(split_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    targets_digest = hashlib.sha256(targets_path.read_bytes()).hexdigest()
    splits_digest = hashlib.sha256(splits_path.read_bytes()).hexdigest()
    provenance = {
        "dataset_id": "synth_phase2",
        "license": "CC0-1.0",
        "source_url": source_url,
        "hash": f"sha256:{targets_digest}|sha256:{splits_digest}",
        "generated_flag": True,
        "generated_policy_ref": "runs/multidataset_phase2/contracts/contamination_policy.json",
        "generator": {
            "script": "scripts/build_phase2_synthetic_pretrain_data.py",
            "seed": int(seed),
            "n": int(n),
            "mode": mode,
            "topology_source": topology_source,
            "rule_set_version": rule_set_version,
        },
        "outputs": {
            "targets_path": str(targets_path),
            "splits_path": str(splits_path),
        },
    }
    _validate_provenance_core(provenance)

    metadata: dict[str, object] = dict(provenance)
    metadata.update(
        {
            "mode": mode,
            "seed": int(seed),
            "source": source_name,
            "provenance": {
                "seed": int(seed),
                "topology_source": topology_source,
                "rule_set_version": rule_set_version,
                "generator_script": "scripts/build_phase2_synthetic_pretrain_data.py",
            },
            "rows": len(rows),
        }
    )

    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for strategy in ("random", "temporal"):
        for split in ("train", "dev", "test"):
            load_split_ids(splits_path, strategy=strategy, split=split)

    return {
        "targets": targets_path,
        "splits": splits_path,
        "provenance": provenance_path,
        "rows": len(rows),
        "seed": int(seed),
        "mode": mode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic synthetic pretraining data for Phase2")
    parser.add_argument("--out", type=Path, default=Path("data"), help="Output data root (default: data)")
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("runs/multidataset_phase2/synthetic/provenance.json"),
        help="Output provenance JSON path",
    )
    parser.add_argument("--n", type=int, default=5000, help="Number of synthetic rows")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument(
        "--mode",
        type=str,
        default="hash",
        choices=["hash", "tree"],
        help="Synthetic generation mode (default: hash)",
    )
    args = parser.parse_args()

    result = build_synth_phase2(
        out_root=Path(args.out),
        provenance_path=Path(args.provenance),
        n=int(args.n),
        seed=int(args.seed),
        mode=str(args.mode),
    )
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in result.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
