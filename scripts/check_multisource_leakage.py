from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Hashable, TypeVar

from pgdn_torch.pgdnv0.splits import load_split_ids


SPLITS: tuple[str, ...] = ("train", "dev", "test")
KeyT = TypeVar("KeyT", bound=Hashable)


def _default_targets(repo_root: Path) -> list[Path]:
    return [
        repo_root / "data/targets/acp_targets.jsonl",
        repo_root / "data/targets/tshet_uinh_targets.jsonl",
        repo_root / "data/targets/unihan_targets.jsonl",
    ]


def _default_split_manifests(repo_root: Path) -> list[Path]:
    return [
        repo_root / "data/splits/manifest.json",
        repo_root / "data/splits/tshet_uinh_splits.json",
        repo_root / "data/splits/unihan_splits.json",
    ]


def _source_from_record_id(record_id: str) -> str:
    if record_id.startswith("tshet:"):
        return "tshet"
    if record_id.startswith("unihan:"):
        return "unihan"
    if record_id.startswith("acp:"):
        return "acp"
    return "acp_legacy"


def _derive_character(record_id: str) -> tuple[str | None, str]:
    if record_id.startswith("unihan:U+"):
        _, code = record_id.split(":", 1)
        code_value = code.split(":", 1)[0]
        try:
            return chr(int(code_value[2:], 16)), "derived_from_unihan_codepoint"
        except (TypeError, ValueError):
            return None, "unihan_codepoint_parse_failed"

    if record_id.startswith("acp:"):
        parts = record_id.split(":")
        if len(parts) >= 2 and parts[1]:
            return parts[1], "derived_from_acp_namespaced_record_id"
        return None, "acp_namespaced_record_id_missing_character"

    if ":" in record_id:
        left = record_id.split(":", 1)[0]
        if left:
            return left, "derived_from_legacy_prefix"

    return None, "no_derivation_rule"


def _load_target_index(paths: list[Path]) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    by_record_id: dict[str, dict[str, object]] = {}
    counts = {
        "rows_total": 0,
        "rows_with_character_field": 0,
        "character_derived": 0,
        "character_missing_and_skipped": 0,
        "duplicate_record_id_rows": 0,
    }

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                counts["rows_total"] += 1
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: expected JSON object")

                raw_record_id = row.get("record_id")
                if raw_record_id is None:
                    raise ValueError(f"{path}:{line_no}: missing record_id")
                record_id = str(raw_record_id)

                source_value = row.get("source")
                source = str(source_value) if source_value is not None else _source_from_record_id(record_id)

                character_value = row.get("character")
                character: str | None
                derivation_reason: str
                if isinstance(character_value, str) and character_value:
                    counts["rows_with_character_field"] += 1
                    character = character_value
                    derivation_reason = "from_field"
                else:
                    character, derivation_reason = _derive_character(record_id)
                    if character is None:
                        counts["character_missing_and_skipped"] += 1
                    else:
                        counts["character_derived"] += 1

                entry: dict[str, object] = {
                    "record_id": record_id,
                    "source": source,
                    "character": character,
                    "character_reason": derivation_reason,
                    "target_path": str(path),
                }

                if record_id in by_record_id:
                    counts["duplicate_record_id_rows"] += 1
                by_record_id[record_id] = entry

    return by_record_id, counts


def _cross_split_collisions(index: dict[KeyT, set[str]]) -> dict[KeyT, list[str]]:
    out: dict[KeyT, list[str]] = {}
    for key, split_names in index.items():
        if len(split_names) > 1:
            out[key] = sorted(split_names)
    return out


def _apply_injected_collision_fixture(split_ids: dict[str, set[str]]) -> str | None:
    if not split_ids["train"]:
        return None
    injected = next(iter(split_ids["train"]))
    split_ids["dev"].add(injected)
    return injected


def main() -> int:
    parser = argparse.ArgumentParser(description="Check cross-split leakage in multi-source manifests")
    parser.add_argument(
        "--targets",
        action="append",
        default=None,
        help="target JSONL path (repeatable). default: ACP + tshet + unihan targets",
    )
    parser.add_argument(
        "--split-manifest",
        action="append",
        default=None,
        help="split manifest JSON path (repeatable). default: ACP + tshet + unihan manifests",
    )
    parser.add_argument(
        "--split-strategy",
        default="random",
        help="split strategy key in manifest (default: random)",
    )
    parser.add_argument(
        "--check-character-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="check same-character cross-split policy and report counts (default: on)",
    )
    parser.add_argument(
        "--fail-on-character-collision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="treat character collisions as fatal (default: off; report-only)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional summary JSON output path",
    )
    parser.add_argument(
        "--inject-collision-fixture",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="inject a train->dev record_id collision for fail-path verification",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    target_paths = [Path(p) for p in args.targets] if args.targets is not None else _default_targets(repo_root)
    manifest_paths = (
        [Path(p) for p in args.split_manifest]
        if args.split_manifest is not None
        else _default_split_manifests(repo_root)
    )

    existing_targets = [p for p in target_paths if p.exists()]
    missing_targets = [str(p) for p in target_paths if not p.exists()]
    existing_manifests = [p for p in manifest_paths if p.exists()]
    missing_manifests = [str(p) for p in manifest_paths if not p.exists()]

    if not existing_targets:
        print("ERROR: no target files found", file=sys.stderr)
        return 2
    if not existing_manifests:
        print("ERROR: no split manifest files found", file=sys.stderr)
        return 2

    target_index, target_counts = _load_target_index(existing_targets)

    split_ids: dict[str, set[str]] = {split: set() for split in SPLITS}
    for manifest_path in existing_manifests:
        for split in SPLITS:
            split_ids[split].update(
                load_split_ids(manifest_path, strategy=args.split_strategy, split=split)
            )

    injected_record_id: str | None = None
    if args.inject_collision_fixture:
        injected_record_id = _apply_injected_collision_fixture(split_ids)
        if injected_record_id is None:
            print("ERROR: could not inject collision fixture because train split is empty", file=sys.stderr)
            return 2

    record_to_splits: dict[str, set[str]] = defaultdict(set)
    character_to_splits_by_source: dict[tuple[str, str], set[str]] = defaultdict(set)
    character_to_sources: dict[str, set[str]] = defaultdict(set)
    source_record_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "dev": 0, "test": 0})
    source_character_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "dev": 0, "test": 0})
    missing_target_records = 0
    missing_character_records = 0
    character_policy_skipped_records = 0

    for split in SPLITS:
        seen_characters_per_source: dict[str, set[str]] = defaultdict(set)
        for record_id in split_ids[split]:
            row = target_index.get(record_id)
            if row is None:
                missing_target_records += 1
                continue

            record_to_splits[record_id].add(split)
            source = str(row["source"])
            source_record_counts[source][split] += 1

            character = row["character"]
            if character is None:
                missing_character_records += 1
                continue

            if source == "acp_legacy":
                character_policy_skipped_records += 1
                continue

            character_str = str(character)
            character_to_splits_by_source[(source, character_str)].add(split)
            character_to_sources[character_str].add(source)
            if character_str not in seen_characters_per_source[source]:
                source_character_counts[source][split] += 1
                seen_characters_per_source[source].add(character_str)

    record_id_collisions = _cross_split_collisions(record_to_splits)
    character_collisions_by_source = {
        key: value
        for key, value in _cross_split_collisions(character_to_splits_by_source).items()
    }
    cross_source_character_overlap = {
        character: sorted(sources)
        for character, sources in character_to_sources.items()
        if len(sources) > 1
    }

    if not args.check_character_collision:
        character_collisions_by_source = {}

    violations: list[str] = []
    if record_id_collisions:
        violations.append(f"record_id_collisions={len(record_id_collisions)}")
    if character_collisions_by_source and args.fail_on_character_collision:
        violations.append(f"character_collisions={len(character_collisions_by_source)}")

    summary = {
        "status": "FAIL" if violations else "PASS",
        "split_strategy": args.split_strategy,
        "targets_checked": [str(p) for p in existing_targets],
        "split_manifests_checked": [str(p) for p in existing_manifests],
        "missing_targets": missing_targets,
        "missing_split_manifests": missing_manifests,
        "record_ids_per_split": {split: len(split_ids[split]) for split in SPLITS},
        "target_index_counts": target_counts,
        "missing_target_records_in_manifests": missing_target_records,
        "missing_character_records_in_manifests": missing_character_records,
        "character_policy_skipped_records": character_policy_skipped_records,
        "record_id_collisions": {
            "count": len(record_id_collisions),
            "examples": [
                {"record_id": k, "splits": v}
                for k, v in sorted(record_id_collisions.items())[:20]
            ],
        },
        "character_collisions": {
            "enabled": bool(args.check_character_collision),
            "fatal": bool(args.fail_on_character_collision),
            "count": len(character_collisions_by_source),
            "examples": [
                {
                    "source": source,
                    "character": character,
                    "splits": splits,
                }
                for (source, character), splits in sorted(character_collisions_by_source.items())[:20]
            ],
        },
        "cross_source_character_overlap": {
            "count": len(cross_source_character_overlap),
            "examples": [
                {"character": c, "sources": s}
                for c, s in sorted(cross_source_character_overlap.items())[:20]
            ],
        },
        "by_source": {
            source: {
                "record_ids_per_split": source_record_counts[source],
                "distinct_characters_per_split": source_character_counts[source],
            }
            for source in sorted(source_record_counts)
        },
        "fixture": {
            "inject_collision_fixture": bool(args.inject_collision_fixture),
            "injected_record_id": injected_record_id,
        },
    }

    print(f"split_strategy={args.split_strategy}")
    print(f"targets_checked={len(existing_targets)} split_manifests_checked={len(existing_manifests)}")
    print(f"missing_target_records_in_manifests={missing_target_records}")
    print(f"missing_character_records_in_manifests={missing_character_records}")
    print(f"character_policy_skipped_records={character_policy_skipped_records}")
    print(f"record_id_collisions={len(record_id_collisions)}")
    print(
        "character_collisions="
        f"{len(character_collisions_by_source)} enabled={bool(args.check_character_collision)}"
    )
    print(f"character_collisions_fatal={bool(args.fail_on_character_collision)}")
    if args.inject_collision_fixture:
        print(f"fixture_injected_record_id={injected_record_id}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"summary_json={args.out}")

    if violations:
        print(f"leakage_status=FAIL details={','.join(violations)}", file=sys.stderr)
        return 1

    print("leakage_status=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
