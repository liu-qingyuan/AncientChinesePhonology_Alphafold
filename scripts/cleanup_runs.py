import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass
class CleanupStats:
    deleted_files: int = 0
    deleted_dirs: int = 0
    bytes_freed: int = 0


def _safe_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _collect_experiment_dirs(runs_dir: Path) -> list[Path]:
    found = []
    for csv_path in runs_dir.rglob("ranking_scores.csv"):
        found.append(csv_path.parent)
    return sorted(set(found))


def _top_record_ids(ranking_csv: Path, keep_top_k: int) -> set[str]:
    rows = []
    with ranking_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    rows.sort(key=lambda r: float(r.get("ranking_score", "-1e9")), reverse=True)
    keep = {
        r["record_id"].replace(":", "_") for r in rows[:keep_top_k] if "record_id" in r
    }
    return keep


def _delete_path(
    path: Path, dry_run: bool, stats: CleanupStats, deleted: list[str]
) -> None:
    size = _safe_size(path)
    if path.is_dir():
        if not dry_run:
            shutil.rmtree(path, ignore_errors=True)
        stats.deleted_dirs += 1
    elif path.is_file():
        if not dry_run:
            try:
                path.unlink()
            except FileNotFoundError:
                return
        stats.deleted_files += 1
    else:
        return
    stats.bytes_freed += size
    deleted.append(str(path))


def _cleanup_experiment_dir(
    exp_dir: Path,
    keep_top_k: int,
    dry_run: bool,
    stats: CleanupStats,
    deleted: list[str],
) -> dict[str, int | str]:
    ranking_csv = exp_dir / "ranking_scores.csv"
    if not ranking_csv.exists():
        return {"experiment": str(exp_dir), "kept_records": 0, "deleted_records": 0}

    keep_ids = _top_record_ids(ranking_csv, keep_top_k=keep_top_k)
    deleted_count = 0

    for sample_root in exp_dir.glob("seed-*_sample-*"):
        if not sample_root.is_dir():
            continue
        for record_dir in sample_root.iterdir():
            if not record_dir.is_dir():
                continue
            if record_dir.name not in keep_ids:
                _delete_path(record_dir, dry_run=dry_run, stats=stats, deleted=deleted)
                deleted_count += 1

    return {
        "experiment": str(exp_dir),
        "kept_records": len(keep_ids),
        "deleted_records": deleted_count,
    }


def _cleanup_checkpoints(
    runs_dir: Path,
    keep_latest: int,
    dry_run: bool,
    stats: CleanupStats,
    deleted: list[str],
) -> dict[str, int]:
    checkpoints = sorted(
        [p for p in runs_dir.rglob("checkpoints/*") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    removed = 0
    for path in checkpoints[keep_latest:]:
        _delete_path(path, dry_run=dry_run, stats=stats, deleted=deleted)
        removed += 1
    return {
        "checkpoint_files_seen": len(checkpoints),
        "checkpoint_files_deleted": removed,
    }


def _remove_empty_dirs(
    base: Path, dry_run: bool, stats: CleanupStats, deleted: list[str]
) -> int:
    removed = 0
    for path in sorted(
        [p for p in base.rglob("*") if p.is_dir()],
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        try:
            is_empty = not any(path.iterdir())
        except OSError:
            continue
        if is_empty:
            _delete_path(path, dry_run=dry_run, stats=stats, deleted=deleted)
            removed += 1
    return removed


def _write_manifest(runs_dir: Path, payload: dict[str, Any]) -> Path:
    manifest_dir = runs_dir / "retention_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = manifest_dir / f"cleanup_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _iter_repro_artifacts(runs_dir: Path) -> Iterable[Path]:
    for pattern in (
        "**/ranking_scores.csv",
        "**/eval.json",
        "**/results_table.csv",
        "retention_manifests/*.json",
    ):
        for path in runs_dir.glob(pattern):
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cleanup run artifacts with paper-style retention policy"
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--keep-top-k",
        type=int,
        default=10,
        help="Keep top-k records per experiment by ranking score",
    )
    parser.add_argument(
        "--keep-latest-checkpoints",
        type=int,
        default=2,
        help="Keep N latest checkpoint files globally",
    )
    parser.add_argument(
        "--remove-empty-dirs",
        action="store_true",
        help="Remove empty directories after cleanup",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Apply deletions. Default is dry-run."
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir
    if not runs_dir.exists():
        raise SystemExit(f"runs dir not found: {runs_dir}")

    stats = CleanupStats()
    deleted_paths: list[str] = []
    experiments = _collect_experiment_dirs(runs_dir)
    experiment_summaries = []
    for exp_dir in experiments:
        experiment_summaries.append(
            _cleanup_experiment_dir(
                exp_dir=exp_dir,
                keep_top_k=max(args.keep_top_k, 1),
                dry_run=not args.apply,
                stats=stats,
                deleted=deleted_paths,
            )
        )

    ckpt_summary = _cleanup_checkpoints(
        runs_dir=runs_dir,
        keep_latest=max(args.keep_latest_checkpoints, 0),
        dry_run=not args.apply,
        stats=stats,
        deleted=deleted_paths,
    )

    empty_removed = 0
    if args.remove_empty_dirs:
        empty_removed = _remove_empty_dirs(
            base=runs_dir,
            dry_run=not args.apply,
            stats=stats,
            deleted=deleted_paths,
        )

    reproducibility_files = sorted({str(p) for p in _iter_repro_artifacts(runs_dir)})
    manifest = {
        "mode": "apply" if args.apply else "dry-run",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "keep_top_k_records_per_experiment": max(args.keep_top_k, 1),
            "keep_latest_checkpoints": max(args.keep_latest_checkpoints, 0),
            "remove_empty_dirs": args.remove_empty_dirs,
        },
        "experiments": experiment_summaries,
        "checkpoint_summary": ckpt_summary,
        "deleted_summary": {
            "deleted_files": stats.deleted_files,
            "deleted_dirs": stats.deleted_dirs,
            "bytes_freed": stats.bytes_freed,
            "empty_dirs_removed": empty_removed,
        },
        "reproducibility_files_retained": reproducibility_files,
        "deleted_paths": deleted_paths,
    }
    manifest_path = _write_manifest(runs_dir, manifest)

    print(f"mode={manifest['mode']}")
    print(f"experiments={len(experiments)}")
    print(f"deleted_files={stats.deleted_files} deleted_dirs={stats.deleted_dirs}")
    print(f"bytes_freed={stats.bytes_freed}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
