from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path


TSHET_UINH_REPO_URL = "https://github.com/nk2028/tshet-uinh-data"
UNIHAN_ZIP_URL = "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip"


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def _check_output(cmd: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(cmd, cwd=str(cwd) if cwd is not None else None).decode("utf-8").strip()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_url(url: str, out_path: Path) -> None:
    req = urllib.request.Request(
        url,
        headers={
            # Some hosts block default Python user agents.
            "User-Agent": "Mozilla/5.0 (compatible; pgdn-fetch/1.0; +https://github.com/)",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted official URL)
        data = resp.read()
    out_path.write_bytes(data)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _fetch_tshet_uinh(out_root: Path, *, update: bool) -> dict[str, object]:
    dst = out_root / "tshet-uinh-data"
    if dst.exists():
        if not (dst / ".git").is_dir():
            raise SystemExit(f"refusing to use existing non-git directory: {dst}")
        if update:
            _run(["git", "-C", str(dst), "pull", "--ff-only"])
    else:
        _ensure_parent_dir(dst)
        _run(["git", "clone", "--depth", "1", TSHET_UINH_REPO_URL, str(dst)])

    head = _check_output(["git", "-C", str(dst), "rev-parse", "HEAD"])
    return {
        "name": "tshet-uinh-data",
        "url": TSHET_UINH_REPO_URL,
        "path": str(dst),
        "git_head": head,
    }


def _download_unihan(out_root: Path, *, refresh: bool) -> dict[str, object]:
    dst_dir = out_root / "unihan"
    dst_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dst_dir / "Unihan.zip"

    if refresh and zip_path.exists():
        zip_path.unlink()

    if not zip_path.exists():
        print(json.dumps({"download": UNIHAN_ZIP_URL, "to": str(zip_path)}, ensure_ascii=False), flush=True)
        tmp_path = zip_path.with_suffix(".zip.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        _download_url(UNIHAN_ZIP_URL, tmp_path)
        tmp_path.replace(zip_path)

    sha = _sha256_file(zip_path)

    # Extract.
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dst_dir)

    readings = dst_dir / "Unihan_Readings.txt"
    if not readings.is_file() or readings.stat().st_size <= 0:
        raise SystemExit(f"expected Unihan_Readings.txt after extraction but not found: {readings}")

    return {
        "name": "unihan",
        "url": UNIHAN_ZIP_URL,
        "zip": str(zip_path),
        "zip_sha256": sha,
        "extracted_to": str(dst_dir),
        "required_file": str(readings),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch external linguistics datasets (local-only) into data/external",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory, e.g. data/external")
    parser.add_argument(
        "--update-tshet-uinh",
        action="store_true",
        help="If tshet-uinh-data already exists, attempt 'git pull --ff-only'.",
    )
    parser.add_argument(
        "--refresh-unihan",
        action="store_true",
        help="Re-download Unihan.zip even if it already exists.",
    )
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if shutil.which("git") is None:
        raise SystemExit("git not found in PATH (required to clone tshet-uinh-data)")

    tshet = _fetch_tshet_uinh(out_root, update=bool(args.update_tshet_uinh))
    unihan = _download_unihan(out_root, refresh=bool(args.refresh_unihan))

    prov = {
        "out": str(out_root),
        "tshet_uinh": tshet,
        "unihan": unihan,
        "note": "Artifacts are intended to remain untracked (local-only).",
    }
    print(json.dumps(prov, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
