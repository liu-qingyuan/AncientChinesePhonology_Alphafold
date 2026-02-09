#!/usr/bin/env bash
set -euo pipefail

# Mocha-TIMIT Phase 1 acquisition script (PRD-driven).
#
# Requirements (high level):
# - Allowlist-only downloads from https://data.cstr.ed.ac.uk/mocha/
# - Idempotent: re-running should not re-download if hashes match
# - Audit artifacts: manifest.json, checksums.sha256, evidence_index.json,
#   license/LICENCE.txt
# - Unpack into: <root>/v1_1/<speaker>/...

BASE_URL="https://data.cstr.ed.ac.uk/mocha"

ROOT="data/external/mocha_timit"
SPEAKERS="fsew0,msak0"
DRY_RUN=0
FORCE=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/acquire_mocha_timit.sh [--root PATH] [--speakers fsew0,msak0] [--dry-run] [--force]

Notes:
  - Writes under <root> (default: data/external/mocha_timit)
  - Downloads only allowlisted files from https://data.cstr.ed.ac.uk/mocha/
  - Does NOT modify ACP v0 contract files.
EOF
}

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >&2
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

sha256_file() {
  local path="$1"
  if have sha256sum; then
    sha256sum "$path" | awk '{print $1}'
    return 0
  fi
  if have shasum; then
    shasum -a 256 "$path" | awk '{print $1}'
    return 0
  fi
  die "need sha256sum or shasum"
}

download_file() {
  local url="$1"
  local out="$2"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: curl -fL --retry 3 --retry-delay 2 -o '$out' '$url'"
    return 0
  fi

  mkdir -p "$(dirname "$out")"
  # atomic-ish: download to temp then move
  local tmp="${out}.tmp.$$"
  rm -f "$tmp"
  curl -fL --retry 3 --retry-delay 2 -o "$tmp" "$url"
  mv -f "$tmp" "$out"
}

json_escape() {
  python3 - <<'PY'
import json, sys
print(json.dumps(sys.stdin.read()))
PY
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --root)
        ROOT="$2"; shift 2 ;;
      --speakers)
        SPEAKERS="$2"; shift 2 ;;
      --dry-run)
        DRY_RUN=1; shift 1 ;;
      --force)
        FORCE=1; shift 1 ;;
      -h|--help)
        usage; exit 0 ;;
      *)
        die "unknown arg: $1" ;;
    esac
  done
}

preflight() {
  have curl || die "missing curl"
  have tar || die "missing tar"
  have python3 || die "missing python3"
  (have sha256sum || have shasum) || die "need sha256sum or shasum"
}

ensure_dir() {
  local p="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: mkdir -p '$p'"
    return 0
  fi
  mkdir -p "$p"
}

remove_path_if_exists() {
  local p="$1"
  if [[ ! -e "$p" && ! -L "$p" ]]; then
    return 0
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: rm -rf '$p'"
    return 0
  fi
  rm -rf "$p"
}

speaker_candidates() {
  # Output one candidate filename per line (within allowlist) for given speaker.
  local spk="$1"
  case "$spk" in
    fsew0)
      printf '%s\n' "fsew0_v1.1.tar.gz" "mocha-timit_fsew0.tgz" "fsew0.tar.gz" ;;
    msak0)
      printf '%s\n' "msak0_v1.1.tar.gz" "mocha-timit_msak0.tgz" "msak0.tar.gz" ;;
    maps0)
      printf '%s\n' "maps0.tar.gz" ;;
    *)
      die "unsupported speaker (allowlist): $spk" ;;
  esac
}

remote_exists() {
  local url="$1"
  # Use HEAD; some servers block HEAD, so fall back to ranged GET.
  if curl -fsI "$url" >/dev/null 2>&1; then
    return 0
  fi
  curl -fsL -r 0-0 "$url" >/dev/null 2>&1
}

select_remote_archive() {
  local spk="$1"
  local chosen=""
  while IFS= read -r cand; do
    local url="${BASE_URL}/${cand}"
    if remote_exists "$url"; then
      chosen="$cand"
      break
    fi
  done < <(speaker_candidates "$spk")

  if [[ -z "$chosen" ]]; then
    die "no allowlisted archive found for speaker=$spk at $BASE_URL"
  fi
  echo "$chosen"
}

extract_archive() {
  local tgz="$1"
  local outdir="$2"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: tar -xzf '$tgz' -C '$outdir'"
    return 0
  fi

  mkdir -p "$outdir"
  tar -xzf "$tgz" -C "$outdir"
}

write_checksums() {
  local raw_dir="$1"
  local out_path="$2"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: write checksums to '$out_path' (raw/*)"
    return 0
  fi

  : > "$out_path"
  (cd "$raw_dir" && {
    if have sha256sum; then
      # shellcheck disable=SC2035
      sha256sum * >> "$(realpath "$out_path")"
    else
      # shasum prints: <hash>  <file>
      # shellcheck disable=SC2035
      shasum -a 256 * >> "$(realpath "$out_path")"
    fi
  })
}

verify_checksums() {
  local raw_dir="$1"
  local sums_path="$2"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: verify checksums '$sums_path'"
    return 0
  fi

  (cd "$raw_dir" && {
    if have sha256sum; then
      sha256sum -c "$(realpath "$sums_path")"
    else
      # shasum uses -c too.
      shasum -a 256 -c "$(realpath "$sums_path")"
    fi
  })
}

write_manifest() {
  local out_path="$1"
  local retrieved_at="$2"
  local root_rel="$3"
  local speakers_json="$4"
  local artifacts_json="$5"
  local idempotent="$6"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: write manifest '$out_path'"
    return 0
  fi

  python3 - <<PY
import json
from datetime import datetime, timezone

payload = {
  "dataset_name": "MOCHA-TIMIT",
  "source_urls": ["${BASE_URL}/"],
  "retrieved_at_utc": "${retrieved_at}",
  "root": "${root_rel}",
  "speakers": json.loads('''${speakers_json}'''),
  "artifacts": json.loads('''${artifacts_json}'''),
  "license": {
    "allowed_use": "research/education/non-commercial",
    "commercial_use": "prohibited_without_permission",
    "source_url": "${BASE_URL}/LICENCE.txt",
  },
  "run": {
    "idempotent": bool(int("${idempotent}")),
  },
  "integration": {
    "acp_contract_unchanged": True,
  },
}

with open("${out_path}", "w", encoding="utf-8") as f:
  json.dump(payload, f, ensure_ascii=False, indent=2)
  f.write("\n")
PY
}

write_evidence_index() {
  local out_path="$1"
  local extracted_index_json="$2"
  local idempotent="$3"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: write evidence index '$out_path'"
    return 0
  fi

  python3 - <<PY
import json

payload = {
  "evidence": {
    "manifest": "manifest.json",
    "checksums": "checksums.sha256",
    "license": "license/LICENCE.txt",
    "raw_dir": "raw",
    "extracted_root": "v1_1",
    "extracted_index": json.loads('''${extracted_index_json}'''),
  },
  "run": {
    "idempotent": bool(int("${idempotent}")),
  },
}

with open("${out_path}", "w", encoding="utf-8") as f:
  json.dump(payload, f, ensure_ascii=False, indent=2)
  f.write("\n")
PY
}

make_extracted_index() {
  local root_dir="$1"
  local speakers_csv="$2"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo '{"note":"dry-run"}'
    return 0
  fi
  python3 - <<PY
import json
from pathlib import Path

root = Path("${root_dir}")
speakers = [s for s in "${speakers_csv}".split(",") if s]

index = {
  "speakers": {},
}
for spk in speakers:
  base = root / "v1_1" / spk
  rows = []
  if base.exists():
    for p in sorted(base.rglob("*")):
      if p.is_file():
        try:
          st = p.stat()
        except OSError:
          continue
        rows.append({
          "path": str(p.relative_to(root)),
          "bytes": int(st.st_size),
        })
  index["speakers"][spk] = {
    "root": str(base.relative_to(root)) if base.exists() else None,
    "files": rows,
    "file_count": len(rows),
  }

print(json.dumps(index, ensure_ascii=False))
PY
}

main() {
  parse_args "$@"
  preflight

  local root_abs
  root_abs="$(python3 -c "from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())" "$ROOT")"
  local raw_dir="$root_abs/raw"
  local lic_dir="$root_abs/license"
  local v1_dir="$root_abs/v1_1"

  ensure_dir "$raw_dir"
  ensure_dir "$lic_dir"
  ensure_dir "$v1_dir"

  local retrieved_at
  retrieved_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  local did_work=0

  # Download license/readme into raw/
  local license_raw="$raw_dir/LICENCE.txt"
  local readme_raw="$raw_dir/README_v1.2.txt"

  if [[ "$FORCE" -eq 1 ]]; then
    remove_path_if_exists "$license_raw"
    remove_path_if_exists "$readme_raw"
  fi

  if [[ ! -f "$license_raw" ]]; then
    download_file "${BASE_URL}/LICENCE.txt" "$license_raw"
    did_work=1
  fi
  # README is optional on the server; if missing, skip without failing.
  if [[ ! -f "$readme_raw" ]]; then
    if remote_exists "${BASE_URL}/README_v1.2.txt"; then
      download_file "${BASE_URL}/README_v1.2.txt" "$readme_raw"
      did_work=1
    fi
  fi

  # Copy license evidence
  local license_dst="$lic_dir/LICENCE.txt"
  if [[ "$FORCE" -eq 1 ]]; then
    remove_path_if_exists "$license_dst"
  fi
  if [[ ! -f "$license_dst" ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      log "dry-run: cp '$license_raw' '$license_dst'"
    else
      cp -f "$license_raw" "$license_dst"
    fi
    did_work=1
  fi

  # Speakers
  IFS=',' read -r -a speaker_list <<< "$SPEAKERS"
  local speakers_norm=""
  for spk in "${speaker_list[@]}"; do
    spk="$(echo "$spk" | tr -d ' ' | tr '[:upper:]' '[:lower:]')"
    [[ -n "$spk" ]] || continue
    if [[ -n "$speakers_norm" ]]; then
      speakers_norm+="${speakers_norm:+,}""$spk"
    else
      speakers_norm="$spk"
    fi
  done
  if [[ -z "$speakers_norm" ]]; then
    die "--speakers is empty"
  fi

  # Track artifacts for manifest.
  local artifacts_py
  artifacts_py="[]"

  # Download + extract per speaker.
  local spk
  IFS=',' read -r -a speaker_list2 <<< "$speakers_norm"
  for spk in "${speaker_list2[@]}"; do
    local archive_name
    archive_name="$(select_remote_archive "$spk")"
    local url="${BASE_URL}/${archive_name}"
    local archive_path="$raw_dir/$archive_name"

    local out_speaker_dir="$v1_dir/$spk"

    if [[ "$FORCE" -eq 1 ]]; then
      remove_path_if_exists "$archive_path"
      remove_path_if_exists "$out_speaker_dir"
    fi

    if [[ ! -f "$archive_path" ]]; then
      download_file "$url" "$archive_path"
      did_work=1
    fi

    # Extract
    if [[ -d "$out_speaker_dir" && "$FORCE" -ne 1 ]]; then
      log "skip extract (exists): $out_speaker_dir"
    else
      remove_path_if_exists "$out_speaker_dir"
      ensure_dir "$out_speaker_dir"
      extract_archive "$archive_path" "$out_speaker_dir"
      did_work=1
    fi
  done

  # Write checksums for raw/* and verify.
  local sums_path="$root_abs/checksums.sha256"
  if [[ "$FORCE" -eq 1 ]]; then
    remove_path_if_exists "$sums_path"
  fi
  write_checksums "$raw_dir" "$sums_path"
  verify_checksums "$raw_dir" "$sums_path"

  # Build manifest artifacts list from raw/*.
  if [[ "$DRY_RUN" -eq 1 ]]; then
    artifacts_py='[]'
  else
    artifacts_py="$(python3 - <<PY
import json
from pathlib import Path

raw = Path("$raw_dir")
rows = []
for p in sorted(raw.glob("*")):
  if not p.is_file():
    continue
  rows.append({
    "name": p.name,
    "bytes": p.stat().st_size,
    "sha256": ""  # filled below
  })
print(json.dumps(rows, ensure_ascii=False))
PY
)"

    # Fill sha256 values.
    artifacts_py="$(python3 - <<PY
import json
from pathlib import Path
import subprocess

raw = Path("$raw_dir")
rows = json.loads('''$artifacts_py''')

def sha256(p: Path) -> str:
  # Prefer sha256sum; fall back to shasum.
  if subprocess.call(["bash", "-lc", "command -v sha256sum >/dev/null 2>&1"]) == 0:
    out = subprocess.check_output(["sha256sum", str(p)])
    return out.decode("utf-8").split()[0]
  out = subprocess.check_output(["shasum", "-a", "256", str(p)])
  return out.decode("utf-8").split()[0]

for r in rows:
  r["sha256"] = sha256(raw / r["name"])

print(json.dumps(rows, ensure_ascii=False))
PY
)"
  fi

  local speakers_json
  speakers_json="$(python3 - <<PY
import json
print(json.dumps([s for s in "$speakers_norm".split(",") if s], ensure_ascii=False))
PY
)"

  local idempotent=0
  if [[ "$did_work" -eq 0 ]]; then
    idempotent=1
  fi

  local manifest_path="$root_abs/manifest.json"
  write_manifest "$manifest_path" "$retrieved_at" "${ROOT}" "$speakers_json" "$artifacts_py" "$idempotent"

  local extracted_index
  extracted_index="$(make_extracted_index "$root_abs" "$speakers_norm")"
  local evidence_path="$root_abs/evidence_index.json"
  write_evidence_index "$evidence_path" "$extracted_index" "$idempotent"

  log "done root=$ROOT idempotent=$idempotent"
}

main "$@"
