# Mocha-TIMIT Integration (Phase 1)

This document covers **Phase 1 only**:

- Download + verify + unpack Mocha-TIMIT into a stable local layout
- Produce audit artifacts (manifest/checksums/license/evidence index)
- Register it via the unified dataset hub (`data/external/links/`)

Non-goals (Phase 1):

- Do NOT merge Mocha-TIMIT into ACP targets
- Do NOT change ACP v0 contract files:
  - `data/targets/acp_targets.jsonl`
  - `src/pgdn/confidence.py`
  - `src/pgdn/infer.py`

## Legal / Safety

Mocha-TIMIT is typically **research/education / non-commercial**.

- Keep the upstream `LICENCE.txt` as evidence under `data/external/mocha_timit/license/`.
- Do NOT commit the dataset to git.

## One-command acquisition

Dry-run (prints intended actions; does not download/write):

```bash
bash scripts/acquire_mocha_timit.sh --dry-run --root data/external/mocha_timit --speakers fsew0,msak0
```

Real run:

```bash
bash scripts/acquire_mocha_timit.sh --root data/external/mocha_timit --speakers fsew0,msak0
```

Force re-download/re-extract (dangerous; deletes the speaker directories under `v1_1/`):

```bash
bash scripts/acquire_mocha_timit.sh --force --root data/external/mocha_timit --speakers fsew0,msak0
```

## Expected layout

After a successful run:

```text
data/external/mocha_timit/
  raw/
    LICENCE.txt
    README_v1.2.txt               (if present upstream)
    fsew0_v1.1.tar.gz             (or allowlisted equivalent)
    msak0_v1.1.tar.gz             (or allowlisted equivalent)
  license/
    LICENCE.txt
  v1_1/
    fsew0/...
    msak0/...
  checksums.sha256
  manifest.json
  evidence_index.json
```

## Verify (agent-executable)

```bash
test -f data/external/mocha_timit/manifest.json
test -f data/external/mocha_timit/checksums.sha256
test -f data/external/mocha_timit/license/LICENCE.txt
```

Checksum verification:

```bash
sha256sum -c data/external/mocha_timit/checksums.sha256 \
  --ignore-missing 2>/dev/null || true
```

(If `sha256sum` is not available, the script uses `shasum -a 256` internally.)

Manifest checks:

```bash
python3 - <<'PY'
import json
m = json.load(open('data/external/mocha_timit/manifest.json','r',encoding='utf-8'))
assert m['dataset_name'] == 'MOCHA-TIMIT'
assert m['integration']['acp_contract_unchanged'] is True
assert 'permission' in m['license']['commercial_use'].lower() or 'prohibited' in m['license']['commercial_use'].lower()
print('manifest:ok')
PY
```

Idempotency check (run twice):

```bash
bash scripts/acquire_mocha_timit.sh --root data/external/mocha_timit --speakers fsew0,msak0
python3 - <<'PY'
import json
m = json.load(open('data/external/mocha_timit/manifest.json','r',encoding='utf-8'))
assert m['run']['idempotent'] is True
print('idempotent:ok')
PY
```

## Register in unified dataset hub

```bash
python3 scripts/organize_datasets.py
test -L data/external/links/mocha_timit
```

## Phase 2 (optional) activation gate

Do **not** implement Phase 2 unless all are true:

- Phase 1 acquisition + verification passes
- Explicit approval to build a Mocha sidecar adapter
- ACP v0 contract remains unchanged (no changes to the protected files)
