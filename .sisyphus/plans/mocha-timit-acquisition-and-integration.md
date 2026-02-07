# Mocha-TIMIT Acquisition And Integration (ACP v0-safe)

## TL;DR

> **Quick Summary**: Acquire Mocha-TIMIT from CSTR with an idempotent script, capture license/compliance evidence, unpack deterministically, and add sidecar integration metadata without changing ACP v0 training/inference contracts.
>
> **Deliverables**:
> - `scripts/acquire_mocha_timit.sh` (download + verify + unpack + manifest)
> - `data/external/mocha_timit/manifest.json`
> - `data/external/mocha_timit/checksums.txt`
> - `data/external/mocha_timit/license/LICENCE.txt`
> - `docs/mocha_timit_integration.md` (how to use sidecar data)
>
> **Estimated Effort**: Short
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 -> Task 2 -> Task 4

---

## Context

### Original Request
User has already downloaded `PHOIBLE`, `CLTS`, `WikiHan`, `WikiPron`; asks for Mocha-TIMIT plan + download script + instructions.

### Interview Summary
**Key Discussions**:
- Mocha-TIMIT is available from CSTR and suitable for academic/non-commercial use.
- This phase should not break ACP-first PGDN v0 contracts.
- User wants immediately actionable acquisition steps.

**Research Findings**:
- Data source page: `https://www.cstr.ed.ac.uk/research/projects/artic/mocha.html`
- Download index: `https://data.cstr.ed.ac.uk/mocha/`
- Local contracts to preserve:
  - `src/pgdn/data/build_acp.py`
  - `src/pgdn/confidence.py`
  - `src/pgdn/infer.py`
  - `data/anchors/acp_anchors.jsonl`

### Metis Review
**Identified Gaps (addressed)**:
- Lock artifact allowlist to avoid accidental extra downloads.
- Persist license as evidence + machine-readable compliance flags.
- Enforce idempotency and deterministic folder layout.
- Keep scope sidecar-only (no ACP schema/confidence changes in this task).

---

## Work Objectives

### Core Objective
Create a reproducible Mocha-TIMIT acquisition workflow with compliance evidence and integration notes, while keeping ACP v0 behavior unchanged.

### Concrete Deliverables
- Script to fetch fixed files (`fsew0_v1.1.tar.gz`, `msak0_v1.1.tar.gz`, `LICENCE.txt`, `README_v1.2.txt` when available)
- Deterministic extracted tree under `data/external/mocha_timit/v1_1/`
- Machine-readable manifest describing source URLs, hashes, timestamps, and usage restrictions
- Integration note describing optional future adapter path into ACP anchors/constraints

### Definition of Done
- [ ] Script runs successfully and creates all evidence artifacts.
- [ ] Script is idempotent (second run does not redownload/rewrite unchanged assets).
- [ ] Manifest explicitly records non-commercial caveat.
- [ ] No changes to ACP target schema or confidence semantics.

### Must Have
- Idempotent acquisition flow
- Evidence-first outputs (checksums, manifest, license copy)
- Strict source allowlist

### Must NOT Have (Guardrails)
- No automatic mutation of `data/targets/acp_targets.jsonl`
- No modification to ranking/confidence bucket logic in `src/pgdn/confidence.py`
- No pretending Mocha license is commercially permissive
- No download of unspecified archives by default

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> All checks below must be executable by agent commands only.

### Test Decision
- **Infrastructure exists**: YES (Python scripts/CLI flow present)
- **Automated tests**: None for this task (script verification via command assertions)
- **Framework**: shell command assertions + JSON assertions

### Agent-Executed QA Scenarios (MANDATORY)

Scenario: First acquisition populates expected artifacts
  Tool: Bash
  Preconditions: Network access to CSTR endpoints
  Steps:
    1. Run `bash scripts/acquire_mocha_timit.sh --root data/external/mocha_timit --speakers fsew0,msak0`
    2. Assert files exist:
       - `data/external/mocha_timit/manifest.json`
       - `data/external/mocha_timit/checksums.txt`
       - `data/external/mocha_timit/license/LICENCE.txt`
    3. Assert extraction paths exist:
       - `data/external/mocha_timit/v1_1/fsew0/`
       - `data/external/mocha_timit/v1_1/msak0/`
  Expected Result: All required artifacts exist and are readable
  Failure Indicators: Missing manifest/license/checksums or missing extracted speaker dirs
  Evidence: `data/external/mocha_timit/manifest.json`

Scenario: Idempotent second run
  Tool: Bash
  Preconditions: Scenario 1 completed
  Steps:
    1. Re-run same command
    2. Compare manifest `run.idempotent` flag is `true`
    3. Validate `sha256sum -c data/external/mocha_timit/checksums.txt` returns all `OK`
  Expected Result: No duplicate extraction; checksums remain valid
  Failure Indicators: hash mismatch, duplicate directories, changed artifact set
  Evidence: `data/external/mocha_timit/checksums.txt`

Scenario: Compliance metadata present
  Tool: Bash
  Preconditions: Scenario 1 completed
  Steps:
    1. Read `manifest.json` fields:
       - `license.allowed_use`
       - `license.commercial_use`
       - `license.source_url`
    2. Assert `commercial_use` indicates restriction / permission required
  Expected Result: Compliance caveat is explicit and machine-readable
  Failure Indicators: missing or permissive commercial flag
  Evidence: `data/external/mocha_timit/manifest.json`

Scenario: ACP contract unchanged guard
  Tool: Bash
  Preconditions: Scenario 1 completed
  Steps:
    1. Assert script does not write to:
       - `data/targets/acp_targets.jsonl`
       - `src/pgdn/confidence.py`
       - `src/pgdn/infer.py`
    2. Confirm manifest flag `integration.acp_contract_unchanged=true`
  Expected Result: Sidecar-only integration
  Failure Indicators: any ACP contract file touched
  Evidence: `data/external/mocha_timit/manifest.json`

---

## Execution Strategy

### Parallel Execution Waves

Wave 1 (Start Immediately):
- Task 1: Source and license lock
- Task 3: Draft integration note skeleton

Wave 2 (After Wave 1):
- Task 2: Implement acquisition script and run verification
- Task 4: Finalize docs + evidence index

Critical Path: Task 1 -> Task 2 -> Task 4

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2 | 3 |
| 2 | 1 | 4 | None |
| 3 | None | 4 | 1 |
| 4 | 2, 3 | None | None |

---

## TODOs

- [ ] 1. Lock source artifacts and compliance policy

  **What to do**:
  - Define fixed allowlist: `fsew0_v1.1.tar.gz`, `msak0_v1.1.tar.gz`, `LICENCE.txt`, `README_v1.2.txt`.
  - Record canonical base URL: `https://data.cstr.ed.ac.uk/mocha/`.
  - Define compliance metadata keys for manifest.

  **Must NOT do**:
  - Do not auto-include extra tarballs not in allowlist.
  - Do not mark commercial usage as allowed.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: single-scope acquisition policy task.
  - **Skills**: [`git-master`]
    - `git-master`: useful for minimal, atomic doc/script edits and traceability.
  - **Skills Evaluated but Omitted**:
    - `playwright`: no browser interaction needed.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 3)
  - **Blocks**: Task 2
  - **Blocked By**: None

  **References**:
  - `src/pgdn/data/build_acp.py:193` - deterministic manifest style to mirror.
  - `src/pgdn/confidence.py:5` - confidence semantics to keep untouched.
  - `src/pgdn/infer.py:95` - calibration note semantics to keep untouched.
  - `data/anchors/acp_anchors.jsonl` - optional future integration channel.
  - `https://www.cstr.ed.ac.uk/research/projects/artic/mocha.html` - official corpus description.
  - `https://data.cstr.ed.ac.uk/mocha/LICENCE.txt` - usage restriction source.

  **Acceptance Criteria**:
  - [ ] Allowlist documented in `docs/mocha_timit_integration.md`.
  - [ ] Manifest schema includes `license.allowed_use`, `license.commercial_use`, `license.source_url`.

  **Commit**: NO

- [ ] 2. Create acquisition script (idempotent, evidence-first)

  **What to do**:
  - Create `scripts/acquire_mocha_timit.sh` with flags:
    - `--root` (default `data/external/mocha_timit`)
    - `--speakers` (default `fsew0,msak0`)
    - `--dry-run` (optional)
  - Download allowlisted artifacts only.
  - Save checksums to `checksums.txt`.
  - Extract to `v1_1/{speaker}/` deterministically.
  - Write `manifest.json` with URL, hashes, retrieval time, compliance, idempotent run status.

  **Must NOT do**:
  - Do not overwrite extracted directories unless explicit `--force`.
  - Do not modify ACP training/infer files.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: bounded scripting task with deterministic outputs.
  - **Skills**: [`git-master`]
    - `git-master`: clean atomic change management.
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: not relevant.

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 4
  - **Blocked By**: Task 1

  **References**:
  - `scripts/cleanup_runs.py` - existing style for deterministic manifest writing patterns.
  - `src/pgdn/data/io.py` - JSON write/read style conventions.
  - `src/pgdn/data/build_acp.py:206` - output location conventions under `data/`.

  **Acceptance Criteria**:
  - [ ] Running script creates manifest/checksums/license copy/extracted dirs.
  - [ ] Second run is idempotent and hash-verified.
  - [ ] `manifest.json` contains `integration.acp_contract_unchanged=true`.

  **Commit**: NO

- [ ] 3. Draft integration note (sidecar now, adapters later)

  **What to do**:
  - Create `docs/mocha_timit_integration.md` documenting:
    - acquisition command,
    - folder layout,
    - non-commercial caveat,
    - optional future adapter path to `data/anchors/` and physiological constraints.

  **Must NOT do**:
  - Do not promise full ACP conversion in this phase.

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: documentation-focused deliverable.
  - **Skills**: [`git-master`]
    - `git-master`: maintain concise, consistent docs updates.
  - **Skills Evaluated but Omitted**:
    - `playwright`: not relevant.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Task 4
  - **Blocked By**: None

  **References**:
  - `PGDN_IMPLEMENTATION_PLAN.md:100` - articulatory feature direction and constraints context.
  - `data/splits/manifest.json` - leakage-rules wording style to mirror.

  **Acceptance Criteria**:
  - [ ] Doc contains exact one-line command for download.
  - [ ] Doc includes explicit license caveat and commercialization note.
  - [ ] Doc includes future adapter TODOs as non-committed roadmap.

  **Commit**: NO

- [ ] 4. Run automated verification and publish evidence index

  **What to do**:
  - Execute QA command suite:
    - acquisition run,
    - idempotent rerun,
    - checksum verification,
    - manifest key assertions.
  - Produce `data/external/mocha_timit/evidence_index.json` summarizing checks and outcomes.

  **Must NOT do**:
  - Do not use manual verification language.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: short command-based validation workflow.
  - **Skills**: [`git-master`]
    - `git-master`: keep evidence updates precise.
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: not relevant.

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (final)
  - **Blocks**: None
  - **Blocked By**: Tasks 2 and 3

  **References**:
  - `runs/retention_manifests/cleanup_20260206T143146Z.json` - manifest/evidence precedent in repo.

  **Acceptance Criteria**:
  - [ ] `evidence_index.json` exists and references all generated artifacts.
  - [ ] All verification commands report pass.
  - [ ] No ACP contract files were touched.

  **Commit**: NO

---

## Script Spec (for executor)

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="data/external/mocha_timit"
SPEAKERS="fsew0,msak0"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --speakers) SPEAKERS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

BASE_URL="https://data.cstr.ed.ac.uk/mocha"
mkdir -p "$ROOT/raw" "$ROOT/license" "$ROOT/v1_1"

ALLOW=("LICENCE.txt" "README_v1.2.txt")
IFS=',' read -r -a SPK <<< "$SPEAKERS"
for s in "${SPK[@]}"; do
  ALLOW+=("mocha-timit_${s}0.tgz" "${s}0_v1.1.tar.gz")
done

download() {
  local name="$1"
  local target="$ROOT/raw/$name"
  if [[ -f "$target" ]]; then
    echo "skip existing: $name"
    return
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "dry-run download: $BASE_URL/$name"
    return
  fi
  curl -fL "$BASE_URL/$name" -o "$target"
}

for a in "${ALLOW[@]}"; do download "$a"; done

if [[ "$DRY_RUN" -eq 0 ]]; then
  cp -f "$ROOT/raw/LICENCE.txt" "$ROOT/license/LICENCE.txt" || true
  (cd "$ROOT/raw" && shasum -a 256 * > "$ROOT/checksums.txt")
fi

echo "TODO: executor extracts speaker archives into $ROOT/v1_1/{speaker}/ and writes manifest.json"
```

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 2+3+4 | `chore(data): add mocha-timit acquisition workflow` | script + docs + manifests | QA command suite |

---

## Success Criteria

### Verification Commands
```bash
bash scripts/acquire_mocha_timit.sh --root data/external/mocha_timit --speakers fsew0,msak0
sha256sum -c data/external/mocha_timit/checksums.txt
python3 - <<'PY'
import json
m = json.load(open('data/external/mocha_timit/manifest.json'))
assert m['integration']['acp_contract_unchanged'] is True
assert 'commercial' in m['license']['commercial_use'].lower() or 'permission' in m['license']['commercial_use'].lower()
print('ok')
PY
```

### Final Checklist
- [ ] Script exists and is executable
- [ ] Manifest, checksum, and license artifacts exist
- [ ] Idempotent rerun passes
- [ ] ACP v0 contracts unchanged
