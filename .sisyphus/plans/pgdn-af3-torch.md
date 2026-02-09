# PGDN (AF3-style) PyTorch Route on Linux

Goal: continue the project end-to-end on Linux with a PyTorch-first training/inference pipeline (GPU if available), while keeping the repo public-safe (no datasets/artifacts committed) and preserving upstream submodules/licenses.

This plan follows the user’s chosen direction **(2) AF3/PGDN route (Pairformer + Diffusion + constraints)**.

---

## Current State (Observed)

- Unified dataset hub exists: `scripts/organize_datasets.py` writes `data/external/dataset_registry.json` and maintains symlinks under `data/external/links/`.
- Mocha-TIMIT Phase 1 acquisition exists and is expected to download into `data/external/mocha_timit/` with audit artifacts (manifest/checksums/evidence/license).
- CLTS detection is supported via `scripts/organize_datasets.py --clts-path ...` and/or env var `CLTS_PATH`.
- Guardrails from PRD must be respected:
  - MUST NOT modify: `data/targets/acp_targets.jsonl`, `src/pgdn/confidence.py`, `src/pgdn/infer.py`.
- Public-safe policy: do NOT commit any external datasets, large binaries, or generated artifacts.

---

## Work Objectives

### Core Objective

Implement a **GPU-capable PyTorch pipeline** that mirrors the repo’s intended PGDN-style architecture (Embedder → Pairformer → Diffusion → constraints), and runs end-to-end reproducibly on Linux.

### Deliverables

- Dataset hub remains the only supported external dataset entrypoint: `data/external/links/<key>`.
- Mocha-TIMIT Phase 1 acquisition verified and documented.
- CLTS available under `data/external/links/clts` when present.
- PyTorch PGDN modules + runnable training/inference entrypoints:
  - Training: 1 epoch fast run + proper checkpoint
  - Inference: sample multiple candidates per record
  - Optional: ranking using existing `src/pgdn/confidence.py` without modifications
- Linux quickstart docs updated (commands, expected outputs).

---

## Scope Boundaries (Guardrails)

IN:
- Add new code under `src/` (new namespaces are OK).
- Add new scripts under `scripts/`.
- Add/extend docs under `docs/`.

OUT:
- Do not alter upstream code inside submodules.
- Do not commit datasets under `data/external/`.
- Do not change the protected files:
  - `data/targets/acp_targets.jsonl`
  - `src/pgdn/confidence.py`
  - `src/pgdn/infer.py`

---

## Verification Strategy

All verification is agent-executed (no human steps). Evidence should be produced as command outputs, small JSON summaries, and/or small logs under `runs/`.

Required checks:

1) Dataset hub:
```bash
python3 scripts/organize_datasets.py --clts-path data/external/clts
python3 scripts/print_dataset_paths.py
```

2) Mocha acquisition (idempotent + checksum):
```bash
bash scripts/acquire_mocha_timit.sh --root data/external/mocha_timit --speakers fsew0,msak0
sha256sum -c data/external/mocha_timit/checksums.sha256
bash scripts/acquire_mocha_timit.sh --root data/external/mocha_timit --speakers fsew0,msak0
python3 - <<'PY'
import json
m=json.load(open('data/external/mocha_timit/manifest.json','r',encoding='utf-8'))
assert m['run']['idempotent'] is True
print('idempotent:ok')
PY
```

3) PyTorch PGDN smoke:
```bash
PYTHONPATH=src python3 -m pgdn_torch.train.pgdnv0_train --synthetic --epochs 1
PYTHONPATH=src python3 -m pgdn_torch.infer.pgdnv0_infer --synthetic --checkpoint runs/pgdn_torch_v0/checkpoint_none.pt
```

4) Static sanity:
```bash
python3 -m compileall -q src scripts
```

5) Public-safe git check:
```bash
git status --porcelain=v1
git diff --stat
```
Confirm no dataset directories become tracked.

---

## Execution Strategy (Waves)

Wave 1 (Data + hub correctness):
- Ensure Mocha-TIMIT acquisition is correct and idempotent.
- Ensure CLTS path detection works and hub link exists.

Wave 2 (PyTorch PGDN core):
- Implement `PGDNTorchV0` model components (embedder/pairformer/diffusion/constraint loss).
- Implement dataset layer:
  - Synthetic dataset for fast smoke
  - Optional ACP JSONL dataset loader for real training runs

Wave 3 (Train/infer entrypoints + ranking):
- Training CLI writes checkpoint + jsonl metrics.
- Inference CLI samples N candidates and (optionally) computes ranking using `pgdn.confidence`.

Wave 4 (Docs + guardrails):
- Document “links-only” rule.
- Add helper scripts (`print_dataset_paths.py`, optional `check_links_only.py`) to keep code honest.

---

## TODOs

- [ ] 1) Confirm dataset hub mapping
  - Run: `python3 scripts/organize_datasets.py --clts-path data/external/clts`
  - Run: `python3 scripts/print_dataset_paths.py`
  - Acceptance: all expected datasets show correct `links/*` targets.

- [ ] 2) Mocha-TIMIT acquisition + audit artifacts
  - Run acquisition (real), then `sha256sum -c`.
  - Acceptance: `manifest.json`, `checksums.sha256`, `evidence_index.json`, `license/LICENCE.txt` exist; second run sets `run.idempotent=true`.

- [ ] 3) PyTorch PGDN model implementation
  - Add modules under `src/pgdn_torch/pgdnv0/`.
  - Acceptance: importable with `PYTHONPATH=src python3 -c 'import pgdn_torch.pgdnv0'`.

- [ ] 4) PyTorch training CLI
  - Command: `PYTHONPATH=src python3 -m pgdn_torch.train.pgdnv0_train --synthetic --epochs 1`.
  - Acceptance: writes `runs/pgdn_torch_v0/checkpoint_*.pt` and `runs/pgdn_torch_v0/train_metrics.jsonl`.

- [ ] 5) PyTorch inference CLI
  - Command: `PYTHONPATH=src python3 -m pgdn_torch.infer.pgdnv0_infer --synthetic --checkpoint runs/pgdn_torch_v0/checkpoint_none.pt`.
  - Acceptance: writes sample tensors under `runs/pgdn_torch_v0_infer/samples/` and `infer_meta.json`; optional `ranking_scores.csv` if confidence module import works.

- [ ] 6) Mocha-TIMIT Phase 2 prep (non-invasive)
  - Provide an indexer to list available modalities (wav/ema/lab) without feature engineering.
  - Acceptance: index script prints counts and exits 0.

---

## Notes on Paper Alignment

The files under `paper/` propose a full AF3/PGDN-style research architecture (heterogeneous graph + Pairformer + diffusion + physiological constraints). This plan implements the **engineering substrate**:

- reproducible datasets + audit + hub links
- GPU-capable torch training/inference scaffolding

It does not claim to fully reproduce the research results yet; the next iteration after this plan is to design the **actual linguistic graph construction and EMA-driven constraint modeling**.
