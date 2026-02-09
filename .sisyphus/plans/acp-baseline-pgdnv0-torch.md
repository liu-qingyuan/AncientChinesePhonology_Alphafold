# ACP Baseline: Torch PGDNv0 End-to-End Smoke (Build -> Train -> Infer)

## TL;DR

> **Quick Summary**: Use the existing ACP CSV already in this repo to generate `acp_targets.jsonl`, then run the torch PGDNv0 train+infer CLIs on that real data as a reproducible baseline.
>
> **Deliverables**:
> - `data/targets/acp_targets.jsonl` + `data/splits/manifest.json` (generated via `src/pgdn/data/build_acp.py`)
> - `runs/acp_baseline_pgdnv0_torch/` (train metrics + checkpoint)
> - `runs/acp_baseline_pgdnv0_torch_infer/` (sampled vectors + infer metadata + optional ranking)
>
> **Estimated Effort**: Short
> **Parallel Execution**: NO (sequential pipeline)
> **Critical Path**: Build ACP artifacts -> Train torch -> Infer torch

---

## Context

### Original Request
- User chose milestone **A**: build a runnable small version first (end-to-end, real data), then iterate toward the architecture described in `paper/西夏语预测模型更新方案.txt`.

### Why ACP Baseline
- This repo already contains an ACP-like CSV at `Ancient-Chinese-Phonology/dataset/ancient_chinese_phonology.csv`.
- This repo already contains a builder that turns that CSV into a stable training artifact contract:
  - `src/pgdn/data/build_acp.py` -> `data/targets/acp_targets.jsonl` and `data/splits/manifest.json`
- Torch PGDNv0 already supports consuming JSONL targets (32-dim `target_vector` + IMNC `mask`) via `src/pgdn_torch/pgdnv0/data.py:ACPJsonlDataset`.

### Metis Review (gaps addressed)
- **Pinned artifact paths**: Use the actual builder outputs (`data/targets/acp_targets.jsonl`, `data/splits/manifest.json`). Do not assume `split_manifest.json`.
- **Smoke definition**: 1 build + 1-epoch (or small step count) train + small-batch inference, all with fixed seeds.
- **Guardrail**: ensure outputs remain uncommitted (generated artifacts only).

---

## Work Objectives

### Core Objective
- Establish a reproducible **real-data** baseline run for torch PGDNv0: build -> train -> infer completes with machine-verifiable artifacts.

### Concrete Deliverables
- Generated artifacts:
  - `data/targets/acp_targets.jsonl`
  - `data/splits/manifest.json`
- Torch training run directory (example):
  - `runs/acp_baseline_pgdnv0_torch/`
- Torch inference run directory (example):
  - `runs/acp_baseline_pgdnv0_torch_infer/`

### Must NOT Have (guardrails)
- Do not modify protected files mentioned in prior PRD context:
  - `data/targets/acp_targets.jsonl` (treat as generated)
  - `src/pgdn/confidence.py`
  - `src/pgdn/infer.py`
- Do not edit submodules unless explicitly asked.
- Do not commit datasets or generated artifacts (anything under `data/` outputs, `runs/`).

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> Every acceptance criterion below is executable by an agent via shell/python.

### Test Decision
- **Infrastructure exists**: No formal unit test suite.
- **Automated tests**: None for now.
- **Primary verification**: agent-executed QA commands and schema assertions.

### Agent-Executed QA Scenarios (Bash)

Scenario: ACP baseline end-to-end (CPU or GPU)
  Tool: Bash
  Preconditions:
    - Repo checkout contains `Ancient-Chinese-Phonology/` directory.
    - Python can import local modules with `PYTHONPATH=src`.
  Steps:
    1. Assert input CSV exists and non-empty:
       - `test -s Ancient-Chinese-Phonology/dataset/ancient_chinese_phonology.csv`
    2. Build ACP artifacts:
       - Run the ACP builder via an in-memory module shim (works even if `src/pgdn/data/__init__.py` is missing):

         ```bash
         PYTHONPATH=src python3 -c "import sys, types, pathlib, importlib.util; \
         import pgdn; \
         data_dir = pathlib.Path(pgdn.__file__).parent / 'data'; \
         pkg = types.ModuleType('pgdn.data'); pkg.__path__ = [str(data_dir)]; sys.modules['pgdn.data'] = pkg; \
         spec = importlib.util.spec_from_file_location('pgdn.data.build_acp', data_dir / 'build_acp.py'); \
         mod = importlib.util.module_from_spec(spec); sys.modules['pgdn.data.build_acp'] = mod; spec.loader.exec_module(mod); \
         sys.argv = ['build_acp', '--csv', 'Ancient-Chinese-Phonology/dataset/ancient_chinese_phonology.csv', '--out', 'data', '--seed', '42']; \
         mod.main()"
         ```
    3. Assert generated artifacts exist and are non-empty:
       - `test -s data/targets/acp_targets.jsonl`
       - `test -s data/splits/manifest.json`
    4. Schema-check targets JSONL:
       - `PYTHONPATH=src python3 -c "import json; p='data/targets/acp_targets.jsonl'; n=0;\
         with open(p,'r',encoding='utf-8') as f:\
           \
           for line in f:\
             r=json.loads(line);\
             assert 'record_id' in r;\
             v=r['target_vector'];\
             assert isinstance(v,list) and len(v)==32;\
             m=r.get('mask',{});\
             assert isinstance(m,dict);\
             n+=1;\
             if n>=10: break;\
         assert n>0;\
         print('ok_targets_rows',n)"`
    5. Sanity-check the torch JSONL dataset loader can read one item:
       - `PYTHONPATH=src python3 -c "from pathlib import Path; from pgdn_torch.pgdnv0.data import ACPJsonlDataset; ds=ACPJsonlDataset(Path('data/targets/acp_targets.jsonl'), limit=1); x=ds[0]; print('ok_dataset', type(x).__name__)"`
    6. Train torch PGDNv0 on real targets (small + deterministic):
       - `PYTHONPATH=src python3 -m pgdn_torch.train.pgdnv0_train --targets data/targets/acp_targets.jsonl --limit 2048 --epochs 1 --batch-size 64 --num-workers 0 --seed 42 --out runs/acp_baseline_pgdnv0_torch`
    7. Assert checkpoint exists:
       - `test -s runs/acp_baseline_pgdnv0_torch/checkpoint_none.pt`
    8. Infer from checkpoint on a small limit:
       - `PYTHONPATH=src python3 -m pgdn_torch.infer.pgdnv0_infer --checkpoint runs/acp_baseline_pgdnv0_torch/checkpoint_none.pt --targets data/targets/acp_targets.jsonl --limit 32 --samples 4 --num-steps 20 --seed 7 --out runs/acp_baseline_pgdnv0_torch_infer`
    9. Assert inference artifacts exist:
       - `test -s runs/acp_baseline_pgdnv0_torch_infer/infer_meta.json`
       - `test -d runs/acp_baseline_pgdnv0_torch_infer/samples`
  Expected Result:
    - Build produces `acp_targets.jsonl` + `manifest.json`.
    - Train produces metrics JSONL + checkpoint.
    - Infer produces `infer_meta.json` and saved samples.
  Failure Indicators:
    - Missing CSV, missing generated JSONL, schema mismatch, training crash, inference crash.

---

## Execution Strategy

Sequential pipeline (no parallel waves):
1) Validate inputs -> 2) Build artifacts -> 3) Schema sanity -> 4) Train -> 5) Infer

---

## TODOs

- [ ] 1) Validate ACP input CSV is present

  References:
  - `Ancient-Chinese-Phonology/dataset/ancient_chinese_phonology.csv` - input CSV for ACP builder

  Acceptance Criteria:
  - `test -s Ancient-Chinese-Phonology/dataset/ancient_chinese_phonology.csv` exits 0

- [ ] 2) Build ACP artifacts (targets + splits)

  What to do:
  - Run `src/pgdn/data/build_acp.py` via module entrypoint to generate artifacts under `data/`.

  References:
  - `src/pgdn/data/build_acp.py` - outputs and schema (`target_vector` len=32, IMNC `mask`)
  - `scripts/build_from_acp.py` - wrapper (note: currently imports `pgdn.data.*`; may be unusable unless `src/pgdn/data/__init__.py` exists)

  Acceptance Criteria:
  - The shim command in QA Scenario Step #2 exits 0
  - `test -s data/targets/acp_targets.jsonl` exits 0
  - `test -s data/splits/manifest.json` exits 0

- [ ] 3) Sanity-check targets JSONL schema

  What to do:
  - Check first N rows have: `record_id`, `target_vector` list of length 32, `mask` is dict.

  References:
  - `src/pgdn/data/build_acp.py:build_graphs_and_targets()` - canonical schema
  - `src/pgdn_torch/pgdnv0/data.py:ACPJsonlDataset` - consumer expectations

  Acceptance Criteria:
  - The python one-liner in QA step #4 exits 0

- [ ] 4) Train torch PGDNv0 smoke on ACP targets

  What to do:
  - Train for `--epochs 1` with `--limit 2048` to bound runtime.
  - Write artifacts under `runs/acp_baseline_pgdnv0_torch/`.

  References:
  - `src/pgdn_torch/train/pgdnv0_train.py` - CLI contract
  - `src/pgdn_torch/pgdnv0/model.py` - Pairformer-ish trunk + DDPM diffusion + constraint loss

  Acceptance Criteria:
  - `PYTHONPATH=src python3 -m pgdn_torch.train.pgdnv0_train --targets data/targets/acp_targets.jsonl --limit 2048 --epochs 1 --batch-size 64 --num-workers 0 --seed 42 --out runs/acp_baseline_pgdnv0_torch` exits 0
  - `test -s runs/acp_baseline_pgdnv0_torch/checkpoint_none.pt` exits 0

- [ ] 5) Infer samples from the trained checkpoint

  What to do:
  - Run diffusion sampling with multiple samples; save `.pt` files per sample.
  - Allow optional ranking via importing `pgdn.confidence` (read-only usage).

  References:
  - `src/pgdn_torch/infer/pgdnv0_infer.py` - CLI contract + optional ranking integration
  - `src/pgdn/confidence.py` - ranking term computation (do not modify)

  Acceptance Criteria:
  - `PYTHONPATH=src python3 -m pgdn_torch.infer.pgdnv0_infer --checkpoint runs/acp_baseline_pgdnv0_torch/checkpoint_none.pt --targets data/targets/acp_targets.jsonl --limit 32 --samples 4 --num-steps 20 --seed 7 --out runs/acp_baseline_pgdnv0_torch_infer` exits 0
  - `test -s runs/acp_baseline_pgdnv0_torch_infer/infer_meta.json` exits 0
  - `test -d runs/acp_baseline_pgdnv0_torch_infer/samples` exits 0

---

## Success Criteria

### Verification Commands
```bash
test -s Ancient-Chinese-Phonology/dataset/ancient_chinese_phonology.csv

PYTHONPATH=src python3 -c "import sys, types, pathlib, importlib.util; \
import pgdn; \
data_dir = pathlib.Path(pgdn.__file__).parent / 'data'; \
pkg = types.ModuleType('pgdn.data'); pkg.__path__ = [str(data_dir)]; sys.modules['pgdn.data'] = pkg; \
spec = importlib.util.spec_from_file_location('pgdn.data.build_acp', data_dir / 'build_acp.py'); \
mod = importlib.util.module_from_spec(spec); sys.modules['pgdn.data.build_acp'] = mod; spec.loader.exec_module(mod); \
sys.argv = ['build_acp', '--csv', 'Ancient-Chinese-Phonology/dataset/ancient_chinese_phonology.csv', '--out', 'data', '--seed', '42']; \
mod.main()"

test -s data/targets/acp_targets.jsonl
test -s data/splits/manifest.json

PYTHONPATH=src python3 -c "from pathlib import Path; from pgdn_torch.pgdnv0.data import ACPJsonlDataset; ds=ACPJsonlDataset(Path('data/targets/acp_targets.jsonl'), limit=1); _=ds[0]; print('ok_dataset')"

PYTHONPATH=src python3 -m pgdn_torch.train.pgdnv0_train \
  --targets data/targets/acp_targets.jsonl \
  --limit 2048 \
  --epochs 1 \
  --batch-size 64 \
  --num-workers 0 \
  --seed 42 \
  --out runs/acp_baseline_pgdnv0_torch

test -s runs/acp_baseline_pgdnv0_torch/checkpoint_none.pt

PYTHONPATH=src python3 -m pgdn_torch.infer.pgdnv0_infer \
  --checkpoint runs/acp_baseline_pgdnv0_torch/checkpoint_none.pt \
  --targets data/targets/acp_targets.jsonl \
  --limit 32 \
  --samples 4 \
  --num-steps 20 \
  --seed 7 \
  --out runs/acp_baseline_pgdnv0_torch_infer

test -s runs/acp_baseline_pgdnv0_torch_infer/infer_meta.json
```

### Final Checklist
- [ ] ACP CSV exists and can be built into targets
- [ ] Targets JSONL schema validates (32-dim vectors + IMNC mask)
- [ ] Torch training produces a checkpoint under `runs/`
- [ ] Torch inference produces samples under `runs/`
- [ ] No generated artifacts are committed to git
