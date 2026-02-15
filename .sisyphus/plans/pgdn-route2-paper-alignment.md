# PGDN Route (2): Paper-Alignment Implementation Plan (Pairformer + Diffusion + Constraints)

## TL;DR

> **Quick Summary**: Keep the torch pipeline runnable end-to-end (ACP baseline stays green), then incrementally upgrade (1) Pairformer trunk, (2) diffusion training/sampling, and (3) constraint system—each behind flags with agent-executable verification.
>
> **Deliverables**:
> - Upgraded modules under `src/pgdn_torch/pgdnv0/` (pairformer trunk, diffusion, constraints, splits/batching)
> - Upgraded CLIs: `python -m pgdn_torch.train.pgdnv0_train` and `python -m pgdn_torch.infer.pgdnv0_infer`
> - Deterministic smoke runs + artifacts under `runs/` (train metrics, checkpoints, infer samples, ranking CSV)
> - Optional: small eval script to quantify constraint satisfaction + range contract
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES (model refactor vs CLI/eval can be parallel)
> **Critical Path**: Decide vector range + batching semantics -> implement trunk changes -> keep train/infer green

---

## Context

### Original Request
- “写一个计划，这些都要实现”：把论文路线里核心三件套都做出来，并且保持可跑。
  - Pairformer 增强
  - Diffusion 增强
  - 约束（IMNC + 分层/硬偏置）增强

### Current Baseline (already working)
- ACP data build:
  - Input: `Ancient-Chinese-Phonology/dataset/ancient_chinese_phonology.csv`
  - Builder: `src/pgdn/data/build_acp.py`
  - Outputs: `data/targets/acp_targets.jsonl`, `data/splits/manifest.json`
- Torch pipeline:
  - Model: `src/pgdn_torch/pgdnv0/model.py`
  - Train CLI: `src/pgdn_torch/train/pgdnv0_train.py`
  - Infer CLI: `src/pgdn_torch/infer/pgdnv0_infer.py`
  - Split filtering: `src/pgdn_torch/pgdnv0/splits.py` and `ACPJsonlDataset(include_ids=...)`
  - Ranking integration (read-only): `src/pgdn/confidence.py`

### Spec Reference (in-repo, verifiable)
- AF3/Pairformer conceptual anchors will be taken from the **actual AF3 code** in this repo submodule:
  - `alphafold3/src/alphafold3/model/model.py` (recycling pattern)
- Internal direction notes:
  - `.sisyphus/plans/pgdn-af3-torch.md` (project-specific mapping notes)

---

## Key Decisions (defaults applied; can be overridden)

### Decision 1: Vector Range Contract (DEFAULT)
- **Default**: enforce model output vectors to remain in roughly `[-1, 1]` to stay compatible with:
  - ACP targets produced by `build_acp.py` (token hashing yields values in [-1, 1])
  - `src/pgdn/confidence.py` ranking penalty that treats `|v| > 1` as “impossible combo” energy.
- Implementation approach: apply `tanh` (or scale+`tanh`) to diffusion x0 predictions and/or final sampled vectors.

### Decision 2: Pairformer Graph Semantics (DEFAULT)
- **Default**: keep “batch-as-graph” semantics for Pairformer, but make it meaningful and controlled by batching:
  - Introduce a **graph-aware batch sampler** so batches are drawn from a single `graph_id` (ACP already has `graph:<language>`).
  - Add a flag to disable coupling (per-record independence) if needed.

### Decision 3: Canonical End-to-End Pipeline
- **Default**: torch pipeline (`pgdn_torch.*`) is the primary runnable path; non-torch `pgdn.*` remains untouched.

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> Every task below must be verifiable by commands and/or saved artifacts. No manual inspection.

### Test Decision
- Repo has no formal unit test harness; rely on:
  - `python3 -m compileall -q src scripts`
  - deterministic smoke runs on ACP splits
  - schema checks + range checks + constraint checks

### Evidence Convention
- Save artifacts under `runs/`.
- If adding extra evidence files (optional), write to `.sisyphus/evidence/`.

---

## Execution Strategy

Wave 1 (Contracts + plumbing; keep baseline runnable)
- Task 1: Range contract + checks
- Task 2: Graph-aware batching (single-graph batches)

Wave 2 (Pairformer upgrades)
- Task 3: Bidirectional triangle multiplication + gating
- Task 4: True recycling semantics (prev state) + convergence metrics

Wave 3 (Diffusion upgrades)
- Task 5: Schedule options + prediction type (eps/v/x0)
- Task 6: Classifier-free guidance (CFG) + EMA (optional)

Wave 4 (Constraint upgrades)
- Task 7: Hierarchical slot-weighted constraints + group coupling options

Wave 5 (Eval + docs)
- Task 8: Lightweight eval script + CLI integration
- Task 9: Docs update + reproducible commands

Critical Path: 1 -> 2 -> 3 -> 4 -> 5 -> 7 -> 8

---

## TODOs

- [x] 1) Lock down vector range contract across diffusion + ranking

  What to do:
  - Add a single “range contract” mechanism used consistently in:
    - diffusion loss path (x0_hat)
    - diffusion sample output
  - Add a CLI flag to toggle enforcement (default ON).
  - Add an automated range check step in infer.

  References:
  - `src/pgdn/data/build_acp.py:token_vector()` - produces target vectors in [-1, 1]
  - `src/pgdn/confidence.py` - penalizes abs(v) > 1
  - `src/pgdn_torch/pgdnv0/model.py:DiffusionHead` - diffusion training/sampling

  Acceptance Criteria:
  - `PYTHONPATH=src python3 -m pgdn_torch.infer.pgdnv0_infer ...` emits (or writes) a metric that `max_abs <= 1.2` (configurable threshold) when enforcement ON

  Agent-Executed QA Scenario (Bash):
  - Train (small):
    - `PYTHONPATH=src python3 -m pgdn_torch.train.pgdnv0_train --targets data/targets/acp_targets.jsonl --split-manifest data/splits/manifest.json --split-strategy random --split train --limit 2048 --epochs 1 --batch-size 32 --num-workers 0 --seed 42 --out runs/range_contract_train`
  - Infer (small):
    - `PYTHONPATH=src python3 -m pgdn_torch.infer.pgdnv0_infer --checkpoint runs/range_contract_train/checkpoint_none.pt --targets data/targets/acp_targets.jsonl --split-manifest data/splits/manifest.json --split-strategy random --split dev --limit 16 --samples 2 --num-steps 10 --seed 7 --out runs/range_contract_infer`
    - Assert: `runs/range_contract_infer/infer_meta.json` exists

- [x] 2) Add graph-aware batching for ACP (so Pairformer coupling is meaningful)

  What to do:
  - Extend `ACPJsonlDataset` to optionally expose `graph_id`.
  - Add a batch sampler that groups record_ids by `graph_id` and yields indices from one graph per batch.
  - Add `--batching {flat,graph}` to train/infer. Default: `graph` when Pairformer enabled.

  References:
  - `src/pgdn/data/build_acp.py` - every target has `graph_id = graph:<language>`
  - `src/pgdn_torch/train/pgdnv0_train.py` - DataLoader setup
  - `src/pgdn_torch/infer/pgdnv0_infer.py` - inference batching
  - `src/pgdn_torch/pgdnv0/model.py:TriangleMultiplicativeUpdate` - O(B^3) cost requires bounded B

  Acceptance Criteria:
  - With `--batching graph`, a training epoch completes with `batch_size <= 64` without OOM on GPU and without excessive CPU time.

- [x] 3) Pairformer: implement bidirectional triangle multiplication + stronger gating

  What to do:
  - Replace single-path `TriangleMultiplicativeUpdate` with two paths:
    - outgoing: i->k, k->j
    - incoming: k->i, j->k (or equivalent transpose path)
  - Each path has its own projections and gates; outputs are summed then projected.
  - Keep ablation flag `no_pairformer` functional.
  - Add `--pairformer-blocks` and `--recycle` support (already present) and ensure defaults are safe for O(B^3).

  References:
  - `src/pgdn_torch/pgdnv0/model.py:PairformerLite` - current trunk
  - `src/pgdn/pairformer.py` - non-torch baseline for naming/intent
  - `alphafold3/src/alphafold3/model/model.py` - recycling pattern reference

  Acceptance Criteria:
  - `python3 -m compileall -q src scripts` passes
  - ACP train smoke with `--batching graph --batch-size 32` exits 0

- [x] 4) Pairformer: “true recycling” semantics (prev state) + convergence metrics

  What to do:
  - Modify PairformerLite to accept optional `prev_single/prev_pair` inputs.
  - In inference, recycle K times and record per-iteration deltas (L2 change) into `infer_meta.json`.
  - Ensure deterministic behavior with fixed seeds and stable batching.

  References:
  - `alphafold3/src/alphafold3/model/model.py` - recycling pattern reference
  - `src/pgdn_torch/pgdnv0/model.py:PairformerLite`

  Acceptance Criteria:
  - In inference with `--recycle 3`, `infer_meta.json` includes `recycle_deltas` array of length 3

- [x] 5) Diffusion: schedule options + prediction type (eps/v/x0)

  What to do:
  - Add schedule choices: linear (current), cosine (common), and a small-step “fast” schedule.
  - Add prediction type options:
    - eps (current)
    - v-pred (often more stable)
  - Ensure sampling uses consistent formulation for the chosen pred type.

  References:
  - `src/pgdn_torch/pgdnv0/model.py:DiffusionHead`
  - `denoising-diffusion-pytorch/` (submodule) - reference implementation patterns (do not modify)

  Acceptance Criteria:
  - Train smoke works with `--diffusion-schedule cosine` and `--diffusion-pred v` (flags to be added)

- [x] 6) Diffusion: classifier-free guidance (CFG) + optional EMA weights

  What to do:
  - Implement conditioning dropout during training (e.g., 10-20%) to enable CFG.
  - Add inference flag `--cfg-scale` (default 1.0 = off).
  - Optionally maintain EMA weights for diffusion net and use EMA at inference.

  References:
  - `src/pgdn_torch/pgdnv0/model.py:DiffusionHead.net` - conditioning input (x_t, cond, t)

  Acceptance Criteria:
  - Infer smoke runs with `--cfg-scale 1.5` and produces samples
  - If EMA enabled, checkpoint includes EMA state and infer can select it

- [x] 7) Constraints: hierarchical slot-weighted loss + optional group coupling

  What to do:
  - Replace single scalar constraint with:
    - per-slot weights (I/M/N/C)
    - optional per-dimension weights within a slot
  - Add “group coupling” options for diffusion sampling:
    - **Group key (locked)**: `character` derived from ACP `record_id` format `<character>:<language>`.
      - Extraction: `character = record_id.split(":", 1)[0]`
      - Malformed handling: count malformed `record_id` in meta; default behavior is warning + fallback `character = record_id` (do NOT crash unless fail-fast flag is set).
    - **Coupling scope (locked)**: global within an infer run (independent of batch size, ordering, DataLoader worker count).
      - Do NOT rely on within-batch co-occurrence.
      - Do NOT change `--batching graph` semantics for this milestone.
    - **shared_noise definition (locked)**: share initial diffusion noise `x_T` only.
      - Determinism contract: for a fixed `--seed`, the same `(character, sample_index)` must produce the same `x_T` across runs.
      - `sample_index` semantics: per-record draw index in `0..(--samples-1)`.
    - **Noise generation approach (recommended)**:
      - Derive a per-group seed via a stable hash of `(seed, character, sample_index)` (e.g., sha256/blake2) and seed an explicit `torch.Generator`.
      - Generate `x_T` on-demand (avoid caching full tensors).
    - **shared_denoise follow-on (strict semantics locked)**: share the CFG unconditional denoise branch per group.
      - When `cfg_scale != 1.0`: at each diffusion step, compute unconditional prediction `y_u` once per group key and reuse it for all records in that group.
      - Compute conditional prediction `y_c` per-record as usual.
      - Combine as `y_hat = y_u + cfg_scale * (y_c - y_u)` (so coupling affects only the shared unconditional component).
      - When `cfg_scale == 1.0`: treat shared_denoise as a no-op (still log meta; do not error).
      - Guardrail: do NOT share per-step noise beyond the sampler's existing behavior (keep this milestone aligned with x_T-only shared_noise).
    - **Verification (locked)**: write coupling diagnostics into `runs/.../infer_meta.json`.
      - Include: mode, key type, group_count, a per-key fingerprint of `x_T` (hash/summary), and invariance checks.
      - Fingerprint should be computed in a device-agnostic way (e.g., `x_T.detach().float().cpu().numpy().tobytes()` + sha256) so CPU/GPU placement does not change the hash.
      - Fail-fast: OFF by default; provide opt-in `--fail-on-coupling-mismatch`.
  - Keep `--ablation no_constraint_loss` working.

  References:
  - `src/pgdn_torch/pgdnv0/model.py:constraint_loss` - current expansion 4*8
  - `.sisyphus/plans/pgdn-af3-torch.md` - internal notes for hierarchical loss + shared noise coupling ideas
  - `src/pgdn/data/build_acp.py` - ACP target schema; `record_id` / `graph_id` / `normalized_pron`
  - `src/pgdn_torch/infer/pgdnv0_infer.py` - infer flags + meta writing entrypoint
  - `src/pgdn_torch/pgdnv0/model.py:DiffusionHead.sample()` - sampling entrypoint (inject shared x_T)

  Acceptance Criteria:
  - Train smoke works with hierarchical constraint enabled
  - A metrics row includes per-slot constraint terms (e.g., `constraint_I`, `constraint_M`, ...)
  - With coupling enabled, `infer_meta.json` includes a `coupling` section with deterministic fingerprints.
  - Invariance check (batch-size independence): running the same infer job twice with different `--batch-size` yields identical `coupling.shared_noise.fingerprints_by_key` given the same `--seed`.
  - With shared_denoise enabled and `--cfg-scale 1.5`, `infer_meta.json` includes a `coupling.shared_denoise` (or equivalent) section with deterministic digests/fingerprints and `mismatch_count == 0`.
  - Invariance check (batch-size independence): shared_denoise digests/fingerprints are identical across `--batch-size 4` vs `--batch-size 32` given the same `--seed`.

  Agent-Executed QA Scenario (Bash + Python):
  - Preconditions: ACP artifacts exist; checkpoint exists.
  - Run infer (batch-size A):
    - `PYTHONPATH=src python3 -m pgdn_torch.infer.pgdnv0_infer --checkpoint runs/pgdnv0_paper_align_smoke/checkpoint_none.pt --targets data/targets/acp_targets.jsonl --split-manifest data/splits/manifest.json --split-strategy random --split dev --limit 64 --batch-size 4 --batching graph --samples 2 --num-steps 10 --seed 7 --group-coupling shared_noise --group-key character --out runs/task7_coupling_bs4`
  - Run infer (batch-size B):
    - `PYTHONPATH=src python3 -m pgdn_torch.infer.pgdnv0_infer --checkpoint runs/pgdnv0_paper_align_smoke/checkpoint_none.pt --targets data/targets/acp_targets.jsonl --split-manifest data/splits/manifest.json --split-strategy random --split dev --limit 64 --batch-size 32 --batching graph --samples 2 --num-steps 10 --seed 7 --group-coupling shared_noise --group-key character --out runs/task7_coupling_bs32`
  - Assert invariance via a non-interactive script:
    - `python3 - <<'PY'
import json
from pathlib import Path
a=json.loads(Path('runs/task7_coupling_bs4/infer_meta.json').read_text(encoding='utf-8'))
b=json.loads(Path('runs/task7_coupling_bs32/infer_meta.json').read_text(encoding='utf-8'))
fa=a['coupling']['shared_noise']['fingerprints_by_key']
fb=b['coupling']['shared_noise']['fingerprints_by_key']
assert fa==fb, 'coupling fingerprints differ across batch sizes'
print('OK: coupling fingerprints invariant across batch sizes')
PY`

  Agent-Executed QA Scenario (Bash + Python) — shared_denoise (unconditional branch):
  - Run infer (batch-size A, CFG enabled):
    - `PYTHONPATH=src python3 -m pgdn_torch.infer.pgdnv0_infer --checkpoint runs/pgdnv0_paper_align_smoke/checkpoint_none.pt --targets data/targets/acp_targets.jsonl --split-manifest data/splits/manifest.json --split-strategy random --split dev --limit 64 --batch-size 4 --batching graph --samples 2 --num-steps 10 --seed 7 --cfg-scale 1.5 --group-coupling shared_denoise --group-key character --out runs/task7_denoise_bs4`
  - Run infer (batch-size B, CFG enabled):
    - `PYTHONPATH=src python3 -m pgdn_torch.infer.pgdnv0_infer --checkpoint runs/pgdnv0_paper_align_smoke/checkpoint_none.pt --targets data/targets/acp_targets.jsonl --split-manifest data/splits/manifest.json --split-strategy random --split dev --limit 64 --batch-size 32 --batching graph --samples 2 --num-steps 10 --seed 7 --cfg-scale 1.5 --group-coupling shared_denoise --group-key character --out runs/task7_denoise_bs32`
  - Assert invariance (digests or fingerprints):
    - `python3 - <<'PY'
import json
from pathlib import Path
a=json.loads(Path('runs/task7_denoise_bs4/infer_meta.json').read_text(encoding='utf-8'))
b=json.loads(Path('runs/task7_denoise_bs32/infer_meta.json').read_text(encoding='utf-8'))
da=a['coupling']['shared_denoise']['fingerprints_digest_xor64_by_sample']
db=b['coupling']['shared_denoise']['fingerprints_digest_xor64_by_sample']
assert da==db, 'shared_denoise digests differ across batch sizes'
print('OK: shared_denoise digests invariant across batch sizes')
PY`

- [x] 8) Add a lightweight eval script for baseline regression checks

  What to do:
  - Create a script (e.g. `scripts/eval_torch_pgdnv0.py`) that:
    - loads a checkpoint
    - runs inference for N samples on dev split
    - reports:
      - max_abs
      - mean(|v|>1) penalty
      - constraint satisfaction stats
      - optional ranking summary

  References:
  - `src/pgdn_torch/infer/pgdnv0_infer.py` - sampling + ranking output
  - `src/pgdn/confidence.py` - ranking terms

  Acceptance Criteria:
  - Script exits 0 and writes `runs/<job>/eval.json`

- [x] 9) Documentation and reproducibility

  What to do:
  - Update `README.md` with the new flags:
    - `--split-manifest --split-strategy --split`
    - `--batching graph`
    - diffusion schedule/pred/CFG flags
  - Add a short doc describing “graph semantics” and why we batch by `graph_id`.

  References:
  - `README.md`
  - `docs/datasets_organization.md` (do not change its constraints; just add a small torch section if needed)

  Acceptance Criteria:
  - Copy/paste commands in README run successfully on a fresh checkout (after submodules init)

- [x] 10) Benchmark: run a repeatable comparison across coupling modes

  What to do:
  - Create `scripts/benchmark_group_coupling.py` as a thin subprocess orchestrator.
  - For a single checkpoint + dataset slice, run infer + eval across 3 modes:
    - `none`
    - `shared_noise`
    - `shared_denoise` (must run with `--cfg-scale != 1.0` so it is effective)
  - Write an aggregated `summary.json` under the output directory:
    - Includes paths to each mode's infer/eval outputs
    - Embeds key fields from `infer_meta.json` and `eval.json`
    - Captures coupling invariants from `infer_meta.json`:
      - `mismatch_count == 0` for enabled modes
      - `shared_denoise.effective == true` when mode is `shared_denoise` and cfg_scale != 1
  - Optional: write `summary.csv` (one row per mode) when `--csv` is set.

  Guardrails:
  - Do NOT commit any artifacts under `runs/`.
  - Keep runtime bounded by defaults: `limit=64`, `samples=2`, `num_steps=10`.
  - Explicit preflight checks with actionable error messages:
    - Missing checkpoint -> suggest the existing train smoke command in this plan
    - Missing ACP artifacts -> suggest `PYTHONPATH=src python3 -m pgdn.data.build_acp ...`
  - Eval script note: `scripts/eval_torch_pgdnv0.py` does its own sampling; treat eval as “quality sanity”, while coupling-specific verification is `infer_meta.json`.

  References:
  - `src/pgdn_torch/infer/pgdnv0_infer.py` - coupling flags + `infer_meta.json` schema
  - `scripts/eval_torch_pgdnv0.py` - writes `eval.json`
  - `scripts/run_v0_experiments.py` - subprocess orchestration style (print commands, `subprocess.run(check=True)`)

  Acceptance Criteria:
  - `python3 -m compileall -q src scripts` passes
  - Happy path (ACP):
    - Preconditions:
      - `test -s runs/pgdnv0_paper_align_smoke/checkpoint_none.pt`
      - `test -s data/targets/acp_targets.jsonl`
      - `test -s data/splits/manifest.json`
    - Run:
      - `PYTHONPATH=src python3 scripts/benchmark_group_coupling.py \
          --checkpoint runs/pgdnv0_paper_align_smoke/checkpoint_none.pt \
          --targets data/targets/acp_targets.jsonl \
          --split-manifest data/splits/manifest.json \
          --split-strategy random \
          --split dev \
          --limit 64 \
          --batch-size 16 \
          --batching graph \
          --samples 2 \
          --num-steps 10 \
          --seed 7 \
          --cfg-scale 1.5 \
          --out runs/bench_group_coupling_smoke \
          --csv`
    - Assert (non-interactive):
      - `python3 - <<'PY'
import json
from pathlib import Path
root = Path('runs/bench_group_coupling_smoke')
summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
assert set(summary['modes']) == {'none','shared_noise','shared_denoise'}
for mode in summary['modes']:
    m = summary['by_mode'][mode]
    assert Path(m['infer_dir']).joinpath('infer_meta.json').is_file()
    assert Path(m['eval_dir']).joinpath('eval.json').is_file()
    c = m['infer_meta']['coupling']
    if mode == 'none':
        assert c['shared_noise']['enabled'] is False
    else:
        assert c['shared_noise']['enabled'] is True
        assert c['shared_noise']['mismatch_count'] == 0
    if mode == 'shared_denoise':
        assert c['shared_denoise']['enabled'] is True
        assert c['shared_denoise']['effective'] is True
    else:
        assert c['shared_denoise']['effective'] is False
print('OK')
PY`
  - Hygiene: run artifacts are ignored and untracked:
    - `git check-ignore -q runs/bench_group_coupling_smoke/summary.json` exits 0
    - `git ls-files --error-unmatch runs/bench_group_coupling_smoke/summary.json` exits non-zero

- [x] 11) Metric: character-consistency from infer samples (prove coupling is useful)

  Goal:
  - Add a linguistics-driven metric that directly measures whether coupling increases within-character consistency across records (periods/languages).

  What to do:
  - Extend `scripts/benchmark_group_coupling.py` to compute a new metric block per mode by reading infer artifacts:
    - Input: `runs/<bench>/<mode>/infer/samples/sample_XXX.pt` written by `pgdn_torch.infer.pgdnv0_infer`.
      - Each file contains: `{ "record_id": list[str], "vector": Tensor[B, 32] }`.
    - For each `record_id`, compute `v_mean(record_id)` by averaging vectors across all sample_XXX.pt files.
    - Define `character = record_id.split(":", 1)[0]`.
    - For each character with `n_records >= 2`:
      - Compute mean pairwise cosine distance across record mean vectors:
        - `cosine_distance(u,v) = 1 - (u·v)/(||u||*||v||)`
      - Exclude any record vectors with zero norm (count and report skips; do not silently treat as 0).
    - Aggregate overall stats (character-weighted; each character contributes equally):
      - `mean`, `median`, `p90` of per-character mean distance
      - `n_characters_total`, `n_characters_used`, `n_characters_excluded_small`, `n_records_used`, `n_records_skipped_zero_norm`
  - Output schema (locked):
    - Write per-mode overall stats into `runs/<bench>/summary.json` under `by_mode[mode]["character_consistency"]`.
    - Write per-mode detailed per-character results to `runs/<bench>/<mode>/infer/metrics/character_consistency/by_character.json`.
      - Guardrail: store summaries only (n_records, mean_distance, optional percentiles). Do NOT store raw vectors or pairwise matrices.
    - Include `metric_version: 1` in both summary and by_character outputs.

  Guardrails:
  - `torch.load` on `.pt` is pickle-based; treat inputs as trusted local artifacts produced by our own infer runs only.
  - Fail-fast on schema mismatch (len(record_id) != vector.shape[0], missing keys, no sample files) with actionable error.
  - Keep runtime bounded: do not compute or serialize O(k^2) matrices.
  - Determinism: write JSON with stable ordering (sort characters; use deterministic key ordering) so sha256 checks are meaningful.

  References:
  - `src/pgdn_torch/infer/pgdnv0_infer.py` - writes `samples/sample_XXX.pt` via `torch.save({"record_id": ..., "vector": ...})`
  - `scripts/benchmark_group_coupling.py` - aggregation point (already writes `summary.json`/`summary.csv`)

  Acceptance Criteria:
  - `python3 -m compileall -q src scripts` passes
  - Happy path (extends existing Task 10 benchmark):
    - `PYTHONPATH=src python3 scripts/benchmark_group_coupling.py --checkpoint runs/pgdnv0_paper_align_smoke/checkpoint_none.pt --targets data/targets/acp_targets.jsonl --split-manifest data/splits/manifest.json --split-strategy random --split dev --limit 64 --batch-size 16 --batching graph --samples 2 --num-steps 10 --seed 7 --cfg-scale 1.5 --out runs/bench_group_coupling_smoke --csv`
    - Assert (non-interactive):
      - `python3 - <<'PY'
import json
from pathlib import Path
root = Path('runs/bench_group_coupling_smoke')
summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
for mode in summary['modes']:
    m = summary['by_mode'][mode]
    cc = m.get('character_consistency')
    assert isinstance(cc, dict)
    assert cc.get('metric_version') == 1
    for k in ['mean','median','p90','n_characters_total','n_characters_used','n_characters_excluded_small']:
        assert k in cc
    by_char = root / mode / 'infer' / 'metrics' / 'character_consistency' / 'by_character.json'
    assert by_char.is_file() and by_char.stat().st_size > 0
    obj = json.loads(by_char.read_text(encoding='utf-8'))
    assert obj.get('metric_version') == 1
    assert 'by_character' in obj
print('OK')
PY`
  - Determinism check (same bench rerun yields identical metric outputs):
    - `sha256sum runs/bench_group_coupling_smoke/none/infer/metrics/character_consistency/by_character.json`
    - rerun the same benchmark command
    - `sha256sum ...` matches exactly

- [x] 12) Data: download external resources (local-only) into `data/external/`

  Goal:
  - Prepare inputs for later extrinsic metrics (Qieyun-class coherence and phonetic-series coherence), without committing or vendoring external datasets.

  What to do:
  - Add a small helper script `scripts/fetch_external_linguistics_data.py` that downloads:
    - Tshet-uinh Qieyun data: `https://github.com/nk2028/tshet-uinh-data` into `data/external/tshet-uinh-data/` (git clone).
    - Unihan data (UAX #38 / UCD) into `data/external/unihan/` (download the official Unihan zip and extract).
  - Local-only policy (locked):
    - Do NOT add submodules.
    - Do NOT vendor snapshot files.
    - Do NOT commit downloaded artifacts.
  - Provenance logging (required): print the source URL + resolved commit hash (for git clone) and the downloaded filename/hash (for Unihan zip).

  Acceptance Criteria:
  - `PYTHONPATH=src python3 scripts/fetch_external_linguistics_data.py --out data/external` exits 0
  - `test -d data/external/tshet-uinh-data/.git` and `git -C data/external/tshet-uinh-data rev-parse HEAD` succeeds
  - `test -s data/external/unihan/Unihan.zip` and `test -s data/external/unihan/Unihan_Readings.txt`
  - Hygiene: artifacts remain untracked:
    - `git ls-files --error-unmatch data/external/tshet-uinh-data` exits non-zero
    - `git ls-files --error-unmatch data/external/unihan/Unihan.zip` exits non-zero

---

## Success Criteria

### Verification Commands (baseline regression)
```bash
python3 -m compileall -q src scripts

# Ensure ACP artifacts exist (or rebuild)
test -s Ancient-Chinese-Phonology/dataset/ancient_chinese_phonology.csv
PYTHONPATH=src python3 -m pgdn.data.build_acp \
  --csv Ancient-Chinese-Phonology/dataset/ancient_chinese_phonology.csv \
  --out data \
  --seed 42

# Train on train split only
PYTHONPATH=src python3 -m pgdn_torch.train.pgdnv0_train \
  --targets data/targets/acp_targets.jsonl \
  --split-manifest data/splits/manifest.json \
  --split-strategy random \
  --split train \
  --limit 2048 \
  --epochs 1 \
  --batch-size 32 \
  --num-workers 0 \
  --seed 42 \
  --out runs/pgdnv0_paper_align_smoke

test -s runs/pgdnv0_paper_align_smoke/checkpoint_none.pt

# Infer on dev split only
PYTHONPATH=src python3 -m pgdn_torch.infer.pgdnv0_infer \
  --checkpoint runs/pgdnv0_paper_align_smoke/checkpoint_none.pt \
  --targets data/targets/acp_targets.jsonl \
  --split-manifest data/splits/manifest.json \
  --split-strategy random \
  --split dev \
  --limit 16 \
  --samples 2 \
  --num-steps 10 \
  --seed 7 \
  --out runs/pgdnv0_paper_align_smoke_infer

test -s runs/pgdnv0_paper_align_smoke_infer/infer_meta.json
```

### Final Checklist
- [x] ACP baseline stays runnable end-to-end
- [x] Pairformer improvements are behind flags and do not break `--ablation` options
- [x] Diffusion improvements (schedule/pred/CFG) are behind flags and are verifiable via smoke runs
- [x] Constraint improvements log measurable terms (not just a single scalar)
- [x] Range contract is consistent with ranking and targets

---

## Option 2 (Framework Core) - Next Execution Pack

Use this when coupling evidence is complete and we switch back to core architecture implementation hardening.

### Immediate Goals
- Validate Pairformer core path (bidirectional triangle + true recycling) with fresh smoke artifacts.
- Validate diffusion knobs (`schedule`, `pred`, `CFG`, `EMA`) on the same ACP slice to ensure no regressions.
- Close the plan checklist by producing one reproducible "all-green" command bundle.

### Execution Batch A (Pairformer core)
1. Train smoke with Pairformer enabled and explicit recycle:
   - `PYTHONPATH=src python3 -m pgdn_torch.train.pgdnv0_train --targets data/targets/acp_targets.jsonl --split-manifest data/splits/manifest.json --split-strategy random --split train --limit 2048 --epochs 1 --batch-size 32 --batching graph --pairformer-blocks 2 --recycle 3 --seed 42 --out runs/option2_pairformer_train`
2. Infer smoke with recycle verification:
   - `PYTHONPATH=src python3 -m pgdn_torch.infer.pgdnv0_infer --checkpoint runs/option2_pairformer_train/checkpoint_none.pt --targets data/targets/acp_targets.jsonl --split-manifest data/splits/manifest.json --split-strategy random --split dev --limit 128 --batch-size 32 --batching graph --samples 2 --num-steps 10 --seed 7 --out runs/option2_pairformer_infer`
3. Assert `recycle_deltas` length is 3 in `runs/option2_pairformer_infer/infer_meta.json`.

### Execution Batch B (Diffusion core)
1. Train smoke with cosine + v-pred + CFG-ready dropout + EMA:
   - `PYTHONPATH=src python3 -m pgdn_torch.train.pgdnv0_train --targets data/targets/acp_targets.jsonl --split-manifest data/splits/manifest.json --split-strategy random --split train --limit 2048 --epochs 1 --batch-size 32 --batching graph --diffusion-schedule cosine --diffusion-pred v --cond-dropout 0.1 --ema-decay 0.999 --seed 42 --out runs/option2_diffusion_train`
2. Infer smoke with CFG + EMA:
   - `PYTHONPATH=src python3 -m pgdn_torch.infer.pgdnv0_infer --checkpoint runs/option2_diffusion_train/checkpoint_none.pt --targets data/targets/acp_targets.jsonl --split-manifest data/splits/manifest.json --split-strategy random --split dev --limit 128 --batch-size 32 --batching graph --samples 2 --num-steps 10 --cfg-scale 1.5 --use-ema --seed 7 --out runs/option2_diffusion_infer`
3. Eval smoke:
   - `PYTHONPATH=src python3 scripts/eval_torch_pgdnv0.py --checkpoint runs/option2_diffusion_train/checkpoint_none.pt --targets data/targets/acp_targets.jsonl --split-manifest data/splits/manifest.json --split-strategy random --split dev --limit 128 --samples 2 --num-steps 10 --batch-size 32 --batching graph --cfg-scale 1.5 --use-ema --out runs/option2_diffusion_eval`

### Exit Criteria for Option 2
- `compileall` passes.
- Pairformer infer meta contains recycle deltas with expected length.
- Diffusion run succeeds for cosine + v-pred + CFG + EMA.
- Eval writes `runs/option2_diffusion_eval/eval.json`.
- No coupling/range regression in infer meta (`range_ok=true`, no mismatch explosions).

### One-Shot Execution Checklist (for direct rerun)

Use this as a single sequential task list when you want to rerun the whole validated path without re-planning.

- [ ] Preflight: `python3 -m compileall -q src scripts`
- [ ] Ensure baseline checkpoint exists: `runs/option2_diffusion_train/checkpoint_none.pt` (or rerun Batch B train)
- [ ] Run final reproducible snapshot: train -> infer -> eval (commands below)
- [ ] Verify `infer_meta.json` contains `range_ok=true` and `recycle_deltas`
- [ ] Verify `eval.json` contains `max_abs`, `frac_abs_gt_1`, `abs_gt_1_penalty_mean`, `constraint_penalty_mean`, `constraint_penalty_min`, `constraint_penalty_max`

```bash
set -euo pipefail
export PYTHONPATH=src

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="runs/route2_final_smoke_${STAMP}"

python3 -m compileall -q src scripts

python3 -m pgdn_torch.train.pgdnv0_train \
  --targets data/targets/acp_targets.jsonl \
  --split-manifest data/splits/manifest.json \
  --split-strategy random \
  --split train \
  --limit 2048 \
  --epochs 1 \
  --batch-size 32 \
  --batching graph \
  --diffusion-schedule cosine \
  --diffusion-pred v \
  --cond-dropout 0.1 \
  --ema-decay 0.999 \
  --seed 42 \
  --out "${OUT_BASE}_train"

python3 -m pgdn_torch.infer.pgdnv0_infer \
  --checkpoint "${OUT_BASE}_train/checkpoint_none.pt" \
  --targets data/targets/acp_targets.jsonl \
  --split-manifest data/splits/manifest.json \
  --split-strategy random \
  --split dev \
  --limit 128 \
  --batch-size 32 \
  --batching graph \
  --samples 2 \
  --num-steps 10 \
  --cfg-scale 1.5 \
  --use-ema \
  --seed 7 \
  --out "${OUT_BASE}_infer"

python3 scripts/eval_torch_pgdnv0.py \
  --checkpoint "${OUT_BASE}_train/checkpoint_none.pt" \
  --targets data/targets/acp_targets.jsonl \
  --split-manifest data/splits/manifest.json \
  --split-strategy random \
  --split dev \
  --limit 128 \
  --samples 2 \
  --num-steps 10 \
  --batch-size 32 \
  --batching graph \
  --cfg-scale 1.5 \
  --use-ema \
  --out "${OUT_BASE}_eval"

python3 - <<'PY'
import json
import os
from pathlib import Path

base = os.environ["OUT_BASE"]
infer_meta = Path(f"{base}_infer/infer_meta.json")
eval_json = Path(f"{base}_eval/eval.json")

assert infer_meta.exists(), f"missing {infer_meta}"
assert eval_json.exists(), f"missing {eval_json}"

im = json.loads(infer_meta.read_text(encoding="utf-8"))
ev = json.loads(eval_json.read_text(encoding="utf-8"))

assert im.get("range_ok") is True, "range_ok is not true"
assert isinstance(im.get("recycle_deltas"), list), "recycle_deltas missing"

required = [
    "max_abs",
    "frac_abs_gt_1",
    "abs_gt_1_penalty_mean",
    "constraint_penalty_mean",
    "constraint_penalty_min",
    "constraint_penalty_max",
]
for k in required:
    assert k in ev, f"missing eval key: {k}"

print("OK final snapshot verified")
print("infer:", infer_meta)
print("eval :", eval_json)
PY
```
