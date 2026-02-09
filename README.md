# AncientChinesePhonology_Alphafold

This is a meta-repository for experiments around Ancient Chinese phonology modeling.

## Linux quickstart

Bootstrap submodules:

```bash
git submodule update --init --recursive
```

Refresh the unified external dataset hub:

```bash
python3 scripts/organize_datasets.py
```

Smoke-check PGDN v0 imports (requires `PYTHONPATH=src`):

```bash
PYTHONPATH=src python3 scripts/smoke_model_interfaces.py
```

Mocha-TIMIT acquisition (Phase 1):

```bash
# dry-run
bash scripts/acquire_mocha_timit.sh --dry-run --root data/external/mocha_timit --speakers fsew0,msak0

# real run
bash scripts/acquire_mocha_timit.sh --root data/external/mocha_timit --speakers fsew0,msak0

# register under data/external/links/
python3 scripts/organize_datasets.py
```

Torch training smoke (1 optimizer step, uses GPU if available):

```bash
PYTHONPATH=src python3 -m pgdn_torch.train.one_step --dataset phoible
```

Torch PGDN v0 (tiny Pairformer + DDPM-style diffusion):

```bash
# train on synthetic data (fast end-to-end sanity check)
PYTHONPATH=src python3 -m pgdn_torch.train.pgdnv0_train --synthetic --epochs 1 --batch-size 32 --pairformer-blocks 2 --recycle 1

# infer from the checkpoint
PYTHONPATH=src python3 -m pgdn_torch.infer.pgdnv0_infer --checkpoint runs/pgdn_torch_v0/checkpoint_none.pt --synthetic --limit 16 --samples 4 --num-steps 20

# ACP baseline (real data)

Generate ACP targets + split manifest (not committed):

```bash
PYTHONPATH=src python3 -m pgdn.data.build_acp \
  --csv Ancient-Chinese-Phonology/dataset/ancient_chinese_phonology.csv \
  --out data \
  --seed 42
```

Train + infer on ACP (defaults use graph-aware batching for Pairformer):

```bash
PYTHONPATH=src python3 -m pgdn_torch.train.pgdnv0_train \
  --targets data/targets/acp_targets.jsonl \
  --split-manifest data/splits/manifest.json \
  --split-strategy random \
  --split train \
  --limit 2048 \
  --epochs 1 \
  --batch-size 32 \
  --batching graph \
  --out runs/pgdnv0_acp_smoke

PYTHONPATH=src python3 -m pgdn_torch.infer.pgdnv0_infer \
  --checkpoint runs/pgdnv0_acp_smoke/checkpoint_none.pt \
  --targets data/targets/acp_targets.jsonl \
  --split-manifest data/splits/manifest.json \
  --split-strategy random \
  --split dev \
  --limit 16 \
  --batch-size 16 \
  --batching graph \
  --samples 2 \
  --num-steps 10 \
  --out runs/pgdnv0_acp_smoke_infer
```

Notes:
- Range contract: diffusion outputs are passed through `tanh` by default to stay compatible with ACP targets and `src/pgdn/confidence.py` (penalizes `|v| > 1`).
- Diffusion knobs: `--diffusion-schedule {linear,cosine,fast}`, `--diffusion-pred {eps,v}`, `--cfg-scale`, `--ema-decay`.
- Pairformer recycling: `infer_meta.json` includes `recycle_deltas` when `--recycle > 1`.
- Group coupling (Task 7): tie diffusion randomness/denoise across records with the same character (derived from `record_id` format `<character>:<language>`):
  - `--group-coupling {none,shared_noise,shared_denoise}` (default `none`)
  - `--group-key character`
  - `shared_noise`: shares initial noise `x_T` globally within the infer run for each `(seed, character, sample_index)`.
  - `shared_denoise`: shares the CFG unconditional branch per character when `--cfg-scale != 1.0` (conditional branch remains per-record).
  - `infer_meta.json` includes `coupling.shared_noise` and `coupling.shared_denoise` with deterministic fingerprints/digests; `--fail-on-coupling-mismatch` turns on fail-fast.
```

## Quick start

Clone with submodules:

```bash
git clone --recurse-submodules <YOUR_GITHUB_REPO_URL>
```

Regenerate the local external dataset index (optional):

```bash
python3 scripts/organize_datasets.py
```

## Third-party code

Several directories are tracked as git submodules and retain their upstream licenses.
See `THIRD_PARTY.md`.

## Data

Generated ACP artifacts under `data/` are intentionally not committed.
They can be rebuilt from upstream sources via:

```bash
python3 scripts/build_from_acp.py --help
```

## Environment

See `docs/linux_environment.md`.
