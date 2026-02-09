# Agents Guide (AncientChinesePhonology_Alphafold)

This repository is a meta-repo for experiments around Ancient Chinese phonology
modeling. It includes several git submodules plus a small Python package under
`src/pgdn/` and runnable orchestration scripts under `scripts/`.

If you are an agent making changes here: keep edits small, prefer extending the
existing scripts/modules, and avoid modifying submodules unless explicitly asked.

## Repo Layout

- `src/pgdn/`: lightweight Python package (train/infer/eval + utilities)
- `scripts/`: entrypoints that call `pgdn.*` modules
- `data/`: generated artifacts (mostly gitignored) + `data/external/` hub
- Submodules: `alphafold3/`, `denoising-diffusion-pytorch/`, `phoible/`, etc.

## Setup Notes

There is no `pyproject.toml` / `setup.py` in this repo. `pgdn` is not installed
as a package by default.

To run anything that imports `pgdn`, set `PYTHONPATH=src`.
Example: `PYTHONPATH=src python3 scripts/smoke_model_interfaces.py`.

If you cloned without submodules, initialize them:

```bash
git submodule update --init --recursive
```

## Build / Lint / Test

This repo does not currently define a formal build/lint/test toolchain:

- No `Makefile`, `pyproject.toml`, `tox.ini`, `noxfile.py`, or GitHub Actions.
- `.gitignore` mentions `.ruff_cache/` and `.pytest_cache/`, but there is no
  repo-level configuration and no test suite checked in.

Treat the scripts below as the canonical "build" entrypoints for experiments.

Common commands (copy/paste):

```bash
# Smoke check (closest thing to "run one test")
PYTHONPATH=src python3 scripts/smoke_model_interfaces.py

# External dataset index + symlink hub
python3 scripts/organize_datasets.py
python3 scripts/organize_datasets.py --dry-run

# Data build from ACP (wrapper around pgdn.data.build_acp)
PYTHONPATH=src python3 -m pgdn.data.build_acp --help
python3 scripts/build_from_acp.py --help

# Train / infer / eval (PGDN v0)
PYTHONPATH=src python3 -m pgdn.train --targets data/targets/targets.jsonl --split-manifest data/splits/split_manifest.json --epochs 1 --ablation none --out runs/pgdn_v0/checkpoints
PYTHONPATH=src python3 -m pgdn.infer --targets data/targets/targets.jsonl --split-manifest data/splits/split_manifest.json --checkpoint runs/pgdn_v0/checkpoints/pgdn_v0_none.json --limit 1 --out runs
PYTHONPATH=src python3 -m pgdn.eval --ranking runs/<job>/ranking_scores.csv

# Experiment driver + report publishing
python3 scripts/run_v0_experiments.py --targets data/targets/targets.jsonl --split-manifest data/splits/split_manifest.json
python3 scripts/publish_v0_report.py --results runs/v0_experiments/results_table.csv --eval runs/pgdn_v0/eval.json --split-manifest data/splits/split_manifest.json

# Cleanup run artifacts (dry-run by default; add --apply to delete)
python3 scripts/cleanup_runs.py --runs-dir runs --keep-top-k 10 --keep-latest-checkpoints 2 --remove-empty-dirs
python3 scripts/cleanup_runs.py --runs-dir runs --keep-top-k 10 --keep-latest-checkpoints 2 --remove-empty-dirs --apply
```

## Code Style (Observed Conventions)

These conventions are inferred from existing code under `src/pgdn/` and
`scripts/`.

### Python Version / Typing

- Write modern Python (3.10+ style): use `Path | None` and `list[str]` when
  editing files that already use that style (e.g. `scripts/organize_datasets.py`).
- When editing older-style files that use `typing.List`/`typing.Dict`, keep the
  local style consistent instead of rewriting everything.
- Prefer explicit return types (`-> None`, `-> float`, etc.).
- Use `typing.cast()` only when needed (already used in `src/pgdn/model.py`).

### Imports

- Group imports: standard library, then third-party, then local (`from .foo ...`).
- Keep a single blank line between import groups.

### Formatting

- 4-space indentation.
- F-strings for user-facing logs (`print(f"wrote={path}")`).
- Keep line lengths reasonable; no strict formatter config is present.

### Naming

- Modules/files: `snake_case.py`.
- Functions/methods/variables: `snake_case`.
- Classes: `PascalCase`.
- Private helpers: prefix `_` (e.g. `_mean_vector`, `_delete_path`).

### Error Handling / Guard Rails

- Prefer guard clauses for empty/degenerate inputs (common pattern:
  `if not xs: return 0.0`).
- Avoid division-by-zero with `max(denom, 1)` or explicit checks.
- For file/OS operations where failure is expected, use narrow `try/except`
  around the failing call; do not swallow exceptions silently.
- For safety-critical filesystem actions, fail loudly (see
  `scripts/organize_datasets.py` refusing to replace a real directory).

### I/O

- Use `pathlib.Path` for filesystem work.
- Always specify `encoding="utf-8"`.
- JSON writing uses `ensure_ascii=False` and usually `indent=2`.

## Working With Submodules

- Submodule directories are upstream code. Avoid edits unless the task is
  explicitly about the submodule.
- If you need to reference submodule APIs, prefer wrappers in this repo.

## Cursor / Copilot Rules

No Cursor rules were found in `.cursor/rules/` or `.cursorrules`. No Copilot
instructions were found at `.github/copilot-instructions.md`.

## If You Add Tooling (Only If Asked)

If explicitly asked to add lint/tests, keep it minimal: `pyproject.toml` with
`ruff` + `pytest`, plus a tiny `tests/` suite. Single test example:
`pytest -q tests/test_x.py::test_y`.
