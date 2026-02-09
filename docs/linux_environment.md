# Linux Environment (Reproducible)

This repo does not ship a packaged Python project (`pyproject.toml`/`setup.py`).
Most entrypoints assume running from repo root and (for `pgdn`) setting
`PYTHONPATH=src`.

Goal: a reproducible environment that works on Linux with optional GPU.

## Option A (Recommended): Conda

Conda is present on many machines running this repo.

```bash
conda create -n acpa python=3.12 -y
conda activate acpa

# Minimal tooling
python -m pip install -U pip

# PyTorch (pick ONE)
# CPU-only:
#   python -m pip install torch
# CUDA (use the official selector for your CUDA version):
#   https://pytorch.org/get-started/locally/
```

Sanity checks:

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
PYTHONPATH=src python3 scripts/smoke_model_interfaces.py
```

## Option B: venv (no conda)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install torch
```

## Repo Bootstrap

```bash
git submodule update --init --recursive
python3 scripts/organize_datasets.py
```

## Public-Safe Policy

- Never commit external datasets or large binaries.
- This repo already `.gitignore`s generated artifacts under `runs/`, `reports/`,
  `data/targets/`, and `data/external/mocha_timit/`.
- Treat git submodules as read-only upstream code unless a task explicitly asks
  to change them.
