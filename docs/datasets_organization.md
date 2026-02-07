# Dataset Organization

This workspace now uses a single, stable dataset hub at:

- `data/external/links/`

Each item in `data/external/links/` is a symlink to the real dataset location.
This avoids moving large repos and keeps existing paths intact.

## One-command refresh

```bash
python3 scripts/organize_datasets.py
```

## What this command does

- Scans known datasets: `PHOIBLE`, `CLTS`, `WikiHan`, `WikiPron`, `Mocha-TIMIT`
- Creates/updates symlinks under `data/external/links/`
- Writes machine-readable status to `data/external/dataset_registry.json`

## Current layout contract

- `data/external/links/phoible` -> dataset root (if found)
- `data/external/links/clts` -> dataset root (if found)
- `data/external/links/wikihan` -> dataset root (if found)
- `data/external/links/wikipron` -> dataset root (if found)
- `data/external/links/mocha_timit` -> Mocha-TIMIT root (if found)

## Notes

- Missing datasets are marked as `"status": "missing"` in `dataset_registry.json`.
- This step does not modify ACP contracts or training/inference code.
