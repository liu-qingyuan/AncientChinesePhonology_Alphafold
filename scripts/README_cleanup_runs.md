# Run Artifact Retention (Paper Workflow)

Use `cleanup_runs.py` to keep reproducibility-critical files while pruning bulk sample artifacts.

## Retention Policy

- Keep aggregate reproducibility files:
  - `ranking_scores.csv`
  - `eval.json`
  - `results_table.csv`
  - cleanup manifests under `runs/retention_manifests/`
- Keep top-K record outputs per experiment by `ranking_score`
- Keep latest N checkpoint files
- Optionally remove empty directories

## Recommended Usage

Dry-run first (safe):

```bash
python3 scripts/cleanup_runs.py \
  --runs-dir runs \
  --keep-top-k 10 \
  --keep-latest-checkpoints 2 \
  --remove-empty-dirs
```

Apply cleanup:

```bash
python3 scripts/cleanup_runs.py \
  --runs-dir runs \
  --keep-top-k 10 \
  --keep-latest-checkpoints 2 \
  --remove-empty-dirs \
  --apply
```

The script writes a machine-readable manifest for each run:

- `runs/retention_manifests/cleanup_<timestamp>.json`

This manifest is the audit trail for what was kept and deleted.
