# External Datasets Hub

Use this directory as the single entrypoint for third-party datasets.

- `links/`: symlinks to dataset locations in the workspace
- `dataset_registry.json`: machine-readable inventory with presence/size/status

Refresh inventory and links:

```bash
python3 scripts/organize_datasets.py
```
