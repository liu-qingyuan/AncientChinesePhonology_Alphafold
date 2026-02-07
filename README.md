# AncientChinesePhonology_Alphafold

This is a meta-repository for experiments around Ancient Chinese phonology modeling.

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
