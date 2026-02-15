# Dataset Unification Roadmap (v1)

This document defines the migration plan from the current ACP-centric v0 contracts to a unified schema (v1) that accommodates multiple phonological datasets while protecting existing ACP functionality.

## Schema Version

- **Current version**: v0 (ACP-only)
- **Target version**: v1
- **Rationale**: v1 introduces a common record schema that maps heterogeneous sources (ACP, Mocha, tshet-uinh, Unihan, CLTS, WikiHan, WikiPron) into a unified target representation compatible with `pgdn_torch` train/infer/eval pipelines.

### Phase 1 Frozen Contract (Task 1)

- Strict Phase 1 row contract for new sidecars: `record_id`, `graph_id`, `target_vector`, `mask`, `source`.
- Strict namespace rule for new sidecars: `record_id` must start with `acp:`, `mocha:`, `tshet:`, or `unihan:`.
- ACP compatibility rule (deterministic): legacy ACP rows are allowed when `source` is missing and `record_id` matches legacy `<character>:<period>` shape; validator treats them as implicit source `acp` in compatibility mode.
- Split manifests must remain backward-compatible with `src/pgdn_torch/pgdnv0/splits.py`:
  - top-level strategies include `random` and `temporal`
  - each strategy maps `train`/`dev`/`test` to lists of IDs
- Validator command:

```bash
PYTHONPATH=src python3 scripts/validate_phase1_contract.py --sample-size 32
```

The validator checks sampled rows in ACP/Mocha/new sidecar target files when present, applies ACP-legacy compatibility by default (`--acp-legacy-compat`), enforces strict keys + namespace on non-ACP sidecars, validates split schema shape, and exits non-zero on contract violations.

## Mapping Table

| Source | Current Status | Target Fields | Blockers |
|--------|----------------|---------------|----------|
| ACP | Active (v0 contract) | `record_id`, `character`, `language`, `period`, `target_vector`, `mask`, `graph_id`, `normalized_pron`, `syllable_slots` | None (protected) |
| Mocha-TIMIT | Sidecar v1 passing smoke | `record_id`, `graph_id`, `target_vector`, `mask`, `source=mocha_timit_sidecar_v1`, `speaker`, `stem`, `paths` | None |
| tshet-uinh | Not integrated | `record_id`, `character`, `language`, `target_vector`, `mask`, `graph_id`, `source=tshet_uinh` | Need CSV/TSV parser, phoneme inventory alignment |
| Unihan | Not integrated | `record_id`, `character`, `cantonese`, `mandarin`, `target_vector`, `mask`, `source=unihan` | Need downloader, field selection |
| CLTS | Linked but not consumed | `record_id`, `graph_id`, `target_vector`, `mask`, `source=clts`, `clts_id`, `phoneme_features` | Need mapping from CLTS segments to vector space |
| WikiHan | Linked but not consumed | `record_id`, `character`, `language`, `target_vector`, `mask`, `graph_id`, `source=wikihan` | Need IPA parsing, language metadata |
| WikiPron | Linked but not consumed | `record_id`, `character`, `language`, `pronunciation`, `target_vector`, `mask`, `graph_id`, `source=wikipron` | Need IPA normalization |

**Protected v0 contracts (must remain untouched)**:
- `data/targets/acp_targets.jsonl`
- `src/pgdn/confidence.py`
- `src/pgdn/infer.py`

## Migration Order

1. **Phase 1** (Safe): Verify Mocha sidecar v1 integration is stable under `pgdn_torch` full runs (not just smoke). Confirm no regressions on ACP baseline.
2. **Phase 2** (Low risk): Add tshet-uinh parser and generate sidecar JSONL. Test graph construction with mixed ACP+tshet-uinh graphs.
3. **Phase 3** (Medium risk): Add Unihan and CLTS mapping layers. Ensure `source` field distinguishes origin in all downstream metrics.
4. **Phase 4** (Medium risk): Add WikiHan and WikiPron parsers. Align IPA normalization with existing CLTS pipeline.
5. **Phase 5** (Validation): Run unified eval on all sources. Verify `character_consistency`, `ranking_score`, and `uncertainty` metrics are meaningful across heterogeneous data.

## Rollback

- **Trigger conditions**:
  - Any new source causes ACP baseline metric regression > 5% on `character_consistency_mean`.
  - `pgdn_torch` train/infer crashes on protected v0 contract files.
  - Split leakage detected between ACP and new sources (shared characters appearing in train and eval).

- **Rollback procedure**:
  1. Revert to last known-good commit.
  2. Disable new source in config (do not delete code).
  3. Re-run ACP-only smoke to confirm baseline.
  4. Document regression in `.sisyphus/notepads/unified-pgdn-phase2-roadmap/issues.md`.

- **Rollback is NOT required** for:
  - New source failing to load gracefully (handled by try/except with warning).
  - Metric variance within normal noise (< 5% relative).
  - Missing fields that are optional in v1 schema.
