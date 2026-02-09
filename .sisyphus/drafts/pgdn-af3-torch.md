# Draft: PGDN (AF3-style) Torch Route

## User Decision
- Implement route (2): AF3/PGDN direction (Pairformer + Diffusion + constraints) rather than the GTenhanced MLM transformer route.

## Hard Constraints (Must Respect)
- MUST NOT modify:
  - `data/targets/acp_targets.jsonl`
  - `src/pgdn/confidence.py`
  - `src/pgdn/infer.py`
- Do not modify upstream submodules unless explicitly asked.
- Do not commit external datasets or big binaries.

## Dataset Organization Decision
- Code must treat `data/external/links/<key>` as the only stable external dataset entrypoint.
- Downloaded datasets live under `data/external/<dataset>/` and remain gitignored.
- Submodules remain at repo root and are mapped into the hub.

## Paper References Noted
- `paper/西夏语预测模型更新方案.txt` proposes PGDN: heterogeneous graph embedder, Pairformer-like transitivity updates, diffusion decoder, and optional physiological constraints using EMA datasets (Mocha-TIMIT) and transcription standardization via CLTS.
- `paper/寻找开放语言数据集.txt` emphasizes building an open dataset ecosystem and using Mocha-TIMIT/CLTS as key components.

## Implemented / In-Progress (Engineering Substrate)
- Dataset hub scripts exist and should be used for all code.
- Mocha-TIMIT Phase 1 acquisition script exists (audit artifacts + idempotency).
- CLTS detection should use explicit path (no blind download).

## Open Items
- Full research-grade PGDN graph construction and EMA-driven constraints are not yet implemented; initial focus is torch scaffolding and verifiable end-to-end run.
