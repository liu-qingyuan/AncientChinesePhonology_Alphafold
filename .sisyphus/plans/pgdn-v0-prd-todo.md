# PGDN v0 PRD TODO List (ACP-first)

## TL;DR

> **Quick Summary**: Build a runnable PGDN v0 using ACP-first data, with AF3-inspired module boundaries and output protocol, plus strict low-resource evaluation.
>
> **Deliverables**:
> - ACP-first end-to-end data/train/infer pipeline
> - Multi-sample candidate generation with ranking/confidence
> - Baseline + ablation report under fixed split policy
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Data contract -> MVP model path -> Evaluation/ranking report

---

## Context

### Original Request
Review and revise `PGDN_IMPLEMENTATION_PLAN.md` using `paper/` materials, then provide a practical PRD-style TODO markdown for progress tracking.

### Confirmed Decisions
- v0 strategy: **ACP-first**
- Diffusion backbone: **`lucidrains/denoising-diffusion-pytorch`**
- AF3 usage mode: **reference architecture/protocol only, not direct diffusion code transplant**

### Metis Gap Review Incorporated
- Add strict MVP in/out boundary to prevent scope creep
- Add explicit loss contract and split/leakage policy
- Add baseline/ablation cap and confidence calibration requirement
- Add reproducibility and risk register

---

## Work Objectives

### Core Objective
Deliver a runnable PGDN v0 that can train on ACP-derived graph data and produce top-K phonology candidates with ranking and confidence under reproducible evaluation.

### Must Have
- ACP-first pipeline fully runnable (data build -> train -> infer)
- Continuous phonological feature-vector output in v0
- Multi-sample ranking/confidence outputs with fixed schema
- Random + temporal + low-resource split reports

### Must NOT Have (v0 guardrails)
- No full Tangut-first production data workflow
- No full AF3 diffusion transplant
- No large Sim-to-Real pretraining as a blocking dependency
- No self-distillation loop as required path
- No open-ended architecture search

### Definition of Done
- [x] End-to-end run completes from raw ACP CSV to ranked inference outputs
- [x] Required output files exist and include required fields
- [x] Baseline and capped ablations are completed on fixed splits
- [x] Report shows ranking/constraint metrics and calibration diagnostics

---

## Verification Strategy (Zero Human Intervention)

### Test Decision
- **Infrastructure exists**: Unknown (executor checks at start)
- **Automated tests**: Tests-after (v0 recommendation)
- **Framework**: Follow repository default if present; otherwise minimal Python test setup

### Agent-Executed QA Principle
All acceptance criteria below are command/tool verifiable by agent. No manual visual/user confirmation is required.

---

## Execution Strategy

### Parallel Waves

Wave 1 (foundation): Tasks 1, 2
Wave 2 (model core): Tasks 3, 4
Wave 3 (eval/report): Tasks 5, 6, 7

Critical Path: 1 -> 3 -> 5 -> 7

---

## TODOs

- [x] 1. Lock v0 data contract and split policy

  **What to do**:
  - Define ACP-first JSONL contract fields: `language`, `period`, `syllable_slots(I/M/N/C)`, masks
  - Freeze split protocol: random + temporal + low-resource(10/5/1/0%)
  - Add leakage-prevention rules in plan text

  **Must NOT do**:
  - Do not introduce Tangut-only mandatory fields as blockers

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
  - **Skills**: `writing`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: 3, 5
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [x] `PGDN_IMPLEMENTATION_PLAN.md` contains section `MVP Scope (v0)`
  - [x] `PGDN_IMPLEMENTATION_PLAN.md` contains section `Data Split Protocol`
  - [x] `PGDN_IMPLEMENTATION_PLAN.md` contains section `Leakage Prevention Rules`

- [x] 2. Define output schema and ranking/confidence semantics

  **What to do**:
  - Standardize output fields for `summary.json` and `ranking_scores.csv`
  - Define language-localized ranking formula and calibration note

  **Must NOT do**:
  - Do not use AF3 biomolecular metrics without linguistic mapping

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
  - **Skills**: `writing`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: 4, 7
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [x] Plan includes section `Output Protocol (Ranking and Confidence)`
  - [x] `summary.json` required fields are explicitly listed
  - [x] Ranking score components (`constraint_satisfaction`, `sample_consistency`, penalty terms, uncertainty) are documented

- [x] 3. Implement ACP-first data build pipeline

  **What to do**:
  - Convert ACP CSV into graph/target JSONL format per contract
  - Ensure masks and target-level consistency checks

  **Must NOT do**:
  - Do not add full multilingual harmonization in v0

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `ultrabrain`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2
  - **Blocks**: 4, 5
  - **Blocked By**: 1

  **Acceptance Criteria**:
  - [x] JSONL files are generated in the agreed directories
  - [x] A validation command confirms required keys for all records
  - [x] Data build script exits with code 0

- [x] 4. Build PGDN v0 model path (Embedder -> Pairformer-lite -> Diffusion)

  **What to do**:
  - Implement minimal module interfaces and tensor contracts
  - Integrate lucidrains diffusion training loop with linguistic conditioning
  - Keep AF3 only as interface/design reference

  **Must NOT do**:
  - Do not transplant AF3 diffusion source directly
  - Do not introduce recycling as a mandatory v0 feature

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `ultrabrain`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2
  - **Blocks**: 5, 6
  - **Blocked By**: 2

  **Acceptance Criteria**:
  - [x] Training command runs and logs loss components
  - [x] Inference command emits multi-sample outputs for at least one seed
  - [x] Module interface shapes are validated in a smoke test

- [x] 5. Add baseline and capped ablations

  **What to do**:
  - Baseline: Transformer(glyph+time) ACP-style
  - Ablations (<=3): no-pairformer, no-diffusion, no-constraint-loss

  **Must NOT do**:
  - Do not exceed ablation cap in v0

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `ultrabrain`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: 7
  - **Blocked By**: 3, 4

  **Acceptance Criteria**:
  - [x] Baseline run artifacts exist
  - [x] Each ablation has metrics on frozen split(s)
  - [x] Results table includes mean/std across seeds

- [x] 6. Evaluate calibration and ranking robustness

  **What to do**:
  - Compute top-K coverage, constraint satisfaction, ranking correlation
  - Add uncertainty calibration diagnostic

  **Must NOT do**:
  - Do not report confidence as probability without calibration disclaimer

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `ultrabrain`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: 7
  - **Blocked By**: 4

  **Acceptance Criteria**:
  - [x] Evaluation output includes top-K coverage and ranking-correlation
  - [x] Calibration metric is present and documented
  - [x] Constraint violation breakdown is present

- [x] 7. Publish v0 report and reproducibility manifest

  **What to do**:
  - Produce consolidated v0 report with methods, splits, metrics, errors
  - Include seed/config hash, dependency versions, and known risks

  **Must NOT do**:
  - Do not leave unresolved placeholders (e.g., TBD)

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `writing`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 final
  - **Blocks**: None
  - **Blocked By**: 5, 6

  **Acceptance Criteria**:
  - [x] Report includes `Risk Register` and `Deferred to v0.2+`
  - [x] Reproducibility section includes seeds/config/dependency lock
  - [x] Final checklist has no unchecked blocker

---

## Risk Register (v0)

1. Diffusion backbone adaptation mismatch (image-first assumptions)
2. Split leakage in historical data organization
3. Uncalibrated confidence interpreted as reliability
4. Scope creep into Tangut-first/data expansion

Mitigation is mandatory per TODO acceptance criteria.

---

## Deferred to v0.2+

1. Tangut-first full graph production pipeline
2. Sim-to-Real large synthetic pretraining
3. Recycling/self-distillation production loops
4. Advanced hypothesis-testing engine for scholarly controversies

---

## Progress Tracking Template

Update this table after each completed task:

| Task | Status | Owner/Agent | Date | Evidence Path |
|------|--------|-------------|------|---------------|
| 1 | DONE | Sisyphus | 2026-02-06 | PGDN_IMPLEMENTATION_PLAN.md |
| 2 | DONE | Sisyphus | 2026-02-06 | PGDN_IMPLEMENTATION_PLAN.md |
| 3 | DONE | Sisyphus | 2026-02-06 | data/graphs/acp_graphs.jsonl |
| 4 | DONE | Sisyphus | 2026-02-06 | src/pgdn/model.py |
| 5 | DONE | Sisyphus | 2026-02-06 | runs/v0_experiments/results_table.csv |
| 6 | DONE | Sisyphus | 2026-02-06 | runs/pgdn_v0_main/eval.json |
| 7 | DONE | Sisyphus | 2026-02-06 | reports/v0/PGDN_v0_report.md |
