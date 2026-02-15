# Group Coupling Benchmark (Task 11)

## Goal

Test whether group coupling (`shared_noise`, `shared_denoise`) improves
character-level consistency compared with `none`.

Metric used:
- `character_consistency` from infer samples
- cosine distance (`1 - cos`), lower is better

## Setup

- Checkpoint: `runs/pgdnv0_paper_align_smoke/checkpoint_none.pt`
- Split: `dev` (`--split-strategy random`)
- Scale: `--limit 2048 --samples 8 --num-steps 10 --batch-size 32 --batching graph`
- Modes: `none`, `shared_noise`, `shared_denoise`
- Seeds: `7`, `17`, `27`

## Results (character_consistency_mean)

| Seed | none | shared_noise | shared_denoise | Delta(shared_noise-none) | Delta(shared_denoise-none) |
|---|---:|---:|---:|---:|---:|
| 7  | 0.887099 | 0.498110 | 0.463747 | -0.388990 | -0.423353 |
| 17 | 0.898038 | 0.498804 | 0.468848 | -0.399234 | -0.429190 |
| 27 | 0.901213 | 0.484488 | 0.453138 | -0.416725 | -0.448075 |

## Conclusion

- Both coupling modes consistently improve character-level consistency vs `none`.
- `shared_denoise` is consistently best across all three seeds.
- The result is stable across random seeds, not a one-off run.

## Evidence Paths

- `runs/bench_group_coupling_limit2048_s8/summary.csv`
- `runs/bench_group_coupling_limit2048_s8_seed17/summary.csv`
- `runs/bench_group_coupling_limit2048_s8_seed27/summary.csv`

Detailed per-character outputs:
- `runs/bench_group_coupling_limit2048_s8/none/infer/metrics/character_consistency/by_character.json`
- `runs/bench_group_coupling_limit2048_s8/shared_noise/infer/metrics/character_consistency/by_character.json`
- `runs/bench_group_coupling_limit2048_s8/shared_denoise/infer/metrics/character_consistency/by_character.json`
