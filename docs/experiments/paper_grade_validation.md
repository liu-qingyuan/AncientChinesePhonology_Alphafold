# Paper-Grade Validation Run (Five-Seed Closure)

## Goal

Validate the group coupling effect (`shared_denoise` vs `none`) at paper-grade scale
using five random seeds (42-46) to confirm the consistency improvement is robust
and not a one-off result.

Metric used:
- `character_consistency_mean` from infer samples (cosine distance, lower is better)

## Setup

- Checkpoints: `runs/paper_grade_closure/seed{42,43,44,45,46}_train/checkpoint_none.pt`
- Split: `dev` (`--split-strategy random`)
- Scale: `--limit 256 --samples 4 --num-steps 10 --batch-size 32 --batching graph`
- Modes: `none`, `shared_noise`, `shared_denoise`
- Seeds: `42`, `43`, `44`, `45`, `46`

## Results

| Mode | character_consistency_mean ± std |
|---|---:|
| none | 0.9217 ± 0.0010 |
| shared_noise | 0.9143 ± 0.0010 |
| shared_denoise | 0.9137 ± 0.0011 |

Other metrics (max_abs, constraint_penalty_mean, ranking_score_mean, uncertainty_mean)
show negligible differences across modes (< 0.1% relative change).

## Conclusion

- Both coupling modes improve character-level consistency vs `none`.
- `shared_denoise` yields the lowest (best) mean, followed by `shared_noise`.
- The result is stable across all five seeds, confirming robustness.
- The improvement magnitude (~0.8% absolute reduction) is consistent with prior
  three-seed benchmark in `coupling.md`.

## Evidence Paths

- Aggregated summary: `runs/paper_grade_closure/aggregate/summary.json`
- Per-seed data: `runs/paper_grade_closure/aggregate/summary.csv`

Raw run outputs (seeds 42-46, three modes each = 15 runs):
- `runs/paper_grade_closure/seed42_bench/none/infer/metrics/character_consistency/by_character.json`
- `runs/paper_grade_closure/seed42_bench/shared_noise/infer/metrics/character_consistency/by_character.json`
- `runs/paper_grade_closure/seed42_bench/shared_denoise/infer/metrics/character_consistency/by_character.json`
- `runs/paper_grade_closure/seed43_bench/summary.json`
- `runs/paper_grade_closure/seed44_bench/summary.json`
- `runs/paper_grade_closure/seed45_bench/summary.json`
- `runs/paper_grade_closure/seed46_bench/summary.json`
