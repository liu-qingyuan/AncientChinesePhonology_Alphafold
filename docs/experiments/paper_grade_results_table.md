# Paper-Grade Results Table (Five-Seed Aggregate)

**Seeds:** 42, 43, 44, 45, 46  
**Modes:** `none`, `shared_noise`, `shared_denoise`

> **Note:** Lower `character_consistency_mean` is better (cosine distance).

| Mode | character_consistency_mean | ranking_score_mean | constraint_penalty_mean | max_abs | uncertainty_mean |
|---|---:|---:|---:|---:|---:|
| none | 0.9217 ± 0.0010 | 0.5705 ± 0.0002 | 0.1356 ± 0.0003 | 0.9914 ± 0.0001 | 0.2623 ± 0.0002 |
| shared_noise | 0.9143 ± 0.0010 | 0.5705 ± 0.0002 | 0.1356 ± 0.0003 | 0.9914 ± 0.0001 | 0.2623 ± 0.0002 |
| shared_denoise | 0.9137 ± 0.0011 | 0.5705 ± 0.0003 | 0.1357 ± 0.0004 | 0.9914 ± 0.0001 | 0.2623 ± 0.0002 |

**Key Takeaway:** `shared_denoise` achieves the lowest (best) character consistency (0.9137 ± 0.0011), outperforming `none` (0.9217 ± 0.0010) by ~0.8 percentage points, confirming that sharing denoising across character groups improves reconstruction consistency.

**Source:** `runs/paper_grade_closure/aggregate/summary.json`
