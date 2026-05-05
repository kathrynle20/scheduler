# Benchmark Suite Comparison

**Suite:** `suite-20260505-035314`  
**Trials per config:** 50  
**Cluster:** read from each run's NVML monitor output

Format: `mean ± std` across trials. Δ column is the percent change of
work stealing relative to baseline. Lower is better for latency, queue
wait, util std, and max imbalance. Avg request time should be invariant
(it is GPU compute, not affected by the scheduler).

## Experiment 1: PTQ inference (100 jobs)

| Metric | Baseline | Work Stealing | Δ |
|--------|----------|---------------|---|
| Avg job latency (ms) | 1503.4 ± 265.4 | 1595.2 ± 374.4 | ↑ 6.1% *worse* |
| p50 job latency (ms) | 1217.3 ± 285.4 | 1289.1 ± 355.5 | ↑ 5.9% *worse* |
| p95 job latency (ms) | 2895.2 ± 704.9 | 3061.7 ± 982.3 | ↑ 5.8% *worse* |
| p99 job latency (ms) | 3345.8 ± 773.0 | 3611.3 ± 1007.0 | ↑ 7.9% *worse* |
| Avg queue wait (ms) | 572.9 ± 261.6 | 663.5 ± 369.7 | ↑ 15.8% *worse* |
| Avg request time (ms) | 46.52 ± 0.28 | 46.58 ± 0.32 | ≈ same |
| Cluster util mean (%) | 67.8 ± 6.0 | 69.5 ± 6.8 | ↑ 2.5% |
| Cluster util std | 2.97 ± 1.32 | 1.60 ± 1.35 | ↓ 46.1% **better** |
| Max util imbalance (%) | 5.9 ± 2.6 | 3.2 ± 2.7 | ↓ 46.1% **better** |
| Avg temperature (C) | 54.7 ± 1.5 | 54.6 ± 1.6 | ≈ same |
| Steal count | — | 20 ± 6 | — |

## Experiment 2: Training (50 jobs, 500 steps)

| Metric | Baseline | Work Stealing | Δ |
|--------|----------|---------------|---|
| Training avg duration (s) | 1.07 ± 0.00 | 1.07 ± 0.00 | ≈ same |
| Cluster util mean (%) | 71.5 ± 1.1 | 69.5 ± 1.1 | ↓ 2.8% |
| Cluster util std | 0.44 ± 0.34 | 0.51 ± 0.39 | ↑ 14.9% *worse* |
| Max util imbalance (%) | 0.9 ± 0.7 | 1.0 ± 0.8 | ↑ 14.9% *worse* |
| Avg temperature (C) | 51.8 ± 0.2 | 52.7 ± 0.7 | ↑ 1.8% |
| Steal count | — | 0 ± 0 | — |

## Experiment 3: Mixed (100 PTQ + 50 training)

| Metric | Baseline | Work Stealing | Δ |
|--------|----------|---------------|---|
| Avg job latency (ms) | 2005.9 ± 441.9 | 1901.1 ± 399.4 | ↓ 5.2% **better** |
| p50 job latency (ms) | 1689.1 ± 431.3 | 1530.4 ± 447.5 | ↓ 9.4% **better** |
| p95 job latency (ms) | 4075.1 ± 1139.8 | 4049.9 ± 1009.6 | ≈ same |
| p99 job latency (ms) | 4631.2 ± 1159.2 | 4821.0 ± 1191.2 | ↑ 4.1% *worse* |
| Avg queue wait (ms) | 1072.6 ± 438.4 | 967.6 ± 395.9 | ↓ 9.8% **better** |
| Avg request time (ms) | 46.57 ± 0.23 | 46.58 ± 0.26 | ≈ same |
| Training avg duration (s) | 1.07 ± 0.00 | 1.07 ± 0.00 | ≈ same |
| Cluster util mean (%) | 71.2 ± 5.0 | 70.4 ± 5.7 | ↓ 1.2% |
| Cluster util std | 1.84 ± 1.20 | 1.64 ± 0.93 | ↓ 10.9% **better** |
| Max util imbalance (%) | 3.7 ± 2.4 | 3.3 ± 1.9 | ↓ 10.9% **better** |
| Avg temperature (C) | 56.0 ± 1.2 | 55.8 ± 1.4 | ≈ same |
| Steal count | — | 29 ± 7 | — |
