# Benchmark Suite Comparison

**Suite:** `suite-20260504-212353`  
**Trials per config:** 5  
**Cluster:** read from each run's NVML monitor output

Format: `mean ± std` across trials. Δ column is the percent change of
work stealing relative to baseline. Lower is better for latency, queue
wait, util std, and max imbalance. Avg request time should be invariant
(it is GPU compute, not affected by the scheduler).

## Experiment 1: PTQ inference (100 jobs)

| Metric | Baseline | Work Stealing | Δ |
|--------|----------|---------------|---|
| Avg job latency (ms) | 1495.8 ± 173.2 | 1488.9 ± 376.7 | ≈ same |
| p50 job latency (ms) | 1286.8 ± 254.2 | 1196.3 ± 349.5 | ↓ 7.0% **better** |
| p95 job latency (ms) | 2803.7 ± 499.3 | 2800.7 ± 771.7 | ≈ same |
| p99 job latency (ms) | 3503.6 ± 1294.4 | 3417.6 ± 844.9 | ↓ 2.5% **better** |
| Avg queue wait (ms) | 563.4 ± 170.8 | 558.8 ± 371.1 | ≈ same |
| Avg request time (ms) | 46.62 ± 0.56 | 46.50 ± 0.38 | ≈ same |
| Cluster util mean (%) | 75.3 ± 5.9 | 66.0 ± 8.8 | ↓ 12.2% |
| Cluster util std | 3.24 ± 0.94 | 1.45 ± 1.34 | ↓ 55.3% **better** |
| Max util imbalance (%) | 6.5 ± 1.9 | 2.9 ± 2.7 | ↓ 55.3% **better** |
| Avg temperature (C) | 55.5 ± 2.9 | 54.7 ± 1.7 | ↓ 1.3% |
| Steal count | — | 19 ± 6 | — |

## Experiment 2: Training (50 jobs, 500 steps)

_No data for either config._

## Experiment 3: Mixed (100 PTQ + 50 training)

_No data for either config._
