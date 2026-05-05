# Benchmark Suite Comparison

**Suite:** `suite-20260504-233504`  
**Trials per config:** 5  
**Cluster:** read from each run's NVML monitor output

Format: `mean ± std` across trials. Δ column is the percent change of
work stealing relative to baseline. Lower is better for latency, queue
wait, util std, and max imbalance. Avg request time should be invariant
(it is GPU compute, not affected by the scheduler).

## Experiment 1: PTQ inference (100 jobs)

| Metric | Baseline | Work Stealing | Δ |
|--------|----------|---------------|---|
| Avg job latency (ms) | 1062.5 ± 90.7 | 1017.1 ± 16.4 | ↓ 4.3% **better** |
| p50 job latency (ms) | 953.0 ± 10.4 | 961.4 ± 1.9 | ≈ same |
| p95 job latency (ms) | 1696.8 ± 809.5 | 1193.5 ± 122.8 | ↓ 29.7% **better** |
| p99 job latency (ms) | 2767.2 ± 1143.4 | 2248.7 ± 549.9 | ↓ 18.7% **better** |
| Avg queue wait (ms) | 145.7 ± 100.7 | 92.5 ± 15.7 | ↓ 36.5% **better** |
| Avg request time (ms) | 45.84 ± 0.56 | 46.23 ± 0.06 | ≈ same |
| Cluster util mean (%) | 32.8 ± 2.6 | 32.6 ± 1.0 | ≈ same |
| Cluster util std | 5.05 ± 1.09 | 3.94 ± 1.27 | ↓ 22.1% **better** |
| Max util imbalance (%) | 13.3 ± 3.3 | 10.1 ± 2.8 | ↓ 24.3% **better** |
| Avg temperature (C) | 48.7 ± 3.4 | 50.9 ± 0.4 | ↑ 4.6% |
| Steal count | — | 62 ± 3 | — |

## Experiment 2: Training (50 jobs, 500 steps)

_No data for either config._

## Experiment 3: Mixed (100 PTQ + 50 training)

_No data for either config._
