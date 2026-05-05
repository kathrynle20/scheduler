# Benchmark Suite Comparison

**Suite:** `suite-20260505-000326`  
**Trials per config:** 30  
**Cluster:** read from each run's NVML monitor output

Format: `mean ± std` across trials. Δ column is the percent change of
work stealing relative to baseline. Lower is better for latency, queue
wait, util std, and max imbalance. Avg request time should be invariant
(it is GPU compute, not affected by the scheduler).

## Experiment 1: PTQ inference (100 jobs)


| Metric                 | Baseline       | Work Stealing   | Δ                  |
| ---------------------- | -------------- | --------------- | ------------------ |
| Avg job latency (ms)   | 1414.8 ± 140.1 | 1602.6 ± 490.0  | ↑ 13.3% *worse*    |
| p50 job latency (ms)   | 1116.7 ± 150.7 | 1260.3 ± 358.8  | ↑ 12.9% *worse*    |
| p95 job latency (ms)   | 2701.4 ± 501.1 | 3308.8 ± 1964.7 | ↑ 22.5% *worse*    |
| p99 job latency (ms)   | 3180.1 ± 597.5 | 3803.0 ± 2080.8 | ↑ 19.6% *worse*    |
| Avg queue wait (ms)    | 484.3 ± 138.1  | 668.8 ± 486.5   | ↑ 38.1% *worse*    |
| Avg request time (ms)  | 46.52 ± 0.20   | 46.69 ± 0.31    | ≈ same             |
| Cluster util mean (%)  | 67.3 ± 5.6     | 67.5 ± 6.1      | ≈ same             |
| Cluster util std       | 2.96 ± 1.26    | 1.72 ± 1.09     | ↓ 42.1% **better** |
| Max util imbalance (%) | 5.9 ± 2.5      | 3.4 ± 2.2       | ↓ 42.1% **better** |
| Avg temperature (C)    | 54.9 ± 1.3     | 55.3 ± 1.5      | ≈ same             |
| Steal count            | —              | 19 ± 5          | —                  |


## Experiment 2: Training (50 jobs, 500 steps)


| Metric                    | Baseline    | Work Stealing | Δ               |
| ------------------------- | ----------- | ------------- | --------------- |
| Training avg duration (s) | 1.07 ± 0.00 | 1.07 ± 0.00   | ≈ same          |
| Cluster util mean (%)     | 71.6 ± 1.0  | 71.7 ± 0.6    | ≈ same          |
| Cluster util std          | 0.36 ± 0.30 | 0.41 ± 0.29   | ↑ 15.0% *worse* |
| Max util imbalance (%)    | 0.7 ± 0.6   | 0.8 ± 0.6     | ↑ 15.0% *worse* |
| Avg temperature (C)       | 53.4 ± 0.3  | 53.2 ± 0.2    | ≈ same          |
| Steal count               | —           | 0 ± 0         | —               |


## Experiment 3: Mixed (100 PTQ + 50 training)


| Metric                    | Baseline        | Work Stealing  | Δ                  |
| ------------------------- | --------------- | -------------- | ------------------ |
| Avg job latency (ms)      | 1721.6 ± 310.6  | 1828.0 ± 362.9 | ↑ 6.2% *worse*     |
| p50 job latency (ms)      | 1389.8 ± 336.8  | 1546.0 ± 435.9 | ↑ 11.2% *worse*    |
| p95 job latency (ms)      | 3552.1 ± 905.6  | 3751.8 ± 855.6 | ↑ 5.6% *worse*     |
| p99 job latency (ms)      | 4098.2 ± 1047.4 | 4551.4 ± 814.6 | ↑ 11.1% *worse*    |
| Avg queue wait (ms)       | 794.8 ± 306.9   | 897.1 ± 359.7  | ↑ 12.9% *worse*    |
| Avg request time (ms)     | 46.25 ± 0.24    | 46.45 ± 0.23   | ≈ same             |
| Training avg duration (s) | 1.07 ± 0.00     | 1.07 ± 0.00    | ≈ same             |
| Cluster util mean (%)     | 67.3 ± 5.3      | 69.9 ± 4.9     | ↑ 3.8%             |
| Cluster util std          | 2.38 ± 1.11     | 1.37 ± 1.10    | ↓ 42.5% **better** |
| Max util imbalance (%)    | 4.8 ± 2.2       | 2.7 ± 2.2      | ↓ 42.5% **better** |
| Avg temperature (C)       | 54.3 ± 1.2      | 55.4 ± 1.2     | ↑ 1.9%             |
| Steal count               | —               | 28 ± 8         | —                  |


