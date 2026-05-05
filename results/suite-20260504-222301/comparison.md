# Benchmark Suite Comparison

**Suite:** `suite-20260504-222301`  
**Trials per config:** 3  
**Cluster:** read from each run's NVML monitor output

Format: `mean ± std` across trials. Δ column is the percent change of
work stealing relative to baseline. Lower is better for latency, queue
wait, util std, and max imbalance. Avg request time should be invariant
(it is GPU compute, not affected by the scheduler).

## Experiment 1: PTQ inference (100 jobs)


| Metric                 | Baseline        | Work Stealing  | Δ                  |
| ---------------------- | --------------- | -------------- | ------------------ |
| Avg job latency (ms)   | 1464.8 ± 145.3  | 1766.9 ± 223.7 | ↑ 20.6% *worse*    |
| p50 job latency (ms)   | 1100.1 ± 221.8  | 1392.6 ± 301.1 | ↑ 26.6% *worse*    |
| p95 job latency (ms)   | 3373.1 ± 1204.4 | 3534.0 ± 523.1 | ↑ 4.8% *worse*     |
| p99 job latency (ms)   | 3839.2 ± 1481.1 | 4224.6 ± 313.2 | ↑ 10.0% *worse*    |
| Avg queue wait (ms)    | 542.5 ± 145.9   | 831.1 ± 220.6  | ↑ 53.2% *worse*    |
| Avg request time (ms)  | 46.12 ± 0.59    | 46.79 ± 0.19   | ↑ 1.5%             |
| Cluster util mean (%)  | 67.0 ± 7.6      | 71.4 ± 5.1     | ↑ 6.6%             |
| Cluster util std       | 2.98 ± 0.96     | 0.95 ± 0.54    | ↓ 68.1% **better** |
| Max util imbalance (%) | 6.0 ± 1.9       | 1.9 ± 1.1      | ↓ 68.1% **better** |
| Avg temperature (C)    | 56.0 ± 4.2      | 58.7 ± 1.3     | ↑ 4.9%             |
| Steal count            | —               | 17 ± 3         | —                  |


## Experiment 2: Training (50 jobs, 500 steps)


| Metric                    | Baseline    | Work Stealing | Δ               |
| ------------------------- | ----------- | ------------- | --------------- |
| Training avg duration (s) | 1.08 ± 0.01 | 1.08 ± 0.00   | ≈ same          |
| Cluster util mean (%)     | 71.7 ± 2.8  | 73.1 ± 1.0    | ↑ 1.9%          |
| Cluster util std          | 0.45 ± 0.14 | 0.63 ± 0.83   | ↑ 39.1% *worse* |
| Max util imbalance (%)    | 0.9 ± 0.3   | 1.3 ± 1.7     | ↑ 39.1% *worse* |
| Avg temperature (C)       | 55.8 ± 0.6  | 56.2 ± 0.2    | ≈ same          |
| Steal count               | —           | 0 ± 0         | —               |


## Experiment 3: Mixed (100 PTQ + 50 training)


| Metric                    | Baseline        | Work Stealing   | Δ                  |
| ------------------------- | --------------- | --------------- | ------------------ |
| Avg job latency (ms)      | 1711.8 ± 393.3  | 1759.6 ± 480.4  | ↑ 2.8% *worse*     |
| p50 job latency (ms)      | 1430.1 ± 358.5  | 1336.5 ± 399.7  | ↓ 6.5% **better**  |
| p95 job latency (ms)      | 3523.4 ± 876.9  | 3600.8 ± 1441.7 | ↑ 2.2% *worse*     |
| p99 job latency (ms)      | 3962.1 ± 1132.2 | 4596.6 ± 1388.5 | ↑ 16.0% *worse*    |
| Avg queue wait (ms)       | 781.0 ± 390.1   | 832.6 ± 476.1   | ↑ 6.6% *worse*     |
| Avg request time (ms)     | 46.45 ± 0.18    | 46.26 ± 0.25    | ≈ same             |
| Training avg duration (s) | 1.08 ± 0.00     | 1.08 ± 0.00     | ≈ same             |
| Cluster util mean (%)     | 71.3 ± 4.1      | 67.2 ± 5.6      | ↓ 5.7%             |
| Cluster util std          | 1.78 ± 0.66     | 1.30 ± 0.65     | ↓ 27.0% **better** |
| Max util imbalance (%)    | 3.6 ± 1.3       | 2.6 ± 1.3       | ↓ 27.0% **better** |
| Avg temperature (C)       | 58.2 ± 0.8      | 57.1 ± 1.3      | ↓ 1.8%             |
| Steal count               | —               | 27 ± 7          | —                  |


