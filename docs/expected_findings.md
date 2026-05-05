# Expected Findings: Baseline vs. Work-Stealing Scheduler

## Research Question

> Can a work-stealing scheduler reduce inference latency and improve GPU utilization balance compared to a naive round-robin baseline on a multi-GPU cluster?

---

## Experimental Setup Recap

- **Cluster:** 2× NVIDIA L40S (44 GB VRAM) on MIT Engaging
- **Target operating region:** 50–70% GPU utilization (energy-proportional, latency-stable)
- **Arrival rate:** 1.0 jobs/sec across both GPUs → ~60% utilization per GPU
- **Three workload scenarios:** PTQ only, training only, mixed PTQ + training

### Why the baseline is expected to struggle

Round-robin assigns jobs to GPUs in strict alternation without any awareness of queue depth or current load. With stochastic job arrivals (jobs are randomly interleaved), one GPU can easily accumulate 3–4 queued jobs while the other sits idle. Those queued jobs wait behind each other unnecessarily — the idle GPU cannot help even though it has spare capacity. This is precisely the gap work stealing is designed to close.

---

## Experiment 1: 100 PTQ Inference Jobs

**Config:** `ptq_100.yaml` vs. `ws_ptq_100.yaml`
Each job: 50 matmuls on a 16384×16384 float16 tensor → ~1.2s/job on L40S.

### Expected baseline behavior

- Round-robin alternates between GPU 0 and GPU 1 regardless of queue depth
- With random arrival spacing (Poisson-like at 1.0/sec), occasional bursts cause one GPU to queue 2–3 jobs while the other is free
- Jobs stuck behind a queue wait their full turn even with an idle GPU available
- **Expected result:** p95/p99 latency noticeably higher than p50 (tail inflation from queuing)


| Metric             | Expected Baseline                                    |
| ------------------ | ---------------------------------------------------- |
| p50 latency        | ~1.2s (single job, no wait)                          |
| p95 latency        | ~2.5–3.5s (queued behind 1–2 jobs)                   |
| p99 latency        | ~3.5–5.0s (worst-case queue of 3)                    |
| GPU util (mean)    | ~60% per GPU                                         |
| Util std (balance) | Low — round-robin is perfectly balanced *on average* |
| Steal count        | N/A                                                  |


### Expected work-stealing behavior

- Jobs are initially placed by queue length + utilization score, same as baseline on average
- When a burst causes GPU 1's queue to grow to 3 while GPU 0 is idle, GPU 0 steals from GPU 1's tail
- The stolen job starts immediately on GPU 0 instead of waiting
- **Expected result:** p95/p99 latency meaningfully reduced; p50 should be similar


| Metric             | Expected Work Stealing    | Change vs. Baseline |
| ------------------ | ------------------------- | ------------------- |
| p50 latency        | ~1.2s                     | ≈ same              |
| p95 latency        | ~1.8–2.5s                 | **10–30% lower**    |
| p99 latency        | ~2.0–3.0s                 | **15–40% lower**    |
| GPU util (mean)    | ~60% per GPU              | ≈ same              |
| Util std (balance) | Similar or slightly lower | Marginally better   |
| Steal count        | 5–20 steals over 100 jobs | —                   |


**Key insight:** Work stealing helps most at the tail (p95/p99), not the median. The p50 job never had to wait anyway — the benefit shows up in the jobs that arrived during a burst.

---

## Experiment 2: 50 Training Jobs (500 steps each)

**Config:** `train_50.yaml` vs. `ws_train_50.yaml`
Each job: ~minutes of sustained GPU compute. Throughput (samples/sec) is the primary metric.

### Expected baseline behavior

- Training jobs are long — once placed, a GPU stays busy for minutes
- Round-robin distributes training jobs evenly from the start
- Very few queue imbalances because jobs are long relative to the arrival interval
- **Expected result:** Minimal difference between baseline and work stealing


| Metric                   | Expected Baseline                         |
| ------------------------ | ----------------------------------------- |
| Avg training duration    | Depends on cluster; ~5–15 min/job         |
| Throughput (samples/sec) | Steady, high — GPU always busy            |
| GPU util                 | ~95–100% (training is compute-saturating) |
| Util std                 | Very low — both GPUs equally loaded       |
| Steal count (WS)         | Near 0 — no imbalance to steal from       |


### Expected work-stealing behavior

- Steal threshold of 2 is unlikely to trigger when job duration >> inter-arrival gap
- **Expected result:** Essentially identical to baseline
- This is the *negative result* the professor mentioned — work stealing should *not* help training as much, because training is throughput-dominated and doesn't have the bursty latency sensitivity of inference

**Key insight for the paper:** This experiment validates the claim directionally. If work stealing helps PTQ but not training, it confirms the mechanism — work stealing targets queue imbalance which matters for latency-critical short jobs, not long sustained ones.

---

## Experiment 3: Mixed 100 PTQ + 50 Training Jobs

**Config:** `mixed_100_50.yaml` vs. `ws_mixed_100_50.yaml`
The most realistic and interesting scenario: inference and training jobs competing for the same GPUs.

### Expected baseline behavior

- Training jobs are heavy (high memory, long duration) and occupy a GPU for a long time
- When a training job lands on GPU 0, subsequent PTQ jobs assigned to GPU 0 by round-robin pile up behind it
- A PTQ job that could run in 1.2s instead waits 5–10 minutes behind a training job
- **Expected result:** PTQ tail latency severely inflated; high latency variance


| Metric                | Expected Baseline                                       |
| --------------------- | ------------------------------------------------------- |
| PTQ p50 latency       | ~1.2s (lucky placement)                                 |
| PTQ p99 latency       | **Very high** — PTQ jobs assigned behind training jobs  |
| Training avg duration | Normal                                                  |
| GPU util              | Uneven — GPU with training job at 100%, other GPU lower |
| Util std              | **Higher** than PTQ-only experiment                     |


### Expected work-stealing behavior

- Work stealing detects when one GPU's queue is long (training job + pending PTQ) while the other is shorter
- Steals PTQ jobs away from the overloaded GPU and runs them immediately on the lighter GPU
- **Expected result:** PTQ tail latency dramatically reduced; GPU utilization more balanced


| Metric                | Expected Work Stealing       | Change vs. Baseline                          |
| --------------------- | ---------------------------- | -------------------------------------------- |
| PTQ p50 latency       | ~1.2s                        | ≈ same                                       |
| PTQ p99 latency       | Significantly lower          | **Largest improvement of all 3 experiments** |
| Training avg duration | Similar                      | Minimal change                               |
| GPU util std          | Lower                        | **More balanced**                            |
| Steal count           | Highest of all 3 experiments | Most stealing triggered                      |


**Key insight:** This is the headline result. The mixed workload is where work stealing shows its biggest advantage — it rescues latency-sensitive PTQ jobs from queuing behind long training jobs, something round-robin cannot do.

---

## Summary Table: All Experiments


| Experiment    | Primary Metric     | Baseline                   | Work Stealing            | Expected Improvement      |
| ------------- | ------------------ | -------------------------- | ------------------------ | ------------------------- |
| PTQ only      | p95/p99 latency    | Tail inflation from bursts | Tail reduced by stealing | 10–40% lower p99          |
| Training only | Throughput         | High, stable               | Same                     | Minimal — negative result |
| Mixed         | PTQ p99 latency    | Severely inflated          | Substantially reduced    | Largest delta             |
| Mixed         | Util balance (std) | Higher imbalance           | Lower imbalance          | Measurable                |


---

## What a Successful Result Looks Like

A convincing result needs to show all three of the following:

1. **Work stealing reduces PTQ tail latency** (p95/p99) — especially in the mixed experiment
2. **Steal count is non-zero** — confirms the mechanism actually fired
3. **Work stealing does not hurt training throughput** — the benefit is targeted, not universal

If steal count is 0 in all runs, that indicates queues are never imbalanced enough — lower `steal_threshold` from 2 to 1, or increase `arrival_rate_hz` slightly to create more competition.

---

## Potential Negative Results and What They Mean


| Observation                        | Interpretation                                                                                       |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Steal count = 0                    | Jobs finish too fast / arrival rate too low to create imbalance — increase `arrival_rate_hz`         |
| p99 latency unchanged              | Stealing fires but stolen jobs are small — try more PTQ in mixed workload                            |
| Work stealing *increases* latency  | Steal overhead outweighs gain — common with very short jobs; expect this if `matrix_size` is reduced |
| Util std higher with work stealing | Queue estimates are stale — monitor sampling rate may need to increase                               |


---

## Connection to Energy Proportionality

Per the professor's guidance: GPU utilization is a proxy for power consumption. At 60–70% utilization:

- Both schedulers are in the energy-proportional region
- Work stealing keeps GPUs from idling (which wastes power with no performance gain) or from saturating (which burns power while latency degrades)
- The balanced utilization from work stealing (lower `util_std`) means the cluster is collectively operating more efficiently — the same total work done at more uniform power draw across devices

