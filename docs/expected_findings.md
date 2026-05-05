# Expected Findings: Baseline vs. Work-Stealing Scheduler

## Research Question

> Can a work-stealing scheduler reduce inference latency and improve GPU utilization balance compared to a naive round-robin baseline on a multi-GPU cluster?

---

## Understanding the Metrics

Before reading the expected results, it is important to understand what each metric measures and why it matters for this comparison.

### Per-request GPU compute time (`PTQ avg request time`)
Measures how long a single matmul takes on the GPU — from kernel launch to `synchronize()`. This is determined entirely by the GPU hardware and the matrix size. **The scheduler has zero effect on this number.** It should be identical (~44ms on L40S with `matrix_size=16384`) in both the baseline and work-stealing runs. It appears in the summary as a sanity check: if it differs between runs, something changed in the workload, not the scheduler.

### Job latency (`PTQ job pctiles`, `PTQ avg job latency`)
Measures from when a job **arrived at the system** to when it **finished executing**:

```
job latency = queue wait time + execution time
```

This is the scheduling-sensitive metric. The execution time is fixed (20 requests × 44ms ≈ 0.94s). The queue wait time depends entirely on whether the GPU was free or had a backlog when the job arrived. A job that waits behind two others in a queue has a job latency of ~2.8s even though the GPU did the same work.

### Queue wait time (`PTQ avg queue wait`)
Measures from arrival to when the GPU actually starts on the job. This is the most direct measure of scheduling quality — the scheduler's entire job is to minimize this. Work stealing should reduce it by moving jobs from long queues to short ones.

---

## Experimental Setup Recap

- **Cluster:** 2× NVIDIA L40S (44 GB VRAM) on MIT Engaging
- **Target operating region:** ~70% GPU utilization (energy-proportional, latency-stable)
- **Arrival rate:** 2.0 jobs/sec (Poisson), 20 requests/job → ~0.94s execution per job
- **Mean utilization:** `(2.0/2) × 0.94 ≈ 0.94` → bursts push above 70%, creating real queuing
- **Three workload scenarios:** PTQ only, training only, mixed PTQ + training

### Why the baseline is expected to struggle

Round-robin assigns jobs in strict alternation with no awareness of queue depth. With Poisson arrivals, bursts naturally occur — two or three jobs arrive close together and stack on one GPU while the other is free. Those queued jobs wait behind each other even though capacity exists next door. The per-request GPU compute time stays the same; the job latency inflates because of queue wait.

---

## Experiment 1: 100 PTQ Inference Jobs

**Config:** `ptq_100.yaml` vs. `ws_ptq_100.yaml`
Each job: 20 matmuls on a 16384×16384 float16 tensor. Execution time ≈ 0.94s.

### Expected baseline behavior

- Round-robin alternates GPUs regardless of queue depth
- Poisson bursts cause occasional 2–3 job queues on one GPU while the other is idle
- Queued jobs wait their full turn; queue wait inflates job latency at the tail

| Metric | Expected Baseline |
|--------|-------------------|
| PTQ avg request time | ~44ms (GPU compute only — scheduler-invariant) |
| PTQ avg job latency | ~1.2–1.5s (execution + some queue wait) |
| PTQ p50 job latency | ~0.94s (landed on free GPU, no wait) |
| PTQ p95 job latency | ~2.0–3.0s (queued behind 1–2 jobs) |
| PTQ p99 job latency | ~3.0–4.5s (worst-case burst queue) |
| PTQ avg queue wait | ~0.3–0.6s |
| GPU util (mean) | ~65–75% per GPU |
| Util std | Low — round-robin is balanced *on average* |

### Expected work-stealing behavior

- Initial placement uses queue-length + utilization scoring (same average distribution as round-robin)
- When a burst causes one GPU's queue to reach 2+ while the other is idle, the idle GPU steals from the tail
- Stolen jobs start immediately instead of waiting

| Metric | Expected Work Stealing | Change vs. Baseline |
|--------|----------------------|---------------------|
| PTQ avg request time | ~44ms | **= same** (sanity check) |
| PTQ avg job latency | ~1.0–1.2s | Lower |
| PTQ p50 job latency | ~0.94s | ≈ same |
| PTQ p95 job latency | ~1.2–1.8s | **20–40% lower** |
| PTQ p99 job latency | ~1.5–2.5s | **30–50% lower** |
| PTQ avg queue wait | ~0.05–0.2s | **Substantially lower** |
| GPU util (mean) | ~65–75% per GPU | ≈ same |
| Steal count | 10–30 over 100 jobs | — |

**Key insight:** The per-request time is identical in both runs. The entire difference shows up in queue wait. Work stealing attacks only the wait — it cannot make the GPU run faster, but it can ensure no job waits unnecessarily.

---

## Experiment 2: 50 Training Jobs (500 steps each)

**Config:** `train_50.yaml` vs. `ws_train_50.yaml`
Each job: sustained MLP training, multiple minutes of GPU compute. Throughput (samples/sec) is the primary metric.

### Expected baseline behavior

- Training jobs are very long relative to the 0.5s inter-arrival gap
- Round-robin distributes training jobs evenly from the start; both GPUs stay saturated
- No meaningful queue imbalance forms because each job locks a GPU for minutes

| Metric | Expected Baseline |
|--------|-------------------|
| Avg training duration | Several minutes per job |
| GPU util | ~95–100% (training is compute-saturating) |
| Util std | Very low — both GPUs equally saturated |

### Expected work-stealing behavior

- `steal_threshold=1` is unlikely to trigger: both queues have length 0 or 1 almost always since jobs take so long
- **Expected result:** Essentially identical to baseline

**Key insight for the paper:** This is the intentional negative result. Training jobs are so long that queuing is never the bottleneck — the GPU is always busy. Work stealing cannot help what is not broken. If PTQ shows improvement and training does not, it directly validates the claim: work stealing targets queue imbalance, which only matters for latency-sensitive short jobs.

---

## Experiment 3: Mixed 100 PTQ + 50 Training Jobs

**Config:** `mixed_100_50.yaml` vs. `ws_mixed_100_50.yaml`
The most realistic and interesting scenario: short inference jobs and long training jobs competing for the same GPUs.

### Expected baseline behavior

- When a training job lands on GPU 0 (round-robin has no choice), GPU 0 is locked for minutes
- Subsequent PTQ jobs assigned to GPU 0 by round-robin pile up behind the training job
- A PTQ job that takes 0.94s of GPU time instead waits several minutes in queue first
- **Queue wait completely dominates job latency** for the unlucky PTQ jobs

| Metric | Expected Baseline |
|--------|-------------------|
| PTQ avg request time | ~44ms (unchanged — GPU compute is still fast) |
| PTQ avg job latency | **Very high** — dominated by queue wait behind training |
| PTQ p99 job latency | Could be 10–30× higher than execution time alone |
| PTQ avg queue wait | **Very high** — the dominant cost |
| GPU util | Uneven — GPU with training job at ~100%, other lower |
| Util std | Higher than PTQ-only experiment |

### Expected work-stealing behavior

- Work stealing detects that GPU 0's queue is long (training job + pending PTQ jobs) while GPU 1 is shorter
- Steals PTQ jobs off GPU 0's queue and runs them immediately on GPU 1
- PTQ jobs are no longer stuck behind training jobs

| Metric | Expected Work Stealing | Change vs. Baseline |
|--------|----------------------|---------------------|
| PTQ avg request time | ~44ms | **= same** |
| PTQ avg job latency | Much lower | **Largest improvement of all 3 experiments** |
| PTQ p99 job latency | Substantially lower | Dramatic reduction |
| PTQ avg queue wait | Much lower | Near execution time in best case |
| GPU util std | Lower | More balanced |
| Steal count | Highest of all 3 experiments | Most stealing triggered |

**Key insight:** This is the headline result. The improvement here is not marginal — it is structural. The baseline cannot rescue a PTQ job from behind a training job because it has no mechanism to do so. Work stealing does it naturally.

---

## Summary Table: All Experiments

| Experiment | Metric to Compare | Why Baseline Struggles | Work Stealing Advantage |
|------------|------------------|----------------------|------------------------|
| PTQ only | Job p95/p99 latency, queue wait | Burst arrivals create uneven queues | Steals from long queues during bursts |
| Training only | Training duration, throughput | Round-robin already balanced | None — negative result |
| Mixed | PTQ job latency, queue wait | PTQ stuck behind training jobs | Steals PTQ jobs off overloaded GPUs |

**In all cases: PTQ avg request time (~44ms) should be identical across both schedulers.** If it differs, the workloads were configured differently, not the scheduler.

---

## What a Successful Result Looks Like

1. **PTQ avg request time is identical** in baseline and work-stealing runs (~44ms) — confirms the workload was the same
2. **PTQ job latency and queue wait are lower** in work-stealing runs, especially at p95/p99
3. **Steal count is non-zero** — confirms the mechanism fired
4. **Training throughput is unchanged** — work stealing did not disrupt training
5. **Largest improvement is in the mixed experiment** — where queue imbalance is most extreme

---

## Potential Negative Results and What They Mean

| Observation | Interpretation |
|-------------|----------------|
| Request time differs between runs | Workload config mismatch — check `num_requests` and `matrix_size` are identical |
| Job latency identical, queue wait near 0 | Arrival rate too low — GPUs mostly idle, no queues form; increase `arrival_rate_hz` |
| Job latency identical, queue wait high | Arrival rate too high — both GPUs saturated equally, stealing can't help; reduce `arrival_rate_hz` |
| Steal count = 0 | Queues never reach `steal_threshold=1` — check `poisson_arrivals: true` is set |
| Work stealing increases job latency | Steal overhead outweighs benefit — only happens with very short jobs or very low imbalance |

---

## Connection to Energy Proportionality

GPU utilization is a proxy for power consumption. At 65–75% utilization:
- Both schedulers are in the energy-proportional region the professor described
- Work stealing keeps utilization more uniform across GPUs (lower `util_std`) — same total work, more balanced power draw
- The baseline may leave one GPU idle while another is saturated — idle GPU wastes static power with no performance benefit
