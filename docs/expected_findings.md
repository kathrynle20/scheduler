# Expected Findings: JSQ-No-Steal vs. Work-Stealing Scheduler

## Research Question

> Does reactive work stealing reduce inference latency and improve GPU utilization
> balance compared to a static join-shortest-queue scheduler, when both operate
> on the same per-GPU queue architecture under mixed ML workloads?

---

## Architecture of the Comparison

Both the baseline and the work-stealing scheduler now run through the **same
runner** (`WorkStealingRunner`) using the **same placement algorithm**
(join-shortest-queue, JSQ). The only variable is whether stealing fires:

| Config | Placement | Per-GPU queues | Stealing |
|--------|-----------|----------------|----------|
| `ptq_100.yaml` (baseline) | JSQ | yes | **disabled** (`steal_threshold=99999`) |
| `ws_ptq_100.yaml` | JSQ | yes | **enabled** (`steal_threshold=1`) |

This is the only valid comparison between these two mechanisms. See
`docs/architecture_decision.md` for why the original "wait-for-free vs.
pre-queued" comparison was architecturally invalid.

---

## Understanding the Metrics

### Per-request GPU compute time (`PTQ avg request time`)
Time for a single matmul on the GPU — kernel launch to `synchronize()`.
Determined entirely by GPU hardware and matrix size. **The scheduler cannot
affect this.** It should be ~47ms on L40S at `matrix_size=16384` in both
runs. If it differs, the workload configs are inconsistent — not the scheduler.

### Job latency (`PTQ avg job latency`, `PTQ job pctiles`)
Time from job **arrival** to job **completion**:
```
job latency = queue wait + execution time
```
Execution time is fixed (~0.94s for 20 matmuls at 47ms each). Queue wait is
what the scheduler controls. This is the primary scheduling-quality metric.

### Queue wait (`PTQ avg queue wait`)
Time from arrival to when the GPU actually starts the job. The most direct
measure of scheduling quality. Both schedulers use JSQ placement, which
minimizes queue wait at dispatch time — but conditions change after dispatch.
Stealing reduces queue wait by correcting post-placement imbalances.

---

## Why JSQ-No-Steal Can Still Build Imbalance

JSQ is optimal **at the moment of placement** — it routes each arriving job to
whichever GPU currently has the shortest queue. But the situation can become
suboptimal after placement:

1. **Burst arrivals**: three jobs arrive within 0.1s while GPU 0 has just started
   a job. All three land on GPU 1 (shorter queue at dispatch time). Now GPU 0
   finishes and sits idle while GPU 1 has a queue of 3. JSQ-no-steal cannot
   correct this; the three jobs stay on GPU 1's queue.

2. **Duration variance in mixed workloads**: a training job (1s+) and several
   PTQ jobs (0.94s) land on the same GPU. The PTQ jobs are correctly queued
   behind a same-duration peer, but a training job arrival changes the math.
   JSQ-no-steal committed; work stealing can rescue the PTQ jobs.

These are the conditions where work stealing adds value on top of JSQ.

---

## Experimental Setup

- **Cluster:** 2× NVIDIA L40S (44 GB VRAM) on MIT Engaging
- **Target operating region:** ~65% GPU utilization (energy-proportional)
- **Arrival rate:** 1.4 jobs/sec (Poisson), 20 requests/job → ~0.94s execution
- **Mean util:** `(1.4/2) × 0.94 ≈ 0.66` → ~66% per GPU
- **Three workload scenarios:** PTQ only, training only, mixed PTQ + training

### How jobs arrive at the scheduler

Jobs are not submitted by a live external client. Before the run starts,
`experiments/run_benchmark.py` builds **all** `Job` objects and assigns each
one an `arrival_time` using a monotonic clock (`time.monotonic()`). Rows in
`workload_mix` become concrete jobs (e.g. 100 PTQ, 50 training); those jobs are
**shuffled**, then arrival times are stamped **in that random order**. So PTQ
and training are **randomly interleaved** in mixed experiments.

During the run, `WorkStealingRunner` iterates over jobs **sorted by
`arrival_time`**. For each job it **blocks until** wall-clock time reaches that
job’s `arrival_time`, then calls `place(...)` and **enqueues** the job on the
chosen GPU. Metrics such as “queue wait” measure time from this **dispatch /
arrival instant** (job enters the per-GPU queue) until the GPU **starts** that
job—not a separate pre-scheduler backlog.

The YAML field `arrival_rate_hz` fixes the **long-run average** job rate
(jobs/sec over the whole experiment). The **actual** spacing between successive
arrivals is either bursty (Poisson) or deterministic (even spacing); see below.

### Arrival modes

`build_job_list` supports three arrival modes, checked in priority order:

**`burst_arrivals: true`** (Experiments 1 and 4)

Jobs arrive in synchronized groups of `burst_size` at the same timestamp,
separated by `burst_interval_s` seconds, modelling a batch-dispatch gateway.
All jobs in a burst land before any start running, so JSQ distributes them
by queue count alone — it has no duration information. If the jobs in a burst
have variable request counts (via `num_requests_min`/`num_requests_max`), the
GPU that gets the longer jobs accumulates queue depth while the other goes idle,
creating clear stealing opportunities that Poisson arrivals suppress.

**`poisson_arrivals: true`** (Experiments 2 and 3)

Inter-arrival gaps are independent exponential draws with mean
`1 / arrival_rate_hz`. Arrivals come one at a time; JSQ sees the correct queue
depth before each decision. Post-placement imbalances are smaller and more
transient than under burst mode, so stealing fires less often.

**Neither flag** (smoke/default configs)

Evenly spaced by `1 / arrival_rate_hz`. Fully deterministic; queues stay flat.
Used for smoke tests.

---

## Experiment 1: 100 PTQ Inference Jobs

**Configs:** `ptq_100.yaml` (JSQ, no steal) vs. `ws_ptq_100.yaml` (JSQ + steal)

Jobs arrive in bursts of 4 every 2.5 seconds. Each job draws its request count
from U[10, 30] at arrival time (mean 20 req × 47ms ≈ 0.94s, range 0.47–1.41s).
The combination of burst arrivals and variable request counts is what makes the
PTQ-only comparison meaningful:

- **Burst arrivals** create a queue of 2 jobs per GPU per burst. JSQ distributes
  by count alone (no duration visibility) so whichever jobs happen to be longer
  end up determining which GPU gets stuck.
- **Variable request counts** mean that even with the same queue length, two GPUs
  can have very different amounts of work. When one GPU finishes its 2 short jobs
  before the other finishes its 2 long jobs, the idle GPU has a clear target to
  steal from.

With Poisson arrivals and a fixed 20-request count, JSQ is near-optimal — it
sees the exact queue depth before every decision and all jobs take the same time.
Stealing rarely fires because there's nothing meaningful to correct.

### Expected no-steal baseline

| Metric | Expected (no-steal baseline) |
|--------|------------------------------|
| PTQ avg request time | varies per job (~47ms/req) |
| PTQ avg job latency | ~1.2–1.6s |
| PTQ p50 job latency | ~0.94s (median-duration jobs, short queue wait) |
| PTQ p95 job latency | ~2.0–3.0s (long jobs stuck behind other long jobs) |
| PTQ p99 job latency | ~2.8–3.8s |
| PTQ avg queue wait | ~0.3–0.7s |
| Cluster util mean | ~70–75% |
| Cluster util std | Moderate — long-job clusters cause imbalance each burst |

### Expected work-stealing behavior

| Metric | Expected (work stealing) | Δ vs. baseline |
|--------|--------------------------|----------------|
| PTQ avg request time | same | = same |
| PTQ avg job latency | ~1.0–1.3s | **15–25% lower** |
| PTQ p50 job latency | ~0.94s | ≈ same |
| PTQ p95 job latency | ~1.5–2.2s | **20–35% lower** |
| PTQ p99 job latency | ~2.0–2.8s | **25–35% lower** |
| PTQ avg queue wait | ~0.1–0.3s | **Lower** |
| Cluster util std | Lower | **Better balance** |
| Steal count | ~20–40 | — |

---

## Experiment 2: 50 Training Jobs (500 steps each)

**Configs:** `train_50.yaml` (JSQ, no steal) vs. `ws_train_50.yaml` (JSQ + steal)

Each training job locks a GPU for ~1s+ at 500 steps. With 2 GPUs and 50 jobs,
both GPUs stay near-continuously occupied. JSQ places each job on the shorter
queue, which is already optimal.

| Metric | Expected both schedulers |
|--------|--------------------------|
| Training avg duration | ~1.07s per job |
| Cluster util mean | ~70% |
| Cluster util std | Very low |
| Steal count | **0** — queues never build imbalance |

**This is the intentional negative result.** Training jobs are uniform and
long. JSQ distributes them perfectly; there is no post-placement imbalance
to steal. If PTQ shows improvement and training does not, it validates that
work stealing specifically targets short-job queue imbalance — not a general
speedup from architecture differences.

---

## Experiment 3: Mixed 100 PTQ + 50 Training Jobs

**Configs:** `mixed_100_50.yaml` (JSQ, no steal) vs. `ws_mixed_100_50.yaml` (JSQ + steal)

This is the headline experiment. Training jobs (~1s) and PTQ jobs (~0.94s)
compete for the same GPUs. A key scenario:

1. JSQ places a training job on GPU 0 and a PTQ job on GPU 0's queue (JSQ
   said GPU 0 had the shorter queue at that moment).
2. GPU 1 finishes its PTQ job and becomes idle.
3. **JSQ-no-steal**: the queued PTQ job sits behind the training job.
4. **Work stealing**: GPU 1 steals the PTQ job from GPU 0's queue and runs it immediately.

### Expected no-steal baseline (JSQ)

| Metric | Expected |
|--------|----------|
| PTQ avg job latency | Elevated — some PTQ jobs queued behind training |
| PTQ p95/p99 latency | Substantially inflated by training-job queuing |
| PTQ avg queue wait | Higher than Experiment 1 |
| Cluster util std | Higher — training jobs lock GPUs unevenly |

### Expected work stealing

| Metric | Expected | Δ vs. baseline |
|--------|----------|----------------|
| PTQ avg request time | ~47ms | = same |
| PTQ avg job latency | Lower | **Largest improvement across all 3 experiments** |
| PTQ p95/p99 latency | Substantially lower | **Significant reduction** |
| PTQ avg queue wait | Much lower | Stealing rescues PTQ from behind training |
| Cluster util std | Lower | More balanced |
| Steal count | Highest of all 3 experiments | Most stealing triggered by training/PTQ mix |

**Key insight:** This experiment makes stealing look best because it creates
a scenario JSQ cannot handle at placement time. When a training job arrives,
JSQ cannot know it will hold the GPU for longer than the next PTQ job — the
duration is not known at dispatch time. Stealing is the only mechanism that
can correct this after the fact.

---

---

## Summary

| Experiment | Expected improvement | Why |
|------------|----------------------|-----|
| PTQ only | **15–35% tail latency** | Burst arrivals + variable request counts give JSQ no duration info; stealing rebalances after fast GPU drains its burst queue |
| Training only | None | Uniform long jobs; no post-placement imbalance to correct |
| Mixed | Largest (significant tail latency reduction) | Training job duration unknown at dispatch — JSQ cannot prevent queue stacking; stealing is the only corrective |

---

## What a Successful Result Looks Like

1. **PTQ avg request time is identical** in both runs (~47ms) — workloads are the same
2. **PTQ queue wait and tail latency are lower** in work-stealing runs
3. **Steal count is non-zero** — stealing actually fired
4. **Training throughput is unchanged** — stealing didn't disrupt long-running jobs
5. **Largest improvement is in the mixed experiment**
6. **Both schedulers show lower latency than the old round-robin baseline** — JSQ
   itself is a major improvement; stealing is the incremental gain on top

---

## Interpreting Smaller-Than-Expected Improvements

Because both schedulers now use JSQ placement, the baseline is harder to beat
than the original round-robin comparison. A modest improvement (5–15% tail
latency reduction in Experiment 1) is a **valid positive result** — it means:
- JSQ did most of the work
- Stealing captured the residual burst-driven imbalance

A negligible improvement in Experiment 1 with a meaningful improvement in
Experiment 3 is also a valid result — it shows stealing specifically targets
the duration-heterogeneity problem, not generic queue imbalance.
