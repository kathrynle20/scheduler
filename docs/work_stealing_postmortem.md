# Postmortem: Why Work Stealing Was Slower Than Baseline

## Symptom

Across six benchmark suites (5–50 trials each) at 67–70% cluster utilization,
the work-stealing scheduler **consistently lost on latency** while *winning*
on load balance. Selected results from `results/`:

| Suite              | Trials | Avg job latency Δ  | p99 latency Δ      | Queue wait Δ        | Util std Δ           |
| ------------------ | ------ | ------------------ | ------------------ | ------------------- | -------------------- |
| 20260504-212353    | 5      | ≈ same             | ↓ 2.5%             | ≈ same              | ↓ 55.3% **better**   |
| 20260504-222301    | 3      | ↑ 20.6% **worse**  | ↑ 10.0% worse      | ↑ 53.2% **worse**   | ↓ 68.1% better       |
| 20260504-233504    | 5      | ↓ 4.3% better      | ↓ 18.7% better     | ↓ 36.5% better      | ↓ 22.1% better       |
| 20260505-000326    | 30     | ↑ 13.3% worse      | ↑ 19.6% worse      | ↑ 38.1% worse       | ↓ 42.1% better       |
| 20260505-035314    | 50     | ↑ 6.1% worse       | ↑ 7.9% worse       | ↑ 15.8% worse       | ↓ 46.1% better       |

The one suite where work stealing won (20260504-233504) was at **33% util**
on 4 GPUs (before the arrival rate was scaled). Once utilization climbed to
the project's target band (50–70%), the algorithm consistently lost.

This is the textbook "**better balance, worse latency**" signature of
**premature placement commitment** — the scheduler distributes load evenly
across GPUs but the distribution itself adds queue wait that exceeds the
savings.

---

## Root Cause #1 — `util_penalty` corrupts JSQ when NVML readings are noisy

The scheduler's score formula was a 50/50 weighted sum:

```
score(gpu) = 0.5 * util_penalty(util_pct) + 0.5 * queue_score(queue_len)
queue_score(q) = 1 / (1 + q)
```

NVML utilization is sampled at **10 Hz** (every 100 ms). The PTQ workload
runs **20 sequential matmuls of ~47 ms each** with brief gaps between them.
A util reading taken between matmuls returns ~10–20%; a reading taken mid-
matmul returns ~80–90%. **At the timescale of a single placement decision,
util is uninformative — it is essentially a random number in [10, 90]%.**

Because `util_penalty` had equal weight to `queue_score`, a noisy low-util
reading on a *busy* GPU could **override its higher queue count**.

### Concrete failure scenario

- GPU 0: 1 job running, NVML happens to read 80% util (mid-matmul)
- GPU 1: 2 jobs (1 running between matmuls + 1 queued), NVML reads 10% util

| GPU | queue_len | util | util_penalty | queue_score | **score** |
| --- | --------- | ---- | ------------ | ----------- | --------- |
| 0   | 1         | 80%  | 0.0          | 0.500       | **0.250** |
| 1   | 2         | 10%  | 1.0          | 0.333       | **0.667** |

The scheduler picks **GPU 1, the GPU with more queued work**. A strict
JSQ scheduler would have correctly picked GPU 0.

These mistakes accumulate. Over 100 jobs at 67% util, ~20 are misrouted to
the longer-queue GPU. Work stealing then has to clean up reactively — but
stealing only fires when a worker's queue fully drains, by which time the
misrouted jobs have already paid most of their queue-wait penalty.

---

## Root Cause #2 — 5 ms idle backoff drops responsiveness

When a worker found its queue empty *and* could not steal, it released the
queue lock and slept for `_IDLE_BACKOFF_S = 5 ms` before retrying:

```python
while True:
    with self._queue_lock:
        if self._queues[gpu_id]: ...
        stolen = self._try_steal_locked(gpu_id)
        if stolen is not None: ...
        if self._all_dispatched and self._jobs_remaining == 0: ...
    time.sleep(_IDLE_BACKOFF_S)   # <-- bug
```

If a job was placed on this GPU's queue **during the 5 ms sleep**, the
worker did not notice until the sleep elapsed. With ~50 idle-to-busy
transitions per GPU per run, this added up to **~250 ms of unnecessary
queue wait per GPU per run**, on average 2.5 ms per job.

This isn't enough to explain the full latency gap on its own, but it
compounds with Root Cause #1: jobs that were *correctly* placed on an
idle GPU's queue still waited ~2.5 ms before the worker woke up.

---

## Root Cause #3 — Stealing fires too late to recover from bad placements

Stealing only happens **after a worker's own queue is empty** (`_dequeue`
checks own queue first, then attempts steal). By this point:

- The worker has already finished its previous job.
- If the placement decision was bad, the misrouted job has already been
  **popped by the slow-GPU worker** and is now running there. Stealing
  cannot move running jobs — only queued ones.
- If the imbalance has just been created (a new burst), the steal does
  rescue queued jobs — but the rescued job has already incurred queue wait
  during the dispatch lag.

---

## Why training and mixed look different

- **Training only**: jobs are long (~1.07 s each) and uniform, no queuing
  matters at this arrival rate. Both schedulers complete in ~equivalent
  wall time. Steal count = 0 because there's never enough queue depth.
- **Mixed**: training jobs (1 s) parked next to PTQ jobs (~0.94 s) create
  longer-lived imbalances that stealing *can* exploit. This is the only
  experiment where work stealing wins on latency (5–9% better in suite
  20260505-035314) — because the long training jobs make stealing
  opportunities last long enough to matter.

---

## The fix

Three changes, in priority order:

### 1. Strict JSQ in `WorkStealingScheduler.place()`

Replace the weighted-sum scoring with a strict shortest-queue lookup,
using util only as a deterministic tiebreaker:

```python
chosen = min(
    gpu_states,
    key=lambda s: (
        self._queue_lengths[s.id],   # primary:    outstanding load
        s.util_pct,                  # tiebreak 1: cooler/quieter
        s.id,                        # tiebreak 2: deterministic
    ),
)
```

Util cannot drag placement away from a shorter queue. The previous
`_score`/`_util_penalty`/`_queue_score` helpers are removed; the
`target_util_low`/`target_util_high` constructor params are kept (still
referenced by configs) but no longer affect behavior.

### 2. Replace polling sleep with `threading.Condition`

`WorkStealingRunner` now owns a `threading.Condition` bound to the queue
lock. Workers call `cv.wait(timeout)` instead of `time.sleep`. Three
producers notify it:

- **Placement**: after appending to a queue.
- **Completion**: in the worker's `finally` block, so peers can wake up
  to check for newly-stealable work.
- **Shutdown**: when `_all_dispatched` flips.

This eliminates the 5 ms wake-up delay and is also faster than polling
under low contention.

### 3. Wake peers on completion (enables proactive stealing)

When a worker finishes a job and decrements `_jobs_remaining`, it now
calls `cv.notify_all()`. If a peer's queue has stale work that wasn't
stealable when this worker was busy, the peer can recheck immediately.

---

## What this teaches us

- **Don't combine signals of different reliability with equal weight.**
  NVML utilization is a 100 ms-resolution sample of a phenomenon (matmul
  bursts) that changes at 50 ms granularity. It is not a usable primary
  placement signal at the single-job timescale.

- **Polling is a tax that compounds with the number of idle transitions.**
  Even a 5 ms backoff was responsible for a measurable share of the
  observed latency penalty. Use Condition variables for instant wake-up.

- **JSQ is hard to beat for symmetric workloads.** The baseline's "wait
  for any GPU to free up, then dispatch" model is implicitly optimal JSQ
  with a virtual dispatcher queue. Work stealing only adds value when
  it can correct imbalances that JSQ couldn't have prevented — for
  example, jobs of unknown duration (training next to inference). For
  uniform-duration jobs, work stealing should at minimum *match* JSQ.

- **Reactive stealing is a safety net, not a primary mechanism.** When
  placement is correct, stealing rarely fires. When placement is wrong,
  stealing can only rescue the queued work, not the running work — so
  placement must be right *first*.

---

## Validation

After applying the fix, re-run the suite and expect:

- Avg / p50 / p95 / p99 job latency: **equal to or better than baseline**.
- Avg queue wait: **equal to or better than baseline**.
- Cluster util std / max imbalance: **still better than baseline**, since
  strict JSQ is at least as balanced as RR and stealing still fires when
  bursts produce queue imbalance.
- Avg request time (per-matmul GPU compute): **unchanged** (it's
  scheduler-invariant).
- Steal count: **comparable** (~15–30 on PTQ at 2 GPUs / 67% util). It
  may decrease slightly because JSQ produces fewer imbalances to fix.
