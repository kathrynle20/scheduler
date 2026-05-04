# Work-Stealing Scheduler: Implementation Plan

## Context

Based on the professor meeting (see `professor_transcript_summary.md`), the project is shifting from thermal-aware scheduling toward **work stealing** as the main contribution. The core research question is:

> Can a work-stealing scheduler reduce latency and improve GPU utilization balance compared to a naive round-robin baseline, for inference workloads on a multi-GPU cluster?

Work stealing is mostly for load balancing: if one GPU has a long queue of pending jobs and another is idle, the idle GPU "steals" a job from the busy one. This keeps GPUs in the energy-proportional utilization range (50–70%) while reducing tail latency caused by queue buildup.

---

## What "Work Stealing" Means Here

In the current codebase, each GPU processes exactly one job at a time (tracked by `_busy_gpus` in `ExperimentRunner`). There are no per-GPU queues — the dispatcher just waits until a GPU is free.

For work stealing to be meaningful, we need **per-GPU job queues** so that:
1. Incoming jobs are assigned to a GPU's queue (even if that GPU is busy)
2. A GPU worker thread pulls from its own queue when it finishes a job
3. If its queue is empty, it **steals** from the GPU with the longest queue
4. The stealing decision is informed by queue length and GPU utilization

The diagram below shows the difference:

```
BEFORE (ExperimentRunner):           AFTER (WorkStealingRunner):
  
  Global queue                          Per-GPU queues
  ─────────────                         ──────────────
  [job1, job2, ...]                     GPU1: [job3, job7]
        │                               GPU2: [job1, job5, job9]  ← victim
        ▼                               GPU3: []  ← thief: steals job9
  Scheduler.place()
        │                               Worker threads pull from own queue,
        ▼                               steal from back of longest queue
  Dispatch to first free GPU            when own queue is empty
```

---

## Architecture Overview

### New Files to Create

| File | Purpose |
|------|---------|
| `schedulers/work_stealing.py` | `WorkStealingScheduler` — initial placement + queue-aware scoring |
| `framework/ws_runner.py` | `WorkStealingRunner` — per-GPU queues, worker threads, steal logic |
| `configs/work_stealing.yaml` | Config for work-stealing experiments |
| `tests/test_work_stealing.py` | Unit tests |

### Files to Modify

| File | Change |
|------|--------|
| `schedulers/__init__.py` | Register `WorkStealingScheduler` in `build()` |
| `experiments/run_benchmark.py` | Add `work_stealing` to `--scheduler` choices; use `WorkStealingRunner` |
| `evaluation/metrics.py` | Add `utilization_balance()` |
| `evaluation/analyze.py` | Add balance metrics + steal count to `SummaryReport` |

---

## Implementation Details

### 1. `schedulers/work_stealing.py` — WorkStealingScheduler

**Role:** Decides which GPU's queue a new arriving job should enter. Also exposes `notify_dequeued()` so the runner can keep queue-length estimates accurate after steals.

```python
class WorkStealingScheduler(Scheduler):
    """
    Assigns incoming jobs to the GPU with the shortest queue weighted by
    utilization. Prefers GPUs below target_util_high; avoids overloaded GPUs.

    Parameters:
        target_util_low  (float): floor of energy-proportional region (default 50%)
        target_util_high (float): ceiling before latency degrades (default 70%)
        steal_threshold  (int):   min queue-length imbalance needed to trigger
                                  stealing in WorkStealingRunner (default 2)
    """

    def __init__(self, target_util_low=50.0, target_util_high=70.0, steal_threshold=2):
        ...
        self._queue_lengths: dict[int, int] = defaultdict(int)
        self._lock = threading.Lock()

    def place(self, job: Job, gpu_states: list[GpuState]) -> int | None:
        # Score each GPU: combine queue length + utilization penalty
        # queue_score  = 1 / (1 + queue_length)      — prefer shorter queues
        # util_score   = 1.0 if util < target_util_high else degrades to 0
        # final score  = 0.5 * util_score + 0.5 * queue_score
        # Returns gpu_id of argmax; increments internal queue length counter

    def notify_dequeued(self, gpu_id: int):
        # Called by runner when a job is dequeued (either executed or stolen away)
        # Decrements self._queue_lengths[gpu_id]

    def get_queue_lengths(self) -> dict[int, int]:
        # Returns a copy of current queue-length estimates (for logging/tests)
```

**Scoring formula:**
```
util_penalty(u) = 1.0                              if u < target_util_low
                = 1 - (u - low) / (high - low)    if low <= u < target_util_high
                = 0.0                              if u >= target_util_high

queue_score(q)  = 1.0 / (1 + q)

score(gpu)      = 0.5 * util_penalty(gpu.util_pct)
                + 0.5 * queue_score(queue_lengths[gpu.id])
```

Assign to `argmax(score)`. If all GPUs are above `target_util_high`, still assign to least-loaded (don't return None — we need to make progress).

---

### 2. `framework/ws_runner.py` — WorkStealingRunner

**Role:** Replaces `ExperimentRunner` when using the work-stealing scheduler. Maintains per-GPU deques, starts persistent worker threads, and implements the steal loop.

```python
class WorkStealingRunner:
    def __init__(
        self,
        scheduler: WorkStealingScheduler,
        monitor: GpuMonitor,
        collector: MetricsCollector,
        output_root: Path,
        worker_gpu_ids: list[int],
        steal_threshold: int = 2,
    ):
        self._queues: dict[int, deque[tuple[Job, Workload]]] = {
            gid: deque() for gid in worker_gpu_ids
        }
        self._queue_lock = threading.Lock()
        self._all_dispatched = False
        self._jobs_remaining = 0       # protected by _queue_lock
        self._steal_count = 0          # protected by _queue_lock
        self._steal_threshold = steal_threshold

    def run(self, jobs, workload_factory) -> RunArtifacts:
        # 1. Create run dir, open results CSV, start MetricsCollector
        # 2. Start one persistent worker thread per GPU
        # 3. Main loop: for each job sorted by arrival_time:
        #      - _wait_until(job.arrival_time)
        #      - gpu_states = monitor.sample()
        #      - gpu_id = scheduler.place(job, available_gpu_states)
        #      - with _queue_lock: _queues[gpu_id].append((job, workload)); _jobs_remaining += 1
        # 4. Set _all_dispatched = True
        # 5. Join all worker threads
        # 6. Stop collector, return RunArtifacts (include steal_count in metadata)
```

**Worker thread loop:**
```python
def _worker_loop(self, gpu_id, writer, csv_lock):
    while True:
        item = self._dequeue(gpu_id)    # own queue or steal
        if item is None:
            break                        # all done

        job, workload = item
        self._set_monitor_load(gpu_id, util_pct=80.0, mem_mb=job.mem_required_mb)
        try:
            result = workload.run(gpu_id)
            with csv_lock:
                writer.writerow([job.id, job.workload_type, gpu_id,
                                  result.start_ts, result.end_ts, result.extra_json()])
        finally:
            self._set_monitor_load(gpu_id, util_pct=0.0, mem_mb=0)
            with self._queue_lock:
                self._jobs_remaining -= 1
            self.scheduler.notify_dequeued(gpu_id)
```

**Dequeue logic (the actual stealing):**
```python
def _dequeue(self, gpu_id) -> tuple[Job, Workload] | None:
    while True:
        with self._queue_lock:
            # 1. Try own queue first
            if self._queues[gpu_id]:
                return self._queues[gpu_id].popleft()

            # 2. Try to steal from the longest queue
            stolen = self._try_steal_locked(gpu_id)
            if stolen:
                return stolen

            # 3. Check exit condition
            if self._all_dispatched and self._jobs_remaining == 0:
                return None

        time.sleep(0.005)   # brief wait; no new work available right now
```

**Steal logic:**
```python
def _try_steal_locked(self, thief_id) -> tuple[Job, Workload] | None:
    """Must be called with _queue_lock held."""
    my_len = len(self._queues[thief_id])
    # Find the most-loaded GPU
    victim_id = max(
        self._queues,
        key=lambda gid: len(self._queues[gid])
    )
    victim_len = len(self._queues[victim_id])

    if victim_id != thief_id and (victim_len - my_len) >= self._steal_threshold:
        # Steal from the TAIL (work stealing convention: tail = newest, least-started work)
        job, workload = self._queues[victim_id].pop()
        self._steal_count += 1
        self.scheduler.notify_dequeued(victim_id)  # keep scheduler estimates accurate
        return job, workload

    return None
```

**Why steal from the tail?** In classic work stealing (e.g., Chase-Lev deque), the owner pops from the front (FIFO order) and thieves steal from the tail. The tail holds the most recently added — and therefore most unstarted — work, minimizing wasted setup effort.

---

### 3. `configs/work_stealing.yaml`

```yaml
gpus:
  ids: [0, 1, 2, 3]
  scheduler_gpu: 0
  worker_gpus: [1, 2, 3]
  neighbors: {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}

scheduler:
  name: work_stealing
  work_stealing:
    target_util_low: 50.0
    target_util_high: 70.0
    steal_threshold: 2

monitor:
  backend: simulated      # change to nvml on real hardware
  sample_hz: 10
  simulated:
    ambient_c: 25
    k_load: 0.6
    k_cool: 0.05
    k_neighbor: 0.1

arrival_rate_hz: 2
workload_mix:
  - {type: ptq, count: 100, mem_required_mb: 2048}
  - {type: training, count: 50, mem_required_mb: 16384, num_steps: 500}

output_dir: ./runs
```

Also create `configs/work_stealing_smoke.yaml` — identical to `smoke.yaml` but with `scheduler.name: work_stealing`.

---

### 4. Updates to `schedulers/__init__.py`

```python
from schedulers.work_stealing import WorkStealingScheduler

__all__ = ["Scheduler", "BaselineScheduler", "HybridScheduler", "WorkStealingScheduler"]

def build(name: str, config: dict) -> Scheduler:
    if name == "baseline":
        return BaselineScheduler()
    if name == "hybrid":
        return HybridScheduler(**(config.get("hybrid") or {}))
    if name == "work_stealing":
        return WorkStealingScheduler(**(config.get("work_stealing") or {}))
    raise ValueError(f"unknown scheduler: {name}")
```

---

### 5. Updates to `experiments/run_benchmark.py`

- Add `"work_stealing"` to `--scheduler` choices
- After building the scheduler, check its type to select the runner:

```python
from framework.ws_runner import WorkStealingRunner
from schedulers.work_stealing import WorkStealingScheduler

# ... existing setup ...

if isinstance(scheduler, WorkStealingScheduler):
    runner = WorkStealingRunner(
        scheduler=scheduler,
        monitor=monitor,
        collector=collector,
        output_root=Path(config.get("output_dir", "runs")),
        worker_gpu_ids=worker_gpu_ids,
        steal_threshold=config["scheduler"].get("work_stealing", {}).get("steal_threshold", 2),
    )
else:
    runner = ExperimentRunner(
        scheduler=scheduler,
        monitor=monitor,
        collector=collector,
        output_root=Path(config.get("output_dir", "runs")),
        worker_gpu_ids=worker_gpu_ids,
    )
```

---

### 6. `evaluation/metrics.py` — Add `utilization_balance()`

```python
def utilization_balance(util_by_gpu: dict[int, list[float]]) -> dict:
    """
    Given per-GPU utilization time series, compute how balanced the cluster is.

    Returns:
        {
          "per_gpu_mean": {gpu_id: mean_util, ...},
          "cluster_mean": float,
          "cluster_std":  float,    # std of per-GPU means — lower = more balanced
          "max_imbalance": float,   # max(per_gpu_mean) - min(per_gpu_mean)
        }
    """
```

This is the key metric for evaluating whether work stealing actually balanced the cluster compared to round-robin.

---

### 7. `evaluation/analyze.py` — Extend SummaryReport

Add to `SummaryReport`:
- `util_balance: dict | None` — output of `utilization_balance()`, built from `timeseries.csv`
- `steal_count: int | None` — read from `results.csv` metadata row (written by `WorkStealingRunner`)

Update `format()` to print these when present.

---

### 8. `tests/test_work_stealing.py`

```python
def test_work_stealing_assigns_to_shortest_queue():
    # GPU 0 queue=3, GPU 1 queue=0. Next job should go to GPU 1.

def test_work_stealing_avoids_overutilized_gpu():
    # GPU 0 util=90%, GPU 1 util=30%. Even with equal queues, prefer GPU 1.

def test_steal_triggers_at_threshold():
    # Queue: GPU1=[j1,j2,j3], GPU2=[]. steal_threshold=2.
    # GPU2 worker calls _dequeue: should steal one job from GPU1.

def test_steal_does_not_trigger_below_threshold():
    # Queue: GPU1=[j1,j2], GPU2=[j1]. steal_threshold=2.
    # Imbalance is 1, below threshold. No steal.

def test_all_jobs_complete():
    # End-to-end: 10 jobs, 2 GPUs, simulated monitor. 
    # All 10 jobs appear in results. steal_count >= 0.

def test_utilization_balance_metric():
    # Verify utilization_balance() returns lower cluster_std with balanced workload.
```

---

## Metrics: What to Compare (Baseline vs. Work Stealing)

Per the professor's guidance, the primary comparison is:

| Metric | How Measured | Where |
|--------|-------------|-------|
| **P50/P95/P99 latency** | Per-request latencies from PTQ jobs | `results.csv` → `latency_percentiles()` |
| **Throughput** | Jobs completed per second | `results.csv` → job count / total duration |
| **GPU utilization (per GPU)** | Mean util% over run | `timeseries.csv` → per-GPU mean |
| **Utilization balance** | Std of per-GPU means | `timeseries.csv` → `utilization_balance()` |
| **Steal count** | Number of steals during run | `WorkStealingRunner._steal_count` |

**Expected results (hypothesis):**
- Work stealing → lower latency variance (no one GPU gets a pile of 20 jobs)
- Work stealing → lower `cluster_std` on utilization (GPUs more evenly loaded)
- Work stealing → similar or better throughput (less idle time)

---

## Running the Experiments

```bash
# Smoke test (no GPU)
python -m experiments.run_benchmark \
  --config configs/work_stealing_smoke.yaml \
  --monitor simulated

# Baseline (for comparison)
python -m experiments.run_benchmark \
  --config configs/default.yaml \
  --scheduler baseline \
  --monitor simulated

# Work stealing
python -m experiments.run_benchmark \
  --config configs/work_stealing.yaml \
  --monitor simulated

# On real hardware
export CUDA_DEVICE_ORDER=PCI_BUS_ID
python -m experiments.run_benchmark --config configs/work_stealing.yaml --monitor nvml

# Analyze
python -m evaluation.analyze runs/<run-id>
```

---

## Implementation Order

1. `evaluation/metrics.py` — add `utilization_balance()` (standalone, no deps)
2. `schedulers/work_stealing.py` — `WorkStealingScheduler` (only depends on existing ABCs)
3. `schedulers/__init__.py` — register it
4. `framework/ws_runner.py` — `WorkStealingRunner` (depends on scheduler)
5. `configs/work_stealing.yaml` + `configs/work_stealing_smoke.yaml`
6. `experiments/run_benchmark.py` — wire it up
7. `evaluation/analyze.py` — extend SummaryReport
8. `tests/test_work_stealing.py` — write tests

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Steal from head or tail? | **Tail** | Classic work-stealing: owner takes head (FIFO), thief takes tail (newest unstarted work). Reduces wasted cache/setup. |
| Per-GPU threads or pool? | **Per-GPU persistent threads** | Each GPU needs its own steal loop; a shared pool would require coordination about which thread owns which GPU. |
| Steal threshold | **2 (configurable)** | Prevents thrashing: don't steal a single job from a GPU with only 1 more than you. |
| Place() returns None? | **Never return None** | Work-stealing should always make progress; assign to least-loaded even above target_util_high. |
| Queue lock granularity | **Single global queue lock** | Simpler to reason about correctness. Contention is low since jobs take seconds each. |
