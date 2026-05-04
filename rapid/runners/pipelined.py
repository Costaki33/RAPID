"""High-level "production" runners that combine parallel CPU preprocessing
with single- or dual-GPU megabatch inference.

These are thin wrappers around the lower-level runners that expose the
common deployment shapes:

- :func:`run_pipelined_single_gpu` — N CPU preprocess workers → 1 GPU actor,
  with micro-batched pipelining (see :mod:`rapid.runners.cpu_worker_sweep`).
- :func:`run_pipelined_dual_gpu` — 2 × (N/2 CPU preprocess workers → 1 GPU
  actor), station list split evenly. Each GPU shard runs its own pipelined
  pool, giving you both data-parallel forward passes *and* parallel
  preprocessing.

The returned :class:`PipelinedResult` summarizes wall time and GPU
utilization so you can tell whether the GPU was ever starved.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import obspy


LOG = logging.getLogger("rapid.runners.pipelined")


@dataclass
class PipelinedResult:
    # ``wall_time_s`` excludes per-shard actor setup / model load / warmup so
    # it is directly comparable to ``run_cpu_worker_sweep`` (which does the
    # same). Effectively: the critical-path compute time for the slower of
    # the two GPU shards.
    wall_time_s: float
    # End-to-end wall including all driver-side orchestration: Ray init (if
    # this call did it), actor spawn, model load, warmup, work, teardown.
    # Use this to reason about first-call latency in a deployment.
    end_to_end_wall_s: float
    sum_stations: int
    sum_windows: int
    n_gpus: int
    n_cpu_workers_per_gpu: int
    batch_size: int
    per_gpu: List[Dict[str, float]] = field(default_factory=list)

    def gpu_utilization_pct(self) -> float:
        if self.wall_time_s <= 0 or not self.per_gpu:
            return 0.0
        # Average across GPUs; each was either fully utilized in parallel.
        utils = [g["gpu_forward_s"] / self.wall_time_s for g in self.per_gpu]
        return 100.0 * sum(utils) / len(utils)


# ---------------------------------------------------------------------------
# Single-GPU pipelined
# ---------------------------------------------------------------------------


def run_pipelined_single_gpu(
    parent_model: str,
    child_model: str,
    streams: List[Tuple[str, obspy.Stream]],
    *,
    n_cpu_workers: int,
    batch_size: int = 256,
    overlap_samples: int = 0,
    dtype: str = "fp32",
    backend_name: str = "lean_pytorch",
) -> PipelinedResult:
    from .cpu_worker_sweep import run_cpu_worker_sweep

    _t0 = time.perf_counter()
    r = run_cpu_worker_sweep(
        parent_model=parent_model,
        child_model=child_model,
        streams=streams,
        n_cpu_workers=n_cpu_workers,
        batch_size=batch_size,
        overlap_samples=overlap_samples,
        dtype=dtype,
        backend_name=backend_name,
    )
    e2e = time.perf_counter() - _t0
    return PipelinedResult(
        wall_time_s=r.wall_time_s,
        end_to_end_wall_s=e2e,
        sum_stations=r.n_stations,
        sum_windows=r.n_windows,
        n_gpus=1,
        n_cpu_workers_per_gpu=n_cpu_workers,
        batch_size=batch_size,
        per_gpu=[{
            "gpu_forward_s": r.gpu_forward_s,
            "gpu_idle_s": r.gpu_idle_s,
            "preprocess_total_s": r.preprocess_total_s,
            "n_gpu_submits": r.n_gpu_submits,
            "n_stations": r.n_stations,
            "n_windows": r.n_windows,
        }],
    )


# ---------------------------------------------------------------------------
# Dual-GPU pipelined (2× single-GPU pipelined shards in parallel via Ray)
# ---------------------------------------------------------------------------


def run_pipelined_dual_gpu(
    parent_model: str,
    child_model: str,
    streams: List[Tuple[str, obspy.Stream]],
    *,
    n_cpu_workers_per_gpu: int,
    batch_size: int = 256,
    overlap_samples: int = 0,
    dtype: str = "fp32",
    backend_name: str = "lean_pytorch",
    num_gpus: int = 2,
) -> PipelinedResult:
    """Split stations evenly across GPUs; each shard runs its own pipelined pool.

    Runs two ``run_cpu_worker_sweep`` invocations concurrently in driver-side
    threads. Both share the same Ray cluster: ``num_cpus = num_gpus *
    (n_cpu_workers_per_gpu + 2)`` so each shard has CPU slots for its own
    preprocess pool plus GPU actor plus driver headroom.
    """
    import threading
    import ray

    from .cpu_worker_sweep import run_cpu_worker_sweep

    total_cpus = num_gpus * (n_cpu_workers_per_gpu + 2)
    if not ray.is_initialized():
        ray.init(
            num_cpus=total_cpus,
            num_gpus=num_gpus,
            ignore_reinit_error=True,
            log_to_driver=False,
        )

    shards: List[List[Tuple[str, obspy.Stream]]] = [[] for _ in range(num_gpus)]
    for i, pair in enumerate(streams):
        shards[i % num_gpus].append(pair)

    results: List[Optional[Any]] = [None] * num_gpus
    errors: List[Optional[BaseException]] = [None] * num_gpus

    def _worker(idx: int) -> None:
        try:
            results[idx] = run_cpu_worker_sweep(
                parent_model=parent_model,
                child_model=child_model,
                streams=shards[idx],
                n_cpu_workers=n_cpu_workers_per_gpu,
                batch_size=batch_size,
                overlap_samples=overlap_samples,
                dtype=dtype,
                backend_name=backend_name,
            )
        except BaseException as e:  # noqa: BLE001
            errors[idx] = e

    t0 = time.perf_counter()
    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(num_gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    end_to_end = time.perf_counter() - t0

    for e in errors:
        if e is not None:
            raise e

    per_gpu = [
        {
            "gpu_forward_s": r.gpu_forward_s,
            "gpu_idle_s": r.gpu_idle_s,
            "preprocess_total_s": r.preprocess_total_s,
            "wall_time_s": r.wall_time_s,
            "n_gpu_submits": r.n_gpu_submits,
            "n_stations": r.n_stations,
            "n_windows": r.n_windows,
        }
        for r in results
        if r is not None
    ]

    # Critical-path compute wall: the slower of the two shards, excluding
    # each shard's own actor setup / model load / warmup (those happen
    # concurrently across shards because each runs in its own driver thread,
    # so ``max(per-shard compute)`` is the correct apples-to-apples number
    # to compare against ``run_cpu_worker_sweep`` single-GPU.
    compute_wall = max((g["wall_time_s"] for g in per_gpu), default=0.0)

    return PipelinedResult(
        wall_time_s=compute_wall,
        end_to_end_wall_s=end_to_end,
        sum_stations=sum(g["n_stations"] for g in per_gpu),
        sum_windows=sum(g["n_windows"] for g in per_gpu),
        n_gpus=num_gpus,
        n_cpu_workers_per_gpu=n_cpu_workers_per_gpu,
        batch_size=batch_size,
        per_gpu=per_gpu,
    )


# ---------------------------------------------------------------------------
# Baseline annotate() on 2 GPUs — the fair comparison
# ---------------------------------------------------------------------------


def run_baseline_dual_gpu(
    parent_model: str,
    child_model: str,
    streams: List[Tuple[str, obspy.Stream]],
    *,
    annotate_kwargs: Optional[Dict[str, Any]] = None,
    num_gpus: int = 2,
) -> PipelinedResult:
    """Run SeisBench ``model.annotate(stream)`` on each of 2 GPUs in parallel.

    Each shard gets its own loaded model and its own station half. Wall time
    is ``max(actor_wall_times)`` plus driver overhead — the fair 2-GPU
    baseline to compare our pipelined lean path against.
    """
    import ray

    if not ray.is_initialized():
        ray.init(num_gpus=num_gpus, ignore_reinit_error=True, log_to_driver=False)

    @ray.remote(num_gpus=1)
    class _BaselineActor:
        def __init__(self, parent, child):
            from rapid.backends.baseline import BaselineAnnotate

            self.backend = BaselineAnnotate(
                parent_model=parent, child_model=child, device="cuda:0",
            )
            self.backend.load()

        def run(self, shard, annotate_kwargs):
            from rapid.runners.single_gpu import run_baseline_single

            res = run_baseline_single(
                self.backend, shard, annotate_kwargs=annotate_kwargs,
            )
            res.predictions = None
            res.annotations_stream = None
            return {
                "wall_time_s": res.total_s,
                "stage_times_s": res.stage_times,
                "n_stations": res.n_stations,
            }

        def ping(self):
            return True

    _t_e2e = time.perf_counter()
    actors = [_BaselineActor.remote(parent_model, child_model) for _ in range(num_gpus)]
    ray.get([a.ping.remote() for a in actors])

    shards: List[List[Tuple[str, obspy.Stream]]] = [[] for _ in range(num_gpus)]
    for i, pair in enumerate(streams):
        shards[i % num_gpus].append(pair)

    t0 = time.perf_counter()
    futures = [a.run.remote(shards[i], annotate_kwargs) for i, a in enumerate(actors)]
    per_gpu = ray.get(futures)
    wall = time.perf_counter() - t0
    end_to_end = time.perf_counter() - _t_e2e

    for a in actors:
        ray.kill(a)

    return PipelinedResult(
        wall_time_s=wall,
        end_to_end_wall_s=end_to_end,
        sum_stations=sum(g["n_stations"] for g in per_gpu),
        sum_windows=-1,
        n_gpus=num_gpus,
        n_cpu_workers_per_gpu=0,
        batch_size=-1,
        per_gpu=[
            {
                "gpu_forward_s": g["wall_time_s"],  # baseline is not split into stages
                "gpu_idle_s": 0.0,
                "preprocess_total_s": 0.0,
                "n_gpu_submits": -1,
                "n_stations": g["n_stations"],
                "n_windows": -1,
            }
            for g in per_gpu
        ],
    )
