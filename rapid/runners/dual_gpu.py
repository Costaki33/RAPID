"""Dual-GPU runner: one backend instance per GPU via Ray actors.

Station list is split in half; each actor runs ``run_lean_single`` on its
shard. End-to-end wall time is ``max(t_actor_0, t_actor_1)`` plus driver
overhead — that's the metric we report so it reflects true parallel gain.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import obspy

from .single_gpu import RunResult, run_baseline_single, run_lean_single


LOG = logging.getLogger("rapid.runners.dual_gpu")


@dataclass
class DualGPUResult:
    per_actor: List[RunResult]
    wall_time_s: float
    sum_stations: int
    sum_windows: int


def _init_ray(num_gpus: int = 2) -> None:
    import ray

    if not ray.is_initialized():
        ray.init(num_gpus=num_gpus, ignore_reinit_error=True, log_to_driver=False)


def run_dual_gpu(
    parent_model: str,
    child_model: str,
    streams: List[Tuple[str, obspy.Stream]],
    *,
    backend_name: str = "lean_pytorch",
    dtype: str = "fp32",
    batch_size: int = 256,
    overlap_samples: int = 0,
    num_gpus: int = 2,
    backend_kwargs: Optional[Dict[str, Any]] = None,
) -> DualGPUResult:
    import ray

    _init_ray(num_gpus=num_gpus)

    @ray.remote(num_gpus=1)
    class _Actor:
        def __init__(self, backend_name, parent, child, dtype, backend_kwargs):
            # Each actor sees a single GPU as ``cuda:0`` (Ray sets CUDA_VISIBLE_DEVICES).
            from rapid.backends import get_backend

            self._backend_name = backend_name
            cls = get_backend(backend_name)
            self.backend = cls(
                parent_model=parent,
                child_model=child,
                device="cuda:0",
                dtype=dtype,
                **(backend_kwargs or {}),
            )
            self.backend.load()

        def run(self, streams, batch_size, overlap_samples):
            # Dispatch to the right runner for the backend. Baseline runs
            # ``model.annotate(merged_stream)`` on its shard (mirrors how a
            # user would actually parallelize annotate() across GPUs);
            # everything else runs the lean megabatch path.
            if self._backend_name == "baseline_annotate":
                from rapid.runners.single_gpu import run_baseline_single

                res = run_baseline_single(self.backend, streams)
            else:
                from rapid.runners.single_gpu import run_lean_single

                res = run_lean_single(
                    self.backend,
                    streams=streams,
                    batch_size=batch_size,
                    overlap_samples=overlap_samples,
                )
            # predictions / streams can be huge; strip for transport.
            res.predictions = None
            res.annotations_stream = None
            return res

        def ping(self) -> bool:
            return True

    actors = [
        _Actor.remote(backend_name, parent_model, child_model, dtype, backend_kwargs)
        for _ in range(num_gpus)
    ]
    ray.get([a.ping.remote() for a in actors])

    # Even split
    shards: List[List[Tuple[str, obspy.Stream]]] = [[] for _ in range(num_gpus)]
    for i, pair in enumerate(streams):
        shards[i % num_gpus].append(pair)

    t0 = time.perf_counter()
    futures = [
        a.run.remote(shards[i], batch_size, overlap_samples)
        for i, a in enumerate(actors)
    ]
    results: List[RunResult] = ray.get(futures)
    wall = time.perf_counter() - t0

    sum_stations = sum(r.n_stations for r in results)
    sum_windows = sum(r.n_windows for r in results)

    for a in actors:
        ray.kill(a)

    return DualGPUResult(
        per_actor=results,
        wall_time_s=wall,
        sum_stations=sum_stations,
        sum_windows=sum_windows,
    )
