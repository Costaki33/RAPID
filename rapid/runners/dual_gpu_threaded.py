"""Thread-based dual-GPU runners (no Ray).

Why a thread-based replacement for ``rapid.runners.dual_gpu`` /
``rapid.runners.pipelined.run_pipelined_dual_gpu``?

The Ray-actor variants spawn a remote actor per GPU and call
``backend.load()`` inside the actor's ``__init__``. Any exception raised at
load time (e.g. :class:`rapid.backends.lean_pytorch.BackendError` for
"EQTransformer cannot run in fp16") manifests as
``ray.exceptions.ActorDiedError: ... actor died because of an error raised
in its creation task``. The original Python exception is wrapped in a Ray
traceback and is harder to handle; the actor process must also be torn down
and recreated each time, multiplying first-call latency.

For dual-GPU dispatch the orchestration is genuinely simple: hand one
station shard to ``cuda:0`` and another to ``cuda:1`` and time the slower
one. PyTorch CUDA APIs release the GIL during kernel launches and
``cudaStreamSynchronize``, so two driver threads can issue concurrent work
to two devices without process-level parallelism.

These functions match the public APIs of their Ray counterparts so they are
drop-in replacements:

- :func:`run_dual_gpu_threaded` -> :class:`~rapid.runners.dual_gpu.DualGPUResult`
- :func:`run_baseline_dual_gpu_threaded` -> :class:`~rapid.runners.pipelined.PipelinedResult`
- :func:`run_pipelined_dual_gpu_threaded` -> :class:`~rapid.runners.pipelined.PipelinedResult`

The ``pipelined`` variant collapses the per-shard CPU worker pool to a
single in-thread preprocess (i.e. it equals the serial-lean variant). This
is intentional for the SeisBench dual-GPU block where each shard has only
1-2 stations and the CPU worker pool would be overkill; the
``n_cpu_workers_per_gpu`` axis is preserved in the row schema so existing
plot/aggregation code keeps working.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import obspy

from rapid.backends.baseline import BaselineAnnotate
from rapid.backends.lean_pytorch import LeanPyTorchBackend
from rapid.runners.dual_gpu import DualGPUResult
from rapid.runners.pipelined import PipelinedResult
from rapid.runners.single_gpu import RunResult, run_baseline_single, run_lean_single

LOG = logging.getLogger("rapid.runners.dual_gpu_threaded")


@dataclass
class LeanTwoGPUHalvesResult:
    """Lean megabatch split across ``cuda:0`` / ``cuda:1`` on contiguous station halves."""

    wall_time_s: float
    end_to_end_wall_s: float
    sum_stations: int
    sum_windows: int
    predictions: Optional[np.ndarray]


@dataclass
class BaselineTwoGPUHalvesResult:
    """Baseline ``annotate`` on each GPU for one contiguous half of the station list."""

    wall_time_s: float
    end_to_end_wall_s: float
    sum_stations: int
    sum_windows: int
    annotations_stream_first_station: Any


def split_streams_two_gpu_halves(
    streams: List[Tuple[str, obspy.Stream]],
) -> Tuple[List[Tuple[str, obspy.Stream]], List[Tuple[str, obspy.Stream]]]:
    """First ``len//2`` stations → GPU 0, remainder → GPU 1 (same order as *streams*)."""
    n = len(streams)
    mid = n // 2
    return streams[:mid], streams[mid:]


def _shard_streams(
    streams: List[Tuple[str, obspy.Stream]], num_gpus: int
) -> List[List[Tuple[str, obspy.Stream]]]:
    shards: List[List[Tuple[str, obspy.Stream]]] = [[] for _ in range(num_gpus)]
    for i, pair in enumerate(streams):
        shards[i % num_gpus].append(pair)
    return shards


def _set_thread_device(idx: int) -> None:
    """Best-effort: set the current CUDA device for this driver thread."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(idx)
    except Exception:
        pass


def run_lean_two_gpu_even_halves(
    parent_model: str,
    child_model: str,
    streams: List[Tuple[str, obspy.Stream]],
    *,
    batch_size: int,
    overlap_samples: int = 0,
    dtype: str = "fp32",
    backend_kwargs: Optional[Dict[str, Any]] = None,
    warmup_iters: int = 1,
) -> LeanTwoGPUHalvesResult:
    """Two driver threads: first half of *streams* on ``cuda:0``, second half on ``cuda:1``.

    Predictions from each shard are concatenated along batch dim (station order
    preserved). ``wall_time_s`` is ``max(shard total_s)`` (parallel critical path).
    """
    backend_kwargs = backend_kwargs or {}
    shard0, shard1 = split_streams_two_gpu_halves(streams)
    results: List[Optional[RunResult]] = [None, None]
    errors: List[Optional[BaseException]] = [None, None]

    e2e0 = time.perf_counter()

    def _worker(idx: int, shard: List[Tuple[str, obspy.Stream]]) -> None:
        device = f"cuda:{idx}"
        _set_thread_device(idx)
        be: Any = None
        try:
            be = LeanPyTorchBackend(
                parent_model=parent_model,
                child_model=child_model,
                device=device,
                dtype=dtype,
                **backend_kwargs,
            )
            be.load()
            res = run_lean_single(
                be,
                shard,
                batch_size=max(1, int(batch_size)),
                overlap_samples=int(overlap_samples),
                warmup_iters=int(warmup_iters),
            )
            results[idx] = res
        except BaseException as exc:  # noqa: BLE001
            errors[idx] = exc
        finally:
            if be is not None:
                try:
                    be.close()
                except Exception:
                    pass

    threads = [
        threading.Thread(target=_worker, args=(0, shard0), name="lean-half-0"),
        threading.Thread(target=_worker, args=(1, shard1), name="lean-half-1"),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    e2e = time.perf_counter() - e2e0

    for e in errors:
        if e is not None:
            raise e

    r0, r1 = results[0], results[1]
    assert r0 is not None and r1 is not None
    preds0, preds1 = r0.predictions, r1.predictions
    merged: Optional[np.ndarray] = None
    if preds0 is not None and preds1 is not None:
        merged = np.concatenate([preds0, preds1], axis=0)
    elif preds0 is not None:
        merged = preds0
    elif preds1 is not None:
        merged = preds1

    compute_wall = max(float(r0.total_s), float(r1.total_s))
    return LeanTwoGPUHalvesResult(
        wall_time_s=compute_wall,
        end_to_end_wall_s=float(e2e),
        sum_stations=int(r0.n_stations + r1.n_stations),
        sum_windows=int(r0.n_windows + r1.n_windows),
        predictions=merged,
    )


def run_baseline_two_gpu_even_halves(
    parent_model: str,
    child_model: str,
    streams: List[Tuple[str, obspy.Stream]],
    *,
    first_station_stream: obspy.Stream,
    annotate_kwargs: Optional[Dict[str, Any]] = None,
) -> BaselineTwoGPUHalvesResult:
    """Parallel baseline annotate on each half.

    Thread 0 also runs ``annotate_stream`` on *first_station_stream* (same
    station as single-GPU ``first_duplicate`` pick path) before closing its
    backend — no second model load on the driver.
    """
    shard0, shard1 = split_streams_two_gpu_halves(streams)
    results: List[Optional[RunResult]] = [None, None]
    errors: List[Optional[BaseException]] = [None, None]
    ann_first_holder: List[Any] = [None]

    e2e0 = time.perf_counter()

    def _worker(idx: int, shard: List[Tuple[str, obspy.Stream]]) -> None:
        device = f"cuda:{idx}"
        _set_thread_device(idx)
        be: Any = None
        try:
            be = BaselineAnnotate(
                parent_model=parent_model,
                child_model=child_model,
                device=device,
            )
            be.load()
            res = run_baseline_single(
                be,
                shard,
                merge_into_one_stream=True,
                annotate_kwargs=annotate_kwargs,
            )
            results[idx] = res
            if idx == 0:
                ann_first_holder[0] = be.annotate_stream(
                    first_station_stream,
                    extra_kwargs=annotate_kwargs,
                )
        except BaseException as exc:  # noqa: BLE001
            errors[idx] = exc
        finally:
            if be is not None:
                try:
                    be.close()
                except Exception:
                    pass

    threads = [
        threading.Thread(target=_worker, args=(0, shard0), name="base-half-0"),
        threading.Thread(target=_worker, args=(1, shard1), name="base-half-1"),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    e2e = time.perf_counter() - e2e0

    for e in errors:
        if e is not None:
            raise e

    r0, r1 = results[0], results[1]
    assert r0 is not None and r1 is not None
    compute_wall = max(float(r0.total_s), float(r1.total_s))

    return BaselineTwoGPUHalvesResult(
        wall_time_s=compute_wall,
        end_to_end_wall_s=float(e2e),
        sum_stations=int(r0.n_stations + r1.n_stations),
        sum_windows=-1,
        annotations_stream_first_station=ann_first_holder[0],
    )


def run_dual_gpu_threaded(
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
    """Drop-in replacement for ``run_dual_gpu`` using driver threads.

    Each thread loads its own backend on its assigned ``cuda:idx`` device and
    runs the appropriate single-GPU runner on its shard. Backend exceptions
    propagate as plain Python exceptions on the calling thread (no Ray actor
    death wrapping).
    """
    backend_kwargs = backend_kwargs or {}
    shards = _shard_streams(streams, num_gpus)

    results: List[Optional[RunResult]] = [None] * num_gpus
    errors: List[Optional[BaseException]] = [None] * num_gpus

    def _worker(idx: int) -> None:
        device = f"cuda:{idx}"
        _set_thread_device(idx)
        be: Any = None
        try:
            if backend_name == "baseline_annotate":
                be = BaselineAnnotate(
                    parent_model=parent_model,
                    child_model=child_model,
                    device=device,
                    dtype=dtype,
                )
                be.load()
                res = run_baseline_single(be, shards[idx])
            else:
                be = LeanPyTorchBackend(
                    parent_model=parent_model,
                    child_model=child_model,
                    device=device,
                    dtype=dtype,
                    **backend_kwargs,
                )
                be.load()
                res = run_lean_single(
                    be,
                    streams=shards[idx],
                    batch_size=batch_size,
                    overlap_samples=overlap_samples,
                )
            res.predictions = None
            res.annotations_stream = None
            results[idx] = res
        except BaseException as exc:  # noqa: BLE001
            errors[idx] = exc
        finally:
            if be is not None:
                try:
                    be.close()
                except Exception:
                    pass

    t0 = time.perf_counter()
    threads = [
        threading.Thread(target=_worker, args=(i,), name=f"dualgpu-{i}")
        for i in range(num_gpus)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    for e in errors:
        if e is not None:
            raise e

    rs = [r for r in results if r is not None]
    return DualGPUResult(
        per_actor=rs,
        wall_time_s=wall,
        sum_stations=sum(r.n_stations for r in rs),
        sum_windows=sum(r.n_windows for r in rs),
    )


def run_baseline_dual_gpu_threaded(
    parent_model: str,
    child_model: str,
    streams: List[Tuple[str, obspy.Stream]],
    *,
    annotate_kwargs: Optional[Dict[str, Any]] = None,
    num_gpus: int = 2,
) -> PipelinedResult:
    """Drop-in replacement for ``run_baseline_dual_gpu`` using driver threads.

    Each thread runs SeisBench ``model.annotate`` on its own ``cuda:idx``.
    """
    shards = _shard_streams(streams, num_gpus)
    per_gpu_data: List[Optional[Dict[str, Any]]] = [None] * num_gpus
    errors: List[Optional[BaseException]] = [None] * num_gpus

    e2e_start = time.perf_counter()

    def _worker(idx: int) -> None:
        device = f"cuda:{idx}"
        _set_thread_device(idx)
        be: Any = None
        try:
            be = BaselineAnnotate(
                parent_model=parent_model,
                child_model=child_model,
                device=device,
            )
            be.load()
            res = run_baseline_single(
                be, shards[idx], annotate_kwargs=annotate_kwargs,
            )
            per_gpu_data[idx] = {
                "wall_time_s": float(res.total_s),
                "n_stations": int(res.n_stations),
            }
        except BaseException as exc:  # noqa: BLE001
            errors[idx] = exc
        finally:
            if be is not None:
                try:
                    be.close()
                except Exception:
                    pass

    t0 = time.perf_counter()
    threads = [
        threading.Thread(target=_worker, args=(i,), name=f"baseline-dualgpu-{i}")
        for i in range(num_gpus)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    end_to_end = time.perf_counter() - e2e_start

    for e in errors:
        if e is not None:
            raise e

    per_gpu = [d for d in per_gpu_data if d is not None]
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
                "gpu_forward_s": g["wall_time_s"],
                "gpu_idle_s": 0.0,
                "preprocess_total_s": 0.0,
                "n_gpu_submits": -1,
                "n_stations": g["n_stations"],
                "n_windows": -1,
            }
            for g in per_gpu
        ],
    )


def run_pipelined_dual_gpu_threaded(
    parent_model: str,
    child_model: str,
    streams: List[Tuple[str, obspy.Stream]],
    *,
    n_cpu_workers_per_gpu: int,
    batch_size: int = 256,
    overlap_samples: int = 0,
    dtype: str = "fp32",
    backend_name: str = "lean_pytorch",
    backend_kwargs: Optional[Dict[str, Any]] = None,
    num_gpus: int = 2,
) -> PipelinedResult:
    """Drop-in replacement for ``run_pipelined_dual_gpu`` using driver threads.

    For workloads with only a couple of stations per shard (e.g. SeisBench
    dual-GPU pairs), a Ray-actor CPU worker pool is overkill — preprocess
    runs in the same thread as the GPU forward. The
    ``n_cpu_workers_per_gpu`` axis is recorded for schema parity but is a
    no-op here (preprocess is single-threaded per shard).
    """
    backend_kwargs = backend_kwargs or {}
    shards = _shard_streams(streams, num_gpus)

    per_gpu_data: List[Optional[Dict[str, Any]]] = [None] * num_gpus
    errors: List[Optional[BaseException]] = [None] * num_gpus

    e2e_start = time.perf_counter()

    def _worker(idx: int) -> None:
        device = f"cuda:{idx}"
        _set_thread_device(idx)
        be: Any = None
        try:
            be = LeanPyTorchBackend(
                parent_model=parent_model,
                child_model=child_model,
                device=device,
                dtype=dtype,
                **backend_kwargs,
            )
            be.load()
            res = run_lean_single(
                be,
                streams=shards[idx],
                batch_size=batch_size,
                overlap_samples=overlap_samples,
            )
            per_gpu_data[idx] = {
                "wall_time_s": float(res.total_s),
                "n_stations": int(res.n_stations),
                "n_windows": int(res.n_windows),
            }
        except BaseException as exc:  # noqa: BLE001
            errors[idx] = exc
        finally:
            if be is not None:
                try:
                    be.close()
                except Exception:
                    pass

    t0 = time.perf_counter()
    threads = [
        threading.Thread(target=_worker, args=(i,), name=f"pipe-dualgpu-{i}")
        for i in range(num_gpus)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    end_to_end = time.perf_counter() - e2e_start

    for e in errors:
        if e is not None:
            raise e

    per_gpu = [d for d in per_gpu_data if d is not None]
    return PipelinedResult(
        wall_time_s=wall,
        end_to_end_wall_s=end_to_end,
        sum_stations=sum(g["n_stations"] for g in per_gpu),
        sum_windows=sum(g["n_windows"] for g in per_gpu),
        n_gpus=num_gpus,
        n_cpu_workers_per_gpu=n_cpu_workers_per_gpu,
        batch_size=batch_size,
        per_gpu=[
            {
                "gpu_forward_s": g["wall_time_s"],
                "gpu_idle_s": 0.0,
                "preprocess_total_s": 0.0,
                "n_gpu_submits": -1,
                "n_stations": g["n_stations"],
                "n_windows": g["n_windows"],
            }
            for g in per_gpu
        ],
    )
