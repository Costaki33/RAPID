"""Pipelined CPU preprocess pool + megabatch inference actor.

Supports both GPU and CPU inference devices. The sweep dimension is
``n_cpu_workers`` (the number of Ray CPU preprocessing actors).

What this runner does differently from ``model.annotate(stream)``
-----------------------------------------------------------------

SeisBench's ``annotate()`` runs a *single-threaded* asyncio pipeline in one
Python process:

  streams_to_arrays  →  cut_fragments  →  predict  →  reassemble  →  to_streams

All five stages run in one event loop. The only "parallelism" is that CUDA
kernels launched in ``predict`` are asynchronous, so while the GPU is busy
on batch N, the CPU can run ``cut_fragments`` on batch N+1. Crucially:

- Filtering, resampling and tapering (the CPU-expensive part) happen in ONE
  OS thread. On 228 stations we measured this as ~350 ms — it is the
  dominant wall-time cost. SeisBench does not parallelize it across cores.
- Every forward pass is a fresh PyTorch dispatch; the megabatch is spread
  across many smaller batches (``batch_size`` default 256) sequenced by the
  asyncio queue, so overall throughput is good but GPU launch overhead
  accumulates.

This runner does three things differently:

1. **Preprocess in parallel on ``n_cpu_workers`` Ray actors.** Each worker
   owns a SeisBench model copy and runs ``annotate_stream_pre`` +
   ``stream_to_3c_array`` + windowing on its shard of the stations.
2. **Pipeline CPU ↔ inference via a fill-and-flush buffer.** As each
   station's preprocessed windows arrive back at the driver, they
   accumulate in an in-memory buffer. When the buffer reaches ``batch_size``
   windows, we immediately dispatch ``infer_batch`` on the inference actor
   without waiting for the remaining stations — this overlaps compute with
   ongoing preprocess.
3. **Stay in one big batched forward.** Each inference call is a contiguous
   megabatch of exactly ``batch_size`` (one final flush may be smaller),
   not per-station.

The sweep dimension is ``n_cpu_workers`` ∈ {1..20}. With a single worker the
path degenerates to serial preprocess + megabatch forward (the "lean" path
we measured earlier). With N workers the preprocess cost roughly scales as
``T_pre / N`` until Ray actor startup and serialization overhead take over.

CPU-device-specific notes
-------------------------

When ``device="cpu"`` the same pipeline runs entirely on CPU silicon, so
preprocess workers and the inference actor *compete for cores*. To avoid
BLAS oversubscription (which consistently tanks throughput by 2-3×) each
preprocess worker is pinned to exactly 1 BLAS thread via ``OMP_NUM_THREADS``
and ``torch.set_num_threads(1)``, and the inference actor is given
``infer_num_threads`` — defaulting to
``max(1, os.cpu_count() - n_cpu_workers - 1)`` so preprocess and inference
split the box without contending. A separate ``infer_num_threads`` axis can
be swept to find the optimal split for a given model and station count.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import obspy


LOG = logging.getLogger("rapid.runners.cpu_worker_sweep")


@dataclass
class CPUWorkerResult:
    n_cpu_workers: int
    wall_time_s: float
    gpu_forward_s: float  # kept name for schema parity — is "inference forward time"
    gpu_idle_s: float
    preprocess_total_s: float
    n_stations: int
    n_windows: int
    batch_size: int
    n_gpu_submits: int
    infer_num_threads: int = -1
    infer_device: str = "cuda:0"
    # Memory metrics captured per-process and aggregated by the driver.
    # ``infer_*`` is the inference actor (the process that holds model
    # weights + activations and runs forward passes). ``worker_*`` is the
    # *maximum* across all preprocess actors, so a single number per run
    # still tells you "how big does one preprocess worker get?".
    peak_rss_bytes_driver: int = 0
    peak_rss_bytes_infer: int = 0
    peak_rss_bytes_worker_max: int = 0
    peak_gpu_mem_bytes_infer: int = 0


def _init_ray(num_cpus: int, num_gpus: int = 1) -> None:
    import ray

    if not ray.is_initialized():
        ray.init(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            ignore_reinit_error=True,
            log_to_driver=False,
        )


def _default_infer_threads(n_cpu_workers: int) -> int:
    """Leave one core for the driver, split the rest so preprocess workers
    and the CPU inference actor don't both grab the full BLAS pool."""
    total = os.cpu_count() or 1
    # 1 core each for preprocess workers + 1 for the driver.
    leftover = total - n_cpu_workers - 1
    return max(1, leftover)


def run_cpu_worker_sweep(
    parent_model: str,
    child_model: str,
    streams: List[Tuple[str, obspy.Stream]],
    *,
    n_cpu_workers: int,
    batch_size: int = 256,
    overlap_samples: int = 0,
    dtype: str = "fp32",
    backend_name: str = "lean_pytorch",
    device: str = "cuda:0",
    infer_num_threads: Optional[int] = None,
) -> CPUWorkerResult:
    """Parallel preprocess + pipelined megabatch inference.

    Parameters
    ----------
    device
        Inference device for the inference actor. ``"cuda:N"`` runs the
        forward pass on GPU (one GPU slot reserved in Ray), ``"cpu"`` runs
        it on CPU and reserves ``infer_num_threads`` cores for BLAS.
    infer_num_threads
        CPU-only. Number of BLAS threads to give the inference actor. Only
        honored when ``device="cpu"``. When ``None`` (default), it auto-splits
        the box so preprocess workers + inference + driver fit.
    """
    import ray

    is_cpu_infer = device.startswith("cpu")
    if is_cpu_infer and infer_num_threads is None:
        infer_num_threads = _default_infer_threads(n_cpu_workers)
    infer_num_threads = int(infer_num_threads) if infer_num_threads else 1

    # Reserve CPUs for: preprocess workers (1 each) + inference actor
    # (infer_num_threads if CPU else 1) + driver (1).
    cpu_reservation = (
        n_cpu_workers
        + (infer_num_threads if is_cpu_infer else 1)
        + 1
    )
    _init_ray(
        num_cpus=cpu_reservation,
        num_gpus=0 if is_cpu_infer else 1,
    )

    # ------------------------------------------------------------------
    # Worker types
    # ------------------------------------------------------------------
    # Preprocess workers are pinned to 1 BLAS thread to prevent oversubscription
    # when the inference actor is also on CPU (on GPU-infer runs this is also
    # harmless — filtering/resampling doesn't parallelize across BLAS threads).
    @ray.remote(num_cpus=1)
    class _PreprocessWorker:
        def __init__(self, parent, child):
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            try:
                import torch as _torch
                _torch.set_num_threads(1)
            except Exception:
                pass
            import seisbench.models as sbm

            model_cls = getattr(sbm, parent)
            self.model = model_cls.from_pretrained(child)
            self.model.eval()
            self.in_samples = getattr(self.model, "in_samples")
            # Lightweight per-process RSS watermark: every ``preprocess``
            # call updates it. Cheap because psutil RSS reads are a single
            # ``/proc/[pid]/statm`` read on Linux.
            self._peak_rss = 0
            try:
                import psutil as _ps
                self._ps_proc = _ps.Process(os.getpid())
                self._peak_rss = int(self._ps_proc.memory_info().rss)
            except Exception:
                self._ps_proc = None

        def _sample_rss(self):
            if self._ps_proc is None:
                return
            try:
                rss = int(self._ps_proc.memory_info().rss)
                if rss > self._peak_rss:
                    self._peak_rss = rss
            except Exception:
                pass

        def preprocess(self, sta, stream, overlap_samples):
            from rapid.data import (
                WindowSpec,
                build_megabatch,
                preprocess_for_model,
                stream_to_3c_array,
            )

            pre = preprocess_for_model(self.model, stream)
            arr = stream_to_3c_array(
                pre, component_order=self.model.component_order or "ZNE"
            )
            if arr is None:
                self._sample_rss()
                return sta, None
            spec = WindowSpec(in_samples=self.in_samples, overlap_samples=overlap_samples)
            mb = build_megabatch([(sta, arr)], spec)
            self._sample_rss()
            return sta, mb.windows  # (n_windows_for_this_station, 3, T)

        def peak_rss(self) -> int:
            self._sample_rss()
            return int(self._peak_rss)

    infer_num_cpus_for_actor = infer_num_threads if is_cpu_infer else 1
    infer_num_gpus_for_actor = 0 if is_cpu_infer else 1

    @ray.remote(num_cpus=infer_num_cpus_for_actor, num_gpus=infer_num_gpus_for_actor)
    class _InferenceActor:
        def __init__(self, parent, child, dtype, batch_size, backend_name,
                     device, infer_num_threads):
            # BLAS thread control for CPU-infer mode. We set both env vars
            # (for libraries that read them at import time) and the PyTorch
            # setter (for runtime control).
            if device.startswith("cpu"):
                os.environ["OMP_NUM_THREADS"] = str(infer_num_threads)
                os.environ["MKL_NUM_THREADS"] = str(infer_num_threads)
                os.environ["OPENBLAS_NUM_THREADS"] = str(infer_num_threads)
            try:
                import torch as _torch
                if device.startswith("cpu"):
                    _torch.set_num_threads(int(infer_num_threads))
            except Exception:
                pass
            from rapid.backends import get_backend

            cls = get_backend(backend_name)
            self.backend = cls(
                parent_model=parent,
                child_model=child,
                device=device,
                dtype=dtype,
            )
            self.backend.load()
            self.device = device
            self.batch_size = batch_size
            self.gpu_forward_s = 0.0
            self.n_windows = 0
            self.n_calls = 0

            # Memory bookkeeping. For CUDA devices, reset the peak counter
            # so we measure just this trial's peak (not model load, which
            # happens before the caller starts timing). For CPU RSS we
            # seed from psutil and update on every infer() call.
            try:
                import torch as _torch
                if device.startswith("cuda") and _torch.cuda.is_available():
                    _torch.cuda.reset_peak_memory_stats(device=device)
            except Exception:
                pass
            self._peak_rss = 0
            try:
                import psutil as _ps
                self._ps_proc = _ps.Process(os.getpid())
                self._peak_rss = int(self._ps_proc.memory_info().rss)
            except Exception:
                self._ps_proc = None

        def _sample_rss(self):
            if self._ps_proc is None:
                return
            try:
                rss = int(self._ps_proc.memory_info().rss)
                if rss > self._peak_rss:
                    self._peak_rss = rss
            except Exception:
                pass

        def warmup(self):
            import numpy as _np

            dummy = _np.zeros(
                (self.batch_size, 3, self.backend.in_samples), dtype=_np.float32
            )
            self.backend.infer_batch(dummy)

        def infer(self, windows):
            """Run one forward pass on a pre-shaped ``(B, 3, T)`` megabatch."""
            import time as _t

            t0 = _t.perf_counter()
            _ = self.backend.infer_batch(windows)
            self.gpu_forward_s += _t.perf_counter() - t0
            self.n_windows += int(windows.shape[0])
            self.n_calls += 1
            self._sample_rss()

        def stats(self):
            """Return (gpu_forward_s, n_windows, n_calls, peak_rss_bytes, peak_gpu_bytes)."""
            self._sample_rss()
            peak_gpu = 0
            try:
                import torch as _torch
                if self.device.startswith("cuda") and _torch.cuda.is_available():
                    peak_gpu = int(_torch.cuda.max_memory_allocated(device=self.device))
            except Exception:
                pass
            return (
                self.gpu_forward_s,
                self.n_windows,
                self.n_calls,
                int(self._peak_rss),
                peak_gpu,
            )

    # ------------------------------------------------------------------
    # Spin up workers + inference actor
    # ------------------------------------------------------------------
    from rapid.memory import RSSPoller

    driver_mem = RSSPoller(interval_s=0.05)
    driver_mem.start()

    workers = [
        _PreprocessWorker.remote(parent_model, child_model)
        for _ in range(n_cpu_workers)
    ]
    infer_actor = _InferenceActor.remote(
        parent_model, child_model, dtype, batch_size, backend_name,
        device, infer_num_threads,
    )
    ray.get(infer_actor.warmup.remote())

    # ------------------------------------------------------------------
    # Dispatch preprocess tasks round-robin
    # ------------------------------------------------------------------
    t_start = time.perf_counter()
    t_pre_s = time.perf_counter()

    worker_iter = cycle(workers)
    preprocess_futures: Dict[Any, str] = {}
    for sta, st in streams:
        w = next(worker_iter)
        preprocess_futures[w.preprocess.remote(sta, st, overlap_samples)] = sta

    # ------------------------------------------------------------------
    # Drain in completion order; flush GPU calls as soon as the buffer
    # reaches batch_size. This is the CPU↔GPU overlap.
    # ------------------------------------------------------------------
    gpu_futures: List[Any] = []
    buffer: List[np.ndarray] = []
    buffer_count = 0
    total_windows = 0

    while preprocess_futures:
        ready, _pending = ray.wait(list(preprocess_futures), num_returns=1)
        fut = ready[0]
        del preprocess_futures[fut]
        sta, windows = ray.get(fut)
        if windows is None or windows.size == 0:
            continue
        buffer.append(windows)
        buffer_count += windows.shape[0]
        total_windows += windows.shape[0]

        while buffer_count >= batch_size:
            mega = np.concatenate(buffer, axis=0)
            to_send = mega[:batch_size]
            leftover = mega[batch_size:]
            gpu_futures.append(infer_actor.infer.remote(to_send))
            if leftover.shape[0] > 0:
                buffer = [leftover]
                buffer_count = leftover.shape[0]
            else:
                buffer = []
                buffer_count = 0

    t_pre_total = time.perf_counter() - t_pre_s

    # Final flush: any leftover windows go as a (smaller) final batch.
    if buffer_count > 0:
        mega = np.concatenate(buffer, axis=0)
        gpu_futures.append(infer_actor.infer.remote(mega))

    ray.get(gpu_futures)
    wall = time.perf_counter() - t_start

    infer_stats = ray.get(infer_actor.stats.remote())
    gpu_fwd, n_windows_reported, n_calls, peak_rss_infer, peak_gpu_infer = infer_stats
    # Harvest worker RSS peaks before we kill the actors. Max across
    # workers is the interesting number — they run in parallel and hold
    # identical-ish model replicas, so the max tells us "how big does
    # one preprocess worker get?".
    worker_peaks: List[int] = []
    try:
        worker_peaks = ray.get([w.peak_rss.remote() for w in workers])
    except Exception:
        worker_peaks = []
    worker_peak_max = max(worker_peaks) if worker_peaks else 0

    for w in workers:
        ray.kill(w)
    ray.kill(infer_actor)

    driver_mem_stats = driver_mem.stop()

    return CPUWorkerResult(
        n_cpu_workers=n_cpu_workers,
        wall_time_s=wall,
        gpu_forward_s=float(gpu_fwd),
        gpu_idle_s=max(0.0, wall - float(gpu_fwd)),
        preprocess_total_s=float(t_pre_total),
        n_stations=len(streams),
        n_windows=int(max(total_windows, n_windows_reported)),
        batch_size=batch_size,
        n_gpu_submits=int(n_calls),
        infer_num_threads=int(infer_num_threads) if is_cpu_infer else -1,
        infer_device=device,
        peak_rss_bytes_driver=int(driver_mem_stats.peak_rss_bytes),
        peak_rss_bytes_infer=int(peak_rss_infer),
        peak_rss_bytes_worker_max=int(worker_peak_max),
        peak_gpu_mem_bytes_infer=int(peak_gpu_infer),
    )
