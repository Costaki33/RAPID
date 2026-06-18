"""Shared fairness core for apples-to-apples phase-picking benchmarks.

Every method (native annotate / classify / slipstream) and every orchestration
strategy (Ripper / Model-Actor / Model-Actor-Slipstream) is measured against a
SINGLE, identical contract so that the same JSON field can be compared directly
across methods:

* Workload: one fixed window per station (``window_samples`` long), taken from
  the start of each rebuilt synthetic trace (which contains both the labelled P
  and S arrival). ``n_windows == n_stations`` for every method.
* Timing: the FULL pipeline, split into per-stage categories so we can see which
  stage dominates -- ``framework_init`` -> ``model_load`` -> ``waveform_access``
  -> ``preprocess`` -> ``inference`` -> ``pick_generation`` (+ ``total``).
* Threads: pinned to the CPU budget (``torch``/TF intra+inter = ``n_cpus``) so a
  trial never oversubscribes its affinity mask.
* Memory: process-tree RSS peak/baseline/growth via the shared sampler.
* Pick quality: predicted P/S vs the labelled P/S of the originating STEAD/TXED
  trace (the manifest catalog), identical tolerance for all methods.

This module holds the primitives; the runners wire them into each execution
strategy. Schema version 2 is emitted by :func:`build_result`.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class GpuVramSampler:
    """Process-tree VRAM (MB) sampler via NVML, mirroring the RAM trio.

    Tracks the VRAM attributable to THIS process tree (the trial process and all
    of its children) using ``nvmlDeviceGetComputeRunningProcesses`` filtered to
    our PIDs -- the same PID-isolated method eqcctpro uses. This is consistent
    across methods: native annotate/slipstream allocate CUDA in-process, native
    classify in pool-worker children, and orchestration inside Ray-actor children
    -- all are in the tree, so all are counted. Exposes:

    * ``baseline_mb``  -- tree VRAM at start (before model load, ~0)
    * ``peak_mb``      -- peak tree VRAM during the run (summed over tracked GPUs)
    * ``end_mb``       -- tree VRAM at stop (process_tree_vram_mb)

    When ``gpu_indices`` names more than one device (e.g. a Model-Actor pool
    spread across both GPUs), the aggregate ``*_mb`` fields sum over the devices
    and the per-device breakdown is exposed via ``per_gpu_peak_mb`` /
    ``per_gpu_end_mb`` / ``per_gpu_baseline_mb`` (keyed by physical NVML index).
    NVML indices are physical and independent of ``CUDA_VISIBLE_DEVICES``, so
    GPU0 and GPU1 are attributed correctly regardless of Ray's per-actor mapping.
    """

    def __init__(self, process=None, gpu_index: int = 0, interval_s: float = 0.1,
                 gpu_indices: Optional[List[int]] = None):
        import psutil

        self.process = process or psutil.Process()
        self.gpu_indices = list(gpu_indices) if gpu_indices is not None else [gpu_index]
        self.gpu_index = self.gpu_indices[0]
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handles: Dict[int, object] = {}
        self.baseline_mb = 0.0
        self.peak_mb = 0.0
        self.end_mb = 0.0
        self.per_gpu_baseline_mb: Dict[int, float] = {}
        self.per_gpu_peak_mb: Dict[int, float] = {}
        self.per_gpu_end_mb: Dict[int, float] = {}

    def _our_pids(self) -> set:
        import psutil

        pids = {self.process.pid}
        try:
            for c in self.process.children(recursive=True):
                try:
                    pids.add(c.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            pass
        return pids

    def _tree_vram_by_gpu(self) -> Dict[int, float]:
        """Per-device tree VRAM (MB), filtered to our PIDs."""
        import pynvml

        our = self._our_pids()
        out: Dict[int, float] = {}
        for idx, handle in self._handles.items():
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except Exception:
                out[idx] = 0.0
                continue
            total = 0.0
            for p in procs:
                if p.pid in our and getattr(p, "usedGpuMemory", None):
                    total += p.usedGpuMemory / 1e6
            out[idx] = total
        return out

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                by_gpu = self._tree_vram_by_gpu()
                total = sum(by_gpu.values())
                self.end_mb = total
                if total > self.peak_mb:
                    self.peak_mb = total
                for idx, v in by_gpu.items():
                    self.per_gpu_end_mb[idx] = v
                    if v > self.per_gpu_peak_mb.get(idx, 0.0):
                        self.per_gpu_peak_mb[idx] = v
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            for idx in self.gpu_indices:
                self._handles[idx] = pynvml.nvmlDeviceGetHandleByIndex(idx)
            by_gpu = self._tree_vram_by_gpu()
            self.per_gpu_baseline_mb = dict(by_gpu)
            self.per_gpu_peak_mb = dict(by_gpu)
            self.per_gpu_end_mb = dict(by_gpu)
            self.baseline_mb = sum(by_gpu.values())
            self.peak_mb = self.baseline_mb
            self.end_mb = self.baseline_mb
        except Exception:
            self._handles = {}
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            if self._handles:
                by_gpu = self._tree_vram_by_gpu()
                self.end_mb = sum(by_gpu.values())
                for idx, v in by_gpu.items():
                    self.per_gpu_end_mb[idx] = v
        except Exception:
            pass
        return self.peak_mb

class ResourceUsageSampler:
    """Low-overhead process-tree resource sampler for one repeat.

    Samples on a fixed interval (default 0.25s) and reports, per repeat:

    * CPU: process-tree user+system seconds (last-seen per PID, so exited
      children keep their final observed value) and mean utilisation of the
      pinned cores (``cpu_busy_s / (wall_s * n_cores)``).
    * Disk: process-tree read/write bytes via ``/proc/<pid>/io``.
    * GPU (when ``gpu_index`` is set): device utilisation %, power draw, and
      energy via NVML total-energy counter (falls back to integrating power
      samples). The scheduler gives each GPU trial exclusive use of its device,
      so device-level numbers are attributable to the trial.
    * RAPL: host package energy from ``/sys/class/powercap``. This is
      HOST-WIDE (all concurrent trials share the package), hence the
      ``host_`` prefix -- use it for cross-checks, not per-trial attribution.
    """

    _RAPL_GLOB = "/sys/class/powercap/intel-rapl:[0-9]*"

    def __init__(self, process=None, gpu_index: Optional[int] = None,
                 n_cores: int = 1, interval_s: float = 0.25):
        import psutil

        self.process = process or psutil.Process()
        self.gpu_index = gpu_index
        self.n_cores = max(1, int(n_cores))
        self.interval_s = max(0.05, float(interval_s))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cpu_by_pid: Dict[int, Tuple[float, float]] = {}
        self._io_by_pid: Dict[int, Tuple[int, int]] = {}
        self._gpu_util: List[float] = []
        self._gpu_power_w: List[float] = []
        self._nvml_handle = None
        self._nvml_energy_start_mj: Optional[int] = None
        self._rapl_start: Dict[str, int] = {}
        self._t0 = 0.0
        self._t1 = 0.0

    def _rapl_read(self) -> Dict[str, int]:
        import glob

        out: Dict[str, int] = {}
        for d in glob.glob(self._RAPL_GLOB):
            if ":" in os.path.basename(d).split("intel-rapl:")[-1]:
                continue  # package domains only, skip subdomains (core/dram)
            try:
                with open(os.path.join(d, "energy_uj")) as f:
                    out[d] = int(f.read().strip())
            except (OSError, ValueError):
                pass
        return out

    def _sample_tree(self) -> None:
        import psutil

        procs = [self.process]
        try:
            procs += self.process.children(recursive=True)
        except Exception:
            pass
        for p in procs:
            try:
                ct = p.cpu_times()
                self._cpu_by_pid[p.pid] = (float(ct.user), float(ct.system))
            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                continue
            try:
                io = p.io_counters()
                self._io_by_pid[p.pid] = (int(io.read_bytes), int(io.write_bytes))
            except Exception:
                pass

    def _sample_gpu(self) -> None:
        if self._nvml_handle is None:
            return
        import pynvml

        try:
            self._gpu_util.append(float(pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu))
        except Exception:
            pass
        try:
            self._gpu_power_w.append(pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0)
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample_tree()
                self._sample_gpu()
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> "ResourceUsageSampler":
        self._t0 = time.monotonic()
        if self.gpu_index is not None:
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(int(self.gpu_index))
                try:
                    self._nvml_energy_start_mj = int(
                        pynvml.nvmlDeviceGetTotalEnergyConsumption(self._nvml_handle)
                    )
                except Exception:
                    self._nvml_energy_start_mj = None
            except Exception:
                self._nvml_handle = None
        self._rapl_start = self._rapl_read()
        self._sample_tree()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self._sample_tree()
            self._sample_gpu()
        except Exception:
            pass
        self._t1 = time.monotonic()
        wall = max(1e-9, self._t1 - self._t0)
        cpu_user = sum(u for u, _ in self._cpu_by_pid.values())
        cpu_sys = sum(s for _, s in self._cpu_by_pid.values())
        busy = cpu_user + cpu_sys
        out: Dict[str, Any] = {
            "wall_s": round(wall, 3),
            "cpu_user_s": round(cpu_user, 3),
            "cpu_system_s": round(cpu_sys, 3),
            "cpu_util_frac_of_cores": round(busy / (wall * self.n_cores), 4),
            "disk_read_mb": round(sum(r for r, _ in self._io_by_pid.values()) / 1e6, 3),
            "disk_write_mb": round(sum(w for _, w in self._io_by_pid.values()) / 1e6, 3),
        }
        if self._nvml_handle is not None:
            if self._gpu_util:
                out["gpu_util_mean_pct"] = round(sum(self._gpu_util) / len(self._gpu_util), 2)
                out["gpu_util_max_pct"] = round(max(self._gpu_util), 2)
            if self._gpu_power_w:
                out["gpu_power_mean_w"] = round(sum(self._gpu_power_w) / len(self._gpu_power_w), 2)
                out["gpu_power_max_w"] = round(max(self._gpu_power_w), 2)
            energy_j: Optional[float] = None
            if self._nvml_energy_start_mj is not None:
                try:
                    import pynvml

                    end_mj = int(pynvml.nvmlDeviceGetTotalEnergyConsumption(self._nvml_handle))
                    energy_j = (end_mj - self._nvml_energy_start_mj) / 1000.0
                except Exception:
                    energy_j = None
            if energy_j is None and self._gpu_power_w:
                energy_j = (sum(self._gpu_power_w) / len(self._gpu_power_w)) * wall
            if energy_j is not None:
                out["gpu_energy_j"] = round(max(0.0, energy_j), 2)
        rapl_end = self._rapl_read()
        if self._rapl_start and rapl_end:
            tot_uj = 0
            for d, v0 in self._rapl_start.items():
                v1 = rapl_end.get(d)
                if v1 is None:
                    continue
                if v1 < v0:  # counter wrapped
                    try:
                        with open(os.path.join(d, "max_energy_range_uj")) as f:
                            v1 += int(f.read().strip())
                    except (OSError, ValueError):
                        continue
                tot_uj += v1 - v0
            if tot_uj > 0:
                out["host_cpu_package_energy_j"] = round(tot_uj / 1e6, 2)
        return out


# Canonical, ordered pipeline stages. Every trial reports a duration for each
# (0.0 if a stage does not apply to that execution strategy) so columns align.
# "warmup" is the first-inference cost (CUDA init / cuDNN autotune /
# torch.compile) measured as its own stage in EVERY family, so it is neither
# silently excluded (old native behaviour) nor folded into inference (old
# orchestration behaviour).
STAGES: Tuple[str, ...] = (
    "framework_init",
    "model_load",
    "waveform_access",
    "preprocess",
    "warmup",
    "inference",
    "pick_generation",
)

SCHEMA_VERSION = 3


# ---------------------------------------------------------------------------
# Thread / core fairness
# ---------------------------------------------------------------------------


def pin_threads(n_cpus: int) -> None:
    """Cap every compute library to ``n_cpus`` threads to match the affinity mask.

    Sets the standard OpenMP/BLAS env vars AND, crucially, ``torch`` intra-op and
    inter-op thread counts (env vars alone are ignored once torch has imported).
    TensorFlow intra/inter are set best-effort only if TF is already importable so
    we never force a heavy import for pure-PyTorch native runs. Safe to call once
    per process (e.g. at the start of each scheduler subprocess).
    """
    n = max(1, int(n_cpus))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    ):
        os.environ[key] = str(n)

    try:
        import torch

        torch.set_num_threads(n)
        # set_num_interop_threads can only be set before any inter-op work and
        # only once; ignore if torch has already started its pool.
        try:
            torch.set_num_interop_threads(n)
        except RuntimeError:
            pass
    except Exception:
        pass

    # TensorFlow only matters for the EQCCT/orchestration TF actors. Don't force
    # an import; configure only if it is already loaded.
    import sys

    if "tensorflow" in sys.modules:
        try:
            tf = sys.modules["tensorflow"]
            tf.config.threading.set_intra_op_parallelism_threads(n)
            tf.config.threading.set_inter_op_parallelism_threads(n)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Per-stage timing
# ---------------------------------------------------------------------------


@dataclass
class StageTimes:
    """Accumulates per-stage wall seconds for one repeat."""

    times: Dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in STAGES})

    @contextmanager
    def stage(self, name: str):
        if name not in self.times:
            raise KeyError(f"unknown stage {name!r}; expected one of {STAGES}")
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] += time.perf_counter() - t0

    def add(self, name: str, seconds: float) -> None:
        self.times[name] = self.times.get(name, 0.0) + float(seconds)

    def as_repeat(self, *, success: bool = True, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rec: Dict[str, Any] = {f"{s}_s": round(self.times.get(s, 0.0), 6) for s in STAGES}
        rec["total_s"] = round(sum(self.times.get(s, 0.0) for s in STAGES), 6)
        rec["success"] = bool(success)
        if extra:
            rec.update(extra)
        return rec


# ---------------------------------------------------------------------------
# Single-window workload construction
# ---------------------------------------------------------------------------


def single_window_from_array(arr: np.ndarray, window_samples: int) -> np.ndarray:
    """Return exactly one ``(C, window_samples)`` window from the start of ``arr``.

    Pads with zeros if the trace is shorter than ``window_samples`` (should not
    happen with the rebuilt 6000-sample networks, but kept defensive).
    """
    c, t = arr.shape
    if t >= window_samples:
        return arr[:, :window_samples]
    out = np.zeros((c, window_samples), dtype=arr.dtype)
    out[:, :t] = arr
    return out


def build_single_window_batch(
    model,
    streams: Sequence[Tuple[str, Any]],
    window_samples: int,
    *,
    component_order: Optional[str] = None,
) -> Tuple[List[str], np.ndarray]:
    """Preprocess each station stream and stack one fixed window per station.

    Returns ``(station_ids, windows)`` where ``windows`` is
    ``(n_stations, 3, window_samples)`` float32. One window per station, so
    ``n_windows == n_stations``.
    """
    from rapid.data import preprocess_for_model, stream_to_3c_array

    co = component_order or getattr(model, "component_order", None) or "ZNE"
    station_ids: List[str] = []
    wins: List[np.ndarray] = []
    for sta, st in streams:
        pre = preprocess_for_model(model, st)
        arr = stream_to_3c_array(pre, component_order=co)
        if arr is None:
            continue
        wins.append(single_window_from_array(arr, window_samples))
        station_ids.append(sta)
    if not wins:
        return [], np.empty((0, 3, window_samples), dtype=np.float32)
    return station_ids, np.stack(wins, axis=0).astype(np.float32, copy=False)


def window_starts(T: int, in_samples: int, overlap_samples: int) -> List[int]:
    """Window start offsets covering ``[0, T)`` with the given overlap.

    Mirrors ``rapid.data.window_from_array`` / the slipstream actor so native and
    orchestration produce identical windows. For ``in_samples == T`` (single-window
    regimes) this returns ``[0]``; for a 3001-sample window over a 6000-sample
    trace at overlap 0.3 it returns the production sliding windows plus a tail.
    """
    if T <= in_samples:
        return [0]
    step = max(1, in_samples - overlap_samples)
    starts = list(range(0, T - in_samples + 1, step))
    if not starts:
        starts = [0]
    if starts[-1] + in_samples < T:
        starts.append(T - in_samples)
    return starts


def build_windowed_batch(
    model,
    streams: Sequence[Tuple[str, Any]],
    in_samples: int,
    overlap_samples: int,
    *,
    component_order: Optional[str] = None,
) -> Tuple[List[str], np.ndarray, List[int], List[int]]:
    """Preprocess each station stream and slide ``in_samples`` windows over it.

    Returns ``(station_ids, windows, n_per_station, starts)`` where ``windows`` is
    ``(sum(n_per_station), 3, in_samples)`` float32. ``starts`` are the shared
    per-window offsets (same for every station since all traces share a length).
    ``n_windows == len(windows)``; for single-window regimes this equals
    ``n_stations``.
    """
    from rapid.data import preprocess_for_model, stream_to_3c_array

    co = component_order or getattr(model, "component_order", None) or "ZNE"
    station_ids: List[str] = []
    blocks: List[np.ndarray] = []
    n_per: List[int] = []
    shared_starts: Optional[List[int]] = None
    for sta, st in streams:
        pre = preprocess_for_model(model, st)
        arr = stream_to_3c_array(pre, component_order=co)
        if arr is None:
            continue
        T = arr.shape[1]
        starts = window_starts(T, in_samples, overlap_samples)
        wins = np.stack(
            [single_window_from_array(arr[:, s : s + in_samples], in_samples) for s in starts],
            axis=0,
        )
        blocks.append(wins)
        n_per.append(len(starts))
        station_ids.append(sta)
        if shared_starts is None:
            shared_starts = starts
    if not blocks:
        return [], np.empty((0, 3, in_samples), dtype=np.float32), [], []
    windows = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
    return station_ids, windows, n_per, (shared_starts or [0])


def windows_to_station_picks(
    preds: np.ndarray,
    station_ids: Sequence[str],
    n_per_station: Sequence[int],
    starts: Sequence[int],
    *,
    p_idx: int,
    s_idx: int,
    p_threshold: float = 0.3,
    s_threshold: float = 0.3,
    min_separation: int = 50,
) -> Dict[str, Dict[str, List[float]]]:
    """Map windowed model output to per-station trace-relative P/S sample indices.

    A detected onset ``o`` in a window starting at ``starts[wi]`` maps to the
    trace-relative sample ``starts[wi] + o`` (which is what the manifest catalog
    stores), so picks from every regime score against the same ground truth.
    """
    from rapid.quality import extract_picks_simple

    out: Dict[str, Dict[str, List[float]]] = {}
    cursor = 0
    for si, sta in enumerate(station_ids):
        nw = n_per_station[si]
        p_samps: List[float] = []
        s_samps: List[float] = []
        for wlocal in range(nw):
            wi = cursor + wlocal
            w_start = starts[wlocal] if wlocal < len(starts) else 0
            p_onsets = extract_picks_simple(preds[wi, :, p_idx], threshold=p_threshold, min_separation=min_separation)
            s_onsets = extract_picks_simple(preds[wi, :, s_idx], threshold=s_threshold, min_separation=min_separation)
            p_samps.extend(float(w_start + o) for o in p_onsets)
            s_samps.extend(float(w_start + o) for o in s_onsets)
        out[sta] = {"p": p_samps, "s": s_samps}
        cursor += nw
    return out


def extract_single_window_picks(
    preds: np.ndarray,
    station_ids: Sequence[str],
    *,
    p_idx: int,
    s_idx: int,
    p_threshold: float = 0.3,
    s_threshold: float = 0.3,
    min_separation: int = 50,
) -> Dict[str, Dict[str, List[float]]]:
    """Convert single-window model output to per-station P/S sample indices.

    ``preds`` is ``(n_windows, T, C)`` (SeisBench annotate_batch_post layout) with
    one window per station, the window starting at sample 0 of the trace -- so the
    detected onset index IS the trace-relative sample index.
    """
    from rapid.quality import extract_picks_simple

    out: Dict[str, Dict[str, List[float]]] = {}
    n = min(preds.shape[0], len(station_ids))
    for i in range(n):
        sta = station_ids[i]
        p_trace = preds[i, :, p_idx]
        s_trace = preds[i, :, s_idx]
        p_onsets = extract_picks_simple(p_trace, threshold=p_threshold, min_separation=min_separation)
        s_onsets = extract_picks_simple(s_trace, threshold=s_threshold, min_separation=min_separation)
        out[sta] = {
            "p": [float(o) for o in p_onsets],
            "s": [float(o) for o in s_onsets],
        }
    return out


# ---------------------------------------------------------------------------
# Aggregation + unified result schema
# ---------------------------------------------------------------------------


def _agg(vals: List[float]) -> Dict[str, float]:
    import statistics

    if not vals:
        return {}
    return {
        "min": float(min(vals)),
        "mean": float(statistics.mean(vals)),
        "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
    }


def summarize_timing(repeats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """min/mean/std for each stage + total across successful repeats."""
    ok = [r for r in repeats if r.get("success")]
    timing: Dict[str, Any] = {
        "stages": list(STAGES),
        "n_repeats": len(repeats),
        "repeats": repeats,
        "success_rate": (len(ok) / len(repeats)) if repeats else 0.0,
    }
    keys = [f"{s}_s" for s in STAGES] + ["total_s"]
    for key in keys:
        stats = _agg([float(r[key]) for r in ok if r.get(key) is not None])
        for k, v in stats.items():
            timing[f"{key}_{k}"] = v
    return timing


MEMORY_KEYS = [
    "baseline_ram_mb",
    "peak_ram_mb",
    "process_tree_ram_mb",
    "ram_growth_mb",
    # PSS trio: proportional set size sums shared pages once across the tree,
    # so multi-process families (Ray) are not over-counted vs single-process
    # natives. Use *_pss_* for cross-family comparisons; RSS kept for history.
    "baseline_pss_mb",
    "peak_pss_mb",
    "process_tree_pss_mb",
    "pss_growth_mb",
    "baseline_vram_mb",
    "peak_vram_mb",
    "process_tree_vram_mb",
    "vram_growth_mb",
    # Per-physical-GPU VRAM (flat scalars so _agg handles them). Present only for
    # the device(s) a trial used: single-GPU trials emit one, the 2-GPU actor
    # split emits both gpu0 and gpu1.
    "peak_vram_mb_gpu0",
    "peak_vram_mb_gpu1",
    "process_tree_vram_mb_gpu0",
    "process_tree_vram_mb_gpu1",
]

RESOURCE_KEYS = [
    "cpu_user_s",
    "cpu_system_s",
    "cpu_util_frac_of_cores",
    "disk_read_mb",
    "disk_write_mb",
    "gpu_util_mean_pct",
    "gpu_util_max_pct",
    "gpu_power_mean_w",
    "gpu_power_max_w",
    "gpu_energy_j",
    "host_cpu_package_energy_j",
]


def summarize_memory(mem_repeats: List[Dict[str, Any]]) -> Dict[str, Any]:
    mem: Dict[str, Any] = {"repeats": mem_repeats}
    for key in MEMORY_KEYS:
        stats = _agg([float(r[key]) for r in mem_repeats if r.get(key) is not None])
        for k, v in stats.items():
            mem[f"{key}_{k}"] = v
    return mem


def summarize_resources(res_repeats: List[Dict[str, Any]]) -> Dict[str, Any]:
    res: Dict[str, Any] = {"repeats": res_repeats}
    for key in RESOURCE_KEYS:
        stats = _agg([float(r[key]) for r in res_repeats if r.get(key) is not None])
        for k, v in stats.items():
            res[f"{key}_{k}"] = v
    return res


def summarize_pick_quality(pq_repeats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-repeat pick-quality dicts: mean/std/min over every numeric
    metric, with the full per-repeat list preserved so no datapoint is lost."""
    out: Dict[str, Any] = {
        "n_repeats_scored": len(pq_repeats),
        "repeats": pq_repeats,
    }
    if not pq_repeats:
        return out

    def _flat(pq: Dict[str, Any]) -> Dict[str, float]:
        flat: Dict[str, float] = {}
        for k, v in pq.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                flat[k] = float(v)
            elif isinstance(v, dict):  # P / S phase sub-dicts
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float)) and not isinstance(v2, bool):
                        flat[f"{k}.{k2}"] = float(v2)
        return flat

    flats = [_flat(pq) for pq in pq_repeats]
    for key in sorted({k for f in flats for k in f}):
        stats = _agg([f[key] for f in flats if key in f])
        for k, v in stats.items():
            out[f"{key}_{k}"] = v
    return out


def build_result(
    *,
    meta: Dict[str, Any],
    timing_repeats: List[Dict[str, Any]],
    memory_repeats: List[Dict[str, Any]],
    pick_quality: Optional[Dict[str, Any]] = None,
    resource_repeats: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Assemble the unified schema-v3 result dict shared by every method."""
    out: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "meta": meta,
        "timing": summarize_timing(timing_repeats),
        "memory": summarize_memory(memory_repeats),
    }
    if resource_repeats:
        out["resources"] = summarize_resources(resource_repeats)
    if pick_quality is not None:
        out["pick_quality_vs_catalog"] = pick_quality
    return out
