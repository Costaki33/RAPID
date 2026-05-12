"""Single-instance runner (one backend, one device).

Runs end-to-end for a list of stations: preprocess, window, forward, post. All
stages are timed. Returns a :class:`RunResult` with per-stage durations plus
raw predictions (for quality comparison).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import obspy
import torch

from ..backends.base import InferenceBackend
from ..data import (
    Megabatch,
    WindowSpec,
    build_megabatch,
    preprocess_for_model,
    stream_to_3c_array,
)
from ..timing import Timer


@dataclass
class RunResult:
    backend_name: str
    model: str
    device: str
    dtype: str
    n_stations: int
    n_windows: int
    batch_size: int
    stage_times: Dict[str, float]
    predictions: Optional[np.ndarray] = None
    annotations_stream: Any = None
    station_ids: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    forward_only_s: Optional[float] = None
    forward_calls: Optional[int] = None

    @property
    def total_s(self) -> float:
        return float(sum(self.stage_times.values()))

    @property
    def throughput_stations_per_s(self) -> float:
        t = self.total_s
        if t <= 0:
            return float("nan")
        return self.n_stations / t


class _ForwardHookTimer:
    """Forward hook that sums wall time spent inside ``module.forward()``.

    Mirrors what the dedicated profiling script does so the matrix's
    ``forward_only_s`` numbers are apples-to-apples with any standalone
    benchmark: it synchronises CUDA before and after each forward call and
    accumulates the elapsed time across calls.
    """

    def __init__(self, module: Optional[torch.nn.Module], device: str):
        self.module = module
        self.device = device
        self.use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        self.total_s: float = 0.0
        self.n_calls: int = 0
        self._t0: Optional[float] = None
        self._pre = None
        self._post = None

    def _on_pre(self, module, inputs):
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        self._t0 = time.perf_counter()

    def _on_post(self, module, inputs, output):
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        if self._t0 is not None:
            self.total_s += time.perf_counter() - self._t0
            self.n_calls += 1
            self._t0 = None

    def __enter__(self):
        self.total_s = 0.0
        self.n_calls = 0
        if self.module is not None:
            self._pre = self.module.register_forward_pre_hook(self._on_pre)
            self._post = self.module.register_forward_hook(self._on_post)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._pre is not None:
            self._pre.remove()
        if self._post is not None:
            self._post.remove()


# ---------------------------------------------------------------------------
# Baseline runner (calls model.annotate(stream) directly)
# ---------------------------------------------------------------------------


def run_baseline_single(
    backend,  # BaselineAnnotate instance (already loaded)
    streams: List[Tuple[str, obspy.Stream]],
    merge_into_one_stream: bool = True,
    annotate_kwargs: Optional[Dict[str, Any]] = None,
) -> RunResult:
    """Call ``model.annotate()`` once on the combined stream (offline batch mode).

    This matches the Table 1 "Ann" column semantics (merged-stream batch call).
    """
    t = Timer(device=backend.device if backend.device.startswith("cuda") else None)

    with t.stage("merge_streams"):
        if merge_into_one_stream:
            merged = obspy.Stream()
            for _, s in streams:
                merged += s
        else:
            merged = streams[0][1] if streams else obspy.Stream()

    fwd_module = getattr(backend, "_model", None)
    with _ForwardHookTimer(fwd_module, backend.device) as fh:
        with t.stage("annotate_end_to_end"):
            annotations = backend.annotate_stream(merged, extra_kwargs=annotate_kwargs)
        fwd_only = fh.total_s
        fwd_calls = fh.n_calls

    return RunResult(
        backend_name=backend.name,
        model=f"{backend.parent_model}/{backend.child_model}",
        device=backend.device,
        dtype=backend.dtype,
        n_stations=len(streams),
        n_windows=-1,  # not known in baseline path
        batch_size=-1,
        stage_times=t.report(),
        annotations_stream=annotations,
        station_ids=[s for s, _ in streams],
        extra={"merged_stream_len": len(merged)},
        forward_only_s=float(fwd_only),
        forward_calls=int(fwd_calls),
    )


# ---------------------------------------------------------------------------
# Lean / ONNX / TRT runner (calls infer_batch on a megabatch)
# ---------------------------------------------------------------------------


def run_lean_single(
    backend: InferenceBackend,
    streams: List[Tuple[str, obspy.Stream]],
    batch_size: int,
    overlap_samples: int = 0,
    argdict: Optional[Dict[str, Any]] = None,
    warmup_iters: int = 1,
) -> RunResult:
    """Preprocess → window → single big batch → forward → post.

    Parameters
    ----------
    backend
        A loaded backend instance (lean PyTorch, ONNX, TensorRT).
    streams
        List of ``(station_id, ObsPy Stream)`` pairs (raw, post-demean only).
    batch_size
        Sub-batch size for ``infer_chunked``. Total windows may exceed this.
    overlap_samples
        Per-window overlap for sliding windows. ``0`` means no overlap.
    """
    if backend.in_samples is None:
        raise RuntimeError("Backend not loaded (in_samples is None).")

    timer_dev = backend.device if backend.device.startswith("cuda") else None
    t = Timer(device=timer_dev)

    # ---- 1) Preprocess each station's stream through SeisBench's annotate_stream_pre
    with t.stage("preprocess"):
        sb_model = _resolve_sb_model(backend)
        arrays: List[Tuple[str, np.ndarray]] = []
        for sta, st in streams:
            pre = preprocess_for_model(sb_model, st, argdict=argdict)
            arr = stream_to_3c_array(pre, component_order=backend.component_order or "ZNE")
            if arr is None:
                continue
            arrays.append((sta, arr))

    # ---- 2) Build megabatch of windows
    with t.stage("window_cut_and_stack"):
        spec = WindowSpec(in_samples=backend.in_samples, overlap_samples=overlap_samples)
        mb: Megabatch = build_megabatch(arrays, spec)

    # ---- 3) Warm-up (compile / cuDNN autotune / TRT allocator) — not counted.
    if warmup_iters > 0 and mb.total_windows > 0:
        dummy = np.zeros(
            (min(batch_size, mb.total_windows), 3, backend.in_samples),
            dtype=np.float32,
        )
        for _ in range(warmup_iters):
            backend.infer_batch(dummy)

    # ---- 4) Forward
    # NOTE: do **not** use `a or b` here. ``backend._fwd_model`` is a
    # ``torch._dynamo.OptimizedModule`` when ``torch.compile`` is active, and
    # truthiness checks on that object call ``__len__`` on the wrapped module,
    # which raises ``TypeError`` for nn.Modules like PhaseNet. Use explicit
    # ``is None`` checks to pick the underlying nn.Module to hook.
    fwd_module = getattr(backend, "_fwd_model", None)
    if fwd_module is None:
        fwd_module = getattr(backend, "_model", None)
    with _ForwardHookTimer(fwd_module, backend.device) as fh:
        with t.stage("forward"):
            preds = backend.infer_chunked(mb.windows, batch_size=batch_size)
        fwd_only = fh.total_s
        fwd_calls = fh.n_calls

    return RunResult(
        backend_name=backend.name,
        model=f"{backend.parent_model}/{backend.child_model}",
        device=backend.device,
        dtype=backend.dtype,
        n_stations=len(arrays),
        n_windows=mb.total_windows,
        batch_size=batch_size,
        stage_times=t.report(),
        predictions=preds,
        station_ids=[s for s, _ in arrays],
        extra={"overlap_samples": overlap_samples},
        forward_only_s=float(fwd_only),
        forward_calls=int(fwd_calls),
    )


def _resolve_sb_model(backend: InferenceBackend):
    """Get the underlying SeisBench WaveformModel from any backend variant."""
    # Each backend keeps a reference to the SeisBench model for pre/post parity.
    for attr in ("_raw_model", "_sb_model", "model", "_model"):
        m = getattr(backend, attr, None)
        if m is not None:
            return m
    raise AttributeError(f"Backend {backend.name} exposes no SeisBench model reference.")
