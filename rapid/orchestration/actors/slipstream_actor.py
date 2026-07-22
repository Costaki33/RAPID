"""Ray Model-Actor that runs RAPID Slipstream (lean PyTorch) inside each worker.

Used when ``mseed_predictor(..., slipstream_inference=True)`` so orchestration
benchmarks (Model-Actor at TexNet scale) can be compared against the default
``SeisBenchModelActor`` path that calls ``model.classify()``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import numpy as np
import ray
from obspy import Stream, UTCDateTime

from rapid.orchestration.support.timing_util import cuda_synchronize_best_effort


def _ensure_rapid_on_path() -> None:
    # rapid/orchestration/actors/thisfile.py -> repo root (…/RAPID)
    root = Path(__file__).resolve().parents[3]
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


def _window_start_indices(T: int, in_samples: int, overlap_samples: int) -> List[int]:
    step = max(1, in_samples - overlap_samples)
    if T < in_samples:
        return [0]
    starts = list(range(0, T - in_samples + 1, step))
    if starts[-1] + in_samples < T:
        starts.append(T - in_samples)
    return starts


def _sample_to_utc(stream_start: UTCDateTime, sample_idx: int, sampling_rate: float) -> UTCDateTime:
    return stream_start + float(sample_idx) / float(sampling_rate)


def _preds_to_picks(
    preds: np.ndarray,
    window_starts: List[int],
    stream_start: UTCDateTime,
    sampling_rate: float,
    p_idx: int,
    s_idx: int,
    *,
    p_threshold: float,
    s_threshold: float,
    min_separation: int = 50,
) -> List[Any]:
    from rapid.quality import extract_picks_simple

    picks: List[Any] = []
    n_win = preds.shape[0]
    for wi in range(n_win):
        w_start = window_starts[wi] if wi < len(window_starts) else 0
        p_trace = preds[wi, :, p_idx]
        s_trace = preds[wi, :, s_idx]
        for phase, trace, thr in (("P", p_trace, p_threshold), ("S", s_trace, s_threshold)):
            onsets = extract_picks_simple(trace, threshold=thr, min_separation=min_separation)
            for o in onsets:
                abs_sample = int(w_start) + int(o)
                t = _sample_to_utc(stream_start, abs_sample, sampling_rate)
                prob = float(trace[int(o)]) if int(o) < trace.shape[0] else thr
                picks.append(
                    SimpleNamespace(
                        phase=phase,
                        peak_time=t,
                        start_time=t,
                        time=t,
                        peak_value=prob,
                        score=prob,
                        value=prob,
                    )
                )
    return picks


def lean_classify_stream(
    backend,
    stream: Stream,
    *,
    overlap_samples: int = 0,
    batch_size: int = 256,
    P_threshold: float = 0.3,
    S_threshold: float = 0.3,
    use_gpu: bool = False,
    **kwargs,
):
    """Slipstream (lean PyTorch) classify-equivalent on an ObsPy Stream.

    Shared by the Slipstream Model-Actor and the Ripper slipstream task so both
    orchestration strategies run the IDENTICAL lean pipeline:
    preprocess -> window -> ``backend.infer_chunked`` -> threshold picks.
    Returns ``SimpleNamespace(picks=[...])`` mirroring SeisBench ClassifyOutput.
    """
    _ensure_rapid_on_path()
    from rapid.orchestration.support.timing_util import monotonic_s
    from rapid.data import preprocess_for_model, stream_to_3c_array, WindowSpec, window_from_array
    from rapid.seisbench_precision_eval import phase_indices

    empty_timing = {"preprocess_s": 0.0, "inference_s": 0.0, "pick_extract_s": 0.0}
    if len(stream) == 0:
        return SimpleNamespace(picks=[], stage_timing=empty_timing)

    # Measured stage segments (preprocess / inference / pick extraction) are
    # attached to the output so the orchestration driver reports measured
    # per-stage times instead of reconstructing them from the call wall time.
    t0 = monotonic_s()
    argdict = dict(kwargs) if kwargs else {}
    pre = preprocess_for_model(backend._raw_model, stream, argdict=argdict)
    if len(pre) == 0:
        return SimpleNamespace(picks=[], stage_timing=empty_timing)

    co = backend.component_order or "ZNE"
    arr = stream_to_3c_array(pre, component_order=co)
    if arr is None:
        return SimpleNamespace(picks=[], stage_timing=empty_timing)

    sr = float(backend.sampling_rate or pre[0].stats.sampling_rate)
    stream_start = min(tr.stats.starttime for tr in pre)

    in_samples = int(backend.in_samples)
    spec = WindowSpec(in_samples=in_samples, overlap_samples=int(overlap_samples))
    windows = window_from_array(arr, spec)
    starts = _window_start_indices(arr.shape[1], in_samples, int(overlap_samples))
    preprocess_s = monotonic_s() - t0

    if windows.shape[0] == 0:
        return SimpleNamespace(picks=[], stage_timing={**empty_timing, "preprocess_s": preprocess_s})

    t1 = monotonic_s()
    preds = backend.infer_chunked(windows, batch_size=int(batch_size))
    if use_gpu:
        cuda_synchronize_best_effort()
    inference_s = monotonic_s() - t1
    if preds.ndim == 2:
        preds = preds[:, :, np.newaxis]

    t2 = monotonic_s()
    p_idx, s_idx = phase_indices(backend._raw_model)
    picks = _preds_to_picks(
        preds,
        starts,
        stream_start,
        sr,
        p_idx,
        s_idx,
        p_threshold=P_threshold,
        s_threshold=S_threshold,
    )
    pick_extract_s = monotonic_s() - t2

    return SimpleNamespace(
        picks=picks,
        stage_timing={
            "preprocess_s": preprocess_s,
            "inference_s": inference_s,
            "pick_extract_s": pick_extract_s,
        },
    )


@ray.remote
class SlipstreamSeisBenchModelActor:
    """Persistent actor with a loaded LeanPyTorchBackend (Slipstream)."""

    def __init__(
        self,
        parent_model_name: str,
        child_model_name: str,
        gpus_to_use=False,
        use_gpu: bool = True,
        *,
        slipstream_dtype: str = "bf16",
        slipstream_compile: bool = False,
        overlap_samples: int = 0,
        lean_batch_size: int = 256,
    ):
        self.logger = logging.getLogger("rapid.slipstream_model_actor")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers[:] = []
        self.logger.propagate = False
        self.logger.addHandler(logging.StreamHandler())

        self.use_gpu = use_gpu
        self.gpus_to_use = gpus_to_use
        self.overlap_samples = int(overlap_samples)
        self.lean_batch_size = int(lean_batch_size)

        _ensure_rapid_on_path()
        import torch
        from rapid.backends.lean_pytorch import LeanPyTorchBackend

        if use_gpu:
            self.device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = "cpu"

        self.logger.info(
            "=== SlipstreamSeisBenchModelActor START parent=%s child=%s dtype=%s compile=%s device=%s ===",
            parent_model_name,
            child_model_name,
            slipstream_dtype,
            slipstream_compile,
            self.device_str,
        )

        self.backend = LeanPyTorchBackend(
            parent_model_name,
            child_model_name,
            device=self.device_str,
            dtype=slipstream_dtype,
            compile=slipstream_compile,
        )
        self.backend.load()
        self.parent_model_name = parent_model_name
        self.child_model_name = child_model_name
        self.slipstream_dtype = slipstream_dtype
        self.slipstream_compile = slipstream_compile

    def ready(self):
        return True

    def warmup(self):
        """One dummy forward through the backend (timed by the driver).

        Pays CUDA init / cuDNN autotune / ``torch.compile`` once, up front, so
        per-task inference timings measure the steady state. The driver times
        the pool-wide warmup wall and reports it as the ``warmup`` stage.
        """
        in_samples = int(self.backend.in_samples)
        dummy = np.zeros((1, 3, in_samples), dtype=np.float32)
        _ = self.backend.infer_chunked(dummy, batch_size=1)
        if self.use_gpu:
            cuda_synchronize_best_effort()
        return True

    def classify(
        self,
        stream: Stream,
        P_threshold: float = 0.3,
        S_threshold: float = 0.3,
        Detection_threshold: float = 0.3,
        **kwargs,
    ):
        del Detection_threshold  # same thresholds as classify() for P/S channels
        return lean_classify_stream(
            self.backend,
            stream,
            overlap_samples=self.overlap_samples,
            batch_size=self.lean_batch_size,
            P_threshold=P_threshold,
            S_threshold=S_threshold,
            use_gpu=self.use_gpu,
            **kwargs,
        )

    def classify_array(
        self,
        arr_3c: np.ndarray,
        stream_start_iso: str,
        sampling_rate: float,
        P_threshold: float = 0.3,
        S_threshold: float = 0.3,
        **kwargs,
    ):
        """Infer on a preprocessed (3, T) float32 array — no ObsPy Stream RPC or re-preprocess.

        The orchestration worker runs ``preprocess_for_model`` once (SeisBench parity) and
        sends only the compact array + timing metadata to avoid double bandpass/resample and
        large Stream serialization (see RAPID ``single_gpu.run_single`` megabatch path).
        """
        del kwargs
        _ensure_rapid_on_path()
        from rapid.orchestration.support.timing_util import monotonic_s
        from rapid.data import WindowSpec, window_from_array
        from rapid.seisbench_precision_eval import phase_indices

        empty_timing = {"preprocess_s": 0.0, "inference_s": 0.0, "pick_extract_s": 0.0}
        if arr_3c is None or arr_3c.size == 0:
            return SimpleNamespace(picks=[], stage_timing=empty_timing)

        t0 = monotonic_s()
        arr = np.asarray(arr_3c, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != 3:
            return SimpleNamespace(picks=[], stage_timing=empty_timing)

        stream_start = UTCDateTime(stream_start_iso)
        sr = float(sampling_rate)
        in_samples = int(self.backend.in_samples)
        spec = WindowSpec(in_samples=in_samples, overlap_samples=self.overlap_samples)
        windows = window_from_array(arr, spec)
        starts = _window_start_indices(arr.shape[1], in_samples, self.overlap_samples)
        preprocess_s = monotonic_s() - t0

        if windows.shape[0] == 0:
            return SimpleNamespace(picks=[], stage_timing={**empty_timing, "preprocess_s": preprocess_s})

        t1 = monotonic_s()
        preds = self.backend.infer_chunked(windows, batch_size=self.lean_batch_size)
        if self.use_gpu:
            cuda_synchronize_best_effort()
        inference_s = monotonic_s() - t1
        if preds.ndim == 2:
            preds = preds[:, :, np.newaxis]

        t2 = monotonic_s()
        p_idx, s_idx = phase_indices(self.backend._raw_model)
        picks = _preds_to_picks(
            preds,
            starts,
            stream_start,
            sr,
            p_idx,
            s_idx,
            p_threshold=P_threshold,
            s_threshold=S_threshold,
        )
        pick_extract_s = monotonic_s() - t2

        return SimpleNamespace(
            picks=picks,
            stage_timing={
                "preprocess_s": preprocess_s,
                "inference_s": inference_s,
                "pick_extract_s": pick_extract_s,
            },
        )
