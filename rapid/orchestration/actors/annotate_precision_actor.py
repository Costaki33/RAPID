"""Ray Model-Actor that runs SeisBench Annotate at FP16/BF16, then classify_aggregate.

Replaces the old Slipstream lean path. Each worker loads a SeisBench model,
casts weights to the requested dtype, runs ``model.annotate(stream)``, and
turns the probability Stream into discrete picks with SeisBench
``classify_aggregate`` (same extractor family as Classify).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import ray
from obspy import Stream

from rapid.orchestration.support.timing_util import cuda_synchronize_best_effort, monotonic_s


def _ensure_rapid_on_path() -> None:
    root = Path(__file__).resolve().parents[3]
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


EQT_FP16_MESSAGE = (
    "EQTransformer cannot run in fp16: it hard-codes -1e10 as a "
    "pooling pad sentinel, which overflows fp16. Use dtype='bf16' "
    "or dtype='fp32'."
)

_DTYPE_MAP = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
}


def _cast_model(model, dtype: str, device_str: str):
    import torch

    dtype = str(dtype).lower()
    parent = type(model).__name__
    if parent == "EQTransformer" and dtype == "fp16":
        raise ValueError(EQT_FP16_MESSAGE)

    device = torch.device(device_str)
    model.to(device)
    if dtype in ("fp16", "bf16"):
        torch_dtype = getattr(torch, _DTYPE_MAP[dtype])
        model.to(torch_dtype)
        # SeisBench annotate feeds FP32 buffers; cast in forward.
        from rapid.api import _wrap_forward_cast

        model = _wrap_forward_cast(model, torch_dtype)
    return model


def annotate_precision_classify_stream(
    model,
    stream: Stream,
    *,
    batch_size: int = 256,
    overlap_samples: int = 0,
    P_threshold: float = 0.3,
    S_threshold: float = 0.3,
    Detection_threshold: float = 0.3,
    use_gpu: bool = False,
    **kwargs,
):
    """Annotate at the model's current dtype, then SeisBench classify_aggregate.

    Returns ``ClassifyOutput`` (or equivalent) with a ``picks`` attribute, plus
    ``stage_timing`` for orchestration accounting.
    """
    empty_timing = {"preprocess_s": 0.0, "inference_s": 0.0, "pick_extract_s": 0.0}
    if len(stream) == 0:
        from types import SimpleNamespace

        return SimpleNamespace(picks=[], stage_timing=empty_timing)

    ann_kw = dict(
        batch_size=int(batch_size),
        overlap=int(overlap_samples),
        strict=False,
        flexible_horizontal_components=True,
    )
    for k in ("batch_size", "overlap", "strict", "flexible_horizontal_components"):
        if k in kwargs:
            ann_kw[k] = kwargs[k]

    t0 = monotonic_s()
    annotations = model.annotate(stream, **ann_kw)
    if use_gpu:
        cuda_synchronize_best_effort()
    annotate_s = monotonic_s() - t0

    argdict = dict(getattr(model, "default_args", {}) or {})
    argdict.update(kwargs)
    argdict.setdefault("P_threshold", P_threshold)
    argdict.setdefault("S_threshold", S_threshold)
    argdict.setdefault("detection_threshold", Detection_threshold)
    # SeisBench PhaseNet uses phase-specific keys like "P_threshold".
    argdict.setdefault("P_threshold", P_threshold)
    argdict["P_threshold"] = P_threshold
    argdict["S_threshold"] = S_threshold
    argdict["detection_threshold"] = Detection_threshold

    t1 = monotonic_s()
    result = model.classify_aggregate(annotations, argdict)
    pick_s = monotonic_s() - t1

    # Attach timing without breaking ClassifyOutput attribute access.
    try:
        result.stage_timing = {
            "preprocess_s": 0.0,
            "inference_s": annotate_s,
            "pick_extract_s": pick_s,
        }
    except Exception:
        from types import SimpleNamespace

        picks = getattr(result, "picks", result)
        result = SimpleNamespace(
            picks=picks,
            stage_timing={
                "preprocess_s": 0.0,
                "inference_s": annotate_s,
                "pick_extract_s": pick_s,
            },
        )
    return result


@ray.remote
class AnnotatePrecisionModelActor:
    """Persistent actor: SeisBench Annotate at FP16/BF16 + classify_aggregate."""

    def __init__(
        self,
        parent_model_name: str,
        child_model_name: str,
        gpus_to_use=False,
        use_gpu: bool = True,
        *,
        annotate_dtype: str = "bf16",
        annotate_compile: bool = False,
        overlap_samples: int = 0,
        annotate_batch_size: int = 256,
        # Backward-compatible aliases from the Slipstream era
        slipstream_dtype: Optional[str] = None,
        slipstream_compile: Optional[bool] = None,
        lean_batch_size: Optional[int] = None,
    ):
        self.logger = logging.getLogger("rapid.annotate_precision_model_actor")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers[:] = []
        self.logger.propagate = False
        self.logger.addHandler(logging.StreamHandler())

        if slipstream_dtype is not None:
            annotate_dtype = slipstream_dtype
        if slipstream_compile is not None:
            annotate_compile = bool(slipstream_compile)
        if lean_batch_size is not None:
            annotate_batch_size = int(lean_batch_size)

        self.use_gpu = use_gpu
        self.gpus_to_use = gpus_to_use
        self.overlap_samples = int(overlap_samples)
        self.annotate_batch_size = int(annotate_batch_size)
        self.annotate_dtype = str(annotate_dtype).lower()
        self.annotate_compile = bool(annotate_compile)

        _ensure_rapid_on_path()
        import torch
        import seisbench.models as sbm

        if use_gpu:
            self.device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = "cpu"

        self.logger.info(
            "=== AnnotatePrecisionModelActor START parent=%s child=%s dtype=%s compile=%s device=%s ===",
            parent_model_name,
            child_model_name,
            self.annotate_dtype,
            self.annotate_compile,
            self.device_str,
        )

        if parent_model_name == "EQTransformer" and self.annotate_dtype == "fp16":
            raise ValueError(EQT_FP16_MESSAGE)

        model = getattr(sbm, parent_model_name).from_pretrained(child_model_name)
        model.eval()
        model = _cast_model(model, self.annotate_dtype, self.device_str)

        if self.annotate_compile:
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                self.logger.warning("torch.compile failed (%s); continuing without compile", e)

        self.model = model
        self.parent_model_name = parent_model_name
        self.child_model_name = child_model_name

    def ready(self):
        return True

    def warmup(self):
        """One dummy annotate on a short zero stream pays CUDA / compile once."""
        import numpy as np
        from obspy import Trace
        from obspy.core import Stats

        in_samples = int(getattr(self.model, "in_samples", 3001) or 3001)
        sr = float(getattr(self.model, "sampling_rate", 100.0) or 100.0)
        co = str(getattr(self.model, "component_order", "ZNE") or "ZNE")
        st = Stream()
        for i, comp in enumerate(co[:3]):
            stats = Stats()
            stats.network = "XX"
            stats.station = "WARM"
            stats.channel = f"HH{comp}"
            stats.sampling_rate = sr
            stats.npts = in_samples
            st += Trace(data=np.zeros(in_samples, dtype=np.float32), header=stats)
        _ = self.model.annotate(
            st,
            batch_size=1,
            overlap=0,
            strict=False,
            flexible_horizontal_components=True,
        )
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
        return annotate_precision_classify_stream(
            self.model,
            stream,
            batch_size=self.annotate_batch_size,
            overlap_samples=self.overlap_samples,
            P_threshold=P_threshold,
            S_threshold=S_threshold,
            Detection_threshold=Detection_threshold,
            use_gpu=self.use_gpu,
            **kwargs,
        )

    def annotate(
        self,
        stream: Stream,
        **kwargs,
    ) -> Stream:
        """Return probability streams only (no pick aggregation)."""
        ann_kw = dict(
            batch_size=self.annotate_batch_size,
            overlap=self.overlap_samples,
            strict=False,
            flexible_horizontal_components=True,
        )
        ann_kw.update(kwargs)
        out = self.model.annotate(stream, **ann_kw)
        if self.use_gpu:
            cuda_synchronize_best_effort()
        return out


# Backward-compatible name used by older import paths.
SlipstreamSeisBenchModelActor = AnnotatePrecisionModelActor


def lean_classify_stream(*args, **kwargs):
    """Removed. Use :func:`annotate_precision_classify_stream`."""
    raise RuntimeError(
        "lean_classify_stream was removed with Slipstream. "
        "Use annotate_precision_classify_stream (SeisBench annotate at "
        "fp16/bf16 + classify_aggregate)."
    )
