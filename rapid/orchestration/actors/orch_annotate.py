"""Ray workers for annotate-precision orchestration trials.

EQCCT loads both SeisBench branches (``EQCCTP`` + ``EQCCTS``) and runs them
on the same payload. Other models load a single WaveformModel.

Ripper tasks use ``max_calls=1`` so the worker process is torn down after
each classify — the model is not cached across stations.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, List

import ray
from obspy import Stream

from rapid.orchestration.actors.annotate_precision_actor import (
    EQT_FP16_MESSAGE,
    _cast_model,
    _ensure_rapid_on_path,
    annotate_precision_classify_stream,
)
from rapid.orchestration.support.timing_util import cuda_synchronize_best_effort

MODELS = {
    "PhaseNet": {"parent": "PhaseNet", "child": "original", "branches": None},
    "PhaseNetLight": {"parent": "PhaseNetLight", "child": "stead", "branches": None},
    "EQTransformer": {"parent": "EQTransformer", "child": "original", "branches": None},
    "EQT-NC": {
        "parent": "EQTransformer",
        "child": "original_nonconservative",
        "branches": None,
    },
    "EQCCT": {
        "parent": "EQCCT",
        "child": "original",
        "branches": [
            {"parent": "EQCCTP", "child": "original"},
            {"parent": "EQCCTS", "child": "original"},
        ],
    },
}


def branches_of(model_name: str) -> List[dict]:
    spec = MODELS[model_name]
    return spec["branches"] or [{"parent": spec["parent"], "child": spec["child"]}]


def load_precision_models(
    model_name: str,
    dtype: str,
    device_str: str,
    use_gpu: bool,
):
    _ensure_rapid_on_path()
    import seisbench.models as sbm

    dtype = str(dtype).lower()
    models = []
    for br in branches_of(model_name):
        parent, child = br["parent"], br["child"]
        if parent == "EQTransformer" and dtype == "fp16":
            raise ValueError(EQT_FP16_MESSAGE)
        model = getattr(sbm, parent).from_pretrained(child)
        model.eval()
        model = _cast_model(model, dtype, device_str if use_gpu else "cpu")
        models.append(model)
    return models


def run_precision_payload(
    models,
    stream: Stream,
    *,
    batch_size: int,
    overlap_samples: int = 0,
    P_threshold: float = 0.3,
    S_threshold: float = 0.3,
    Detection_threshold: float = 0.3,
    use_gpu: bool = False,
):
    """Annotate (+ classify_aggregate) every branch; merge picks."""
    if len(stream) == 0:
        return SimpleNamespace(
            picks=[],
            stage_timing={
                "preprocess_s": 0.0,
                "inference_s": 0.0,
                "pick_extract_s": 0.0,
            },
        )

    all_picks: List[Any] = []
    inf = 0.0
    pick_s = 0.0
    for model in models:
        out = annotate_precision_classify_stream(
            model,
            stream,
            batch_size=int(batch_size),
            overlap_samples=int(overlap_samples),
            P_threshold=P_threshold,
            S_threshold=S_threshold,
            Detection_threshold=Detection_threshold,
            use_gpu=use_gpu,
        )
        all_picks.extend(getattr(out, "picks", None) or [])
        stg = getattr(out, "stage_timing", {}) or {}
        inf += float(stg.get("inference_s") or 0.0)
        pick_s += float(stg.get("pick_extract_s") or 0.0)
    return SimpleNamespace(
        picks=all_picks,
        stage_timing={
            "preprocess_s": 0.0,
            "inference_s": inf,
            "pick_extract_s": pick_s,
        },
    )


def _warmup_models(models, use_gpu: bool) -> None:
    import numpy as np
    from obspy import Trace
    from obspy.core import Stats

    model0 = models[0]
    in_samples = int(getattr(model0, "in_samples", 3001) or 3001)
    sr = float(getattr(model0, "sampling_rate", 100.0) or 100.0)
    co = str(getattr(model0, "component_order", "ZNE") or "ZNE")
    st = Stream()
    for comp in co[:3]:
        stats = Stats()
        stats.network = "XX"
        stats.station = "WARM"
        stats.channel = f"HH{comp}"
        stats.sampling_rate = sr
        stats.npts = in_samples
        st += Trace(data=np.zeros(in_samples, dtype=np.float32), header=stats)
    for model in models:
        _ = model.annotate(
            st,
            batch_size=1,
            overlap=0,
            strict=False,
            flexible_horizontal_components=True,
        )
    if use_gpu:
        cuda_synchronize_best_effort()


@ray.remote
class OrchAnnotateActor:
    """Persistent Model-Actor: annotate-precision, dual-branch EQCCT."""

    def __init__(
        self,
        model_name: str,
        annotate_dtype: str = "bf16",
        use_gpu: bool = True,
        annotate_batch_size: int = 512,
        overlap_samples: int = 0,
        gpus_to_use=False,
    ):
        self.logger = logging.getLogger("rapid.orch_annotate_actor")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers[:] = []
        self.logger.propagate = False
        self.logger.addHandler(logging.StreamHandler())

        self.model_name = model_name
        self.annotate_dtype = str(annotate_dtype).lower()
        self.use_gpu = bool(use_gpu)
        self.annotate_batch_size = int(annotate_batch_size)
        self.overlap_samples = int(overlap_samples)
        self.gpus_to_use = gpus_to_use

        _ensure_rapid_on_path()
        import torch

        if self.use_gpu and torch.cuda.is_available():
            self.device_str = "cuda:0"
        else:
            self.device_str = "cpu"
            self.use_gpu = False

        self.models = load_precision_models(
            model_name, self.annotate_dtype, self.device_str, self.use_gpu
        )
        self.logger.info(
            "OrchAnnotateActor model=%s dtype=%s device=%s n_branches=%d bs=%d",
            model_name,
            self.annotate_dtype,
            self.device_str,
            len(self.models),
            self.annotate_batch_size,
        )

    def ready(self):
        return True

    def warmup(self):
        _warmup_models(self.models, self.use_gpu)
        return True

    def classify(
        self,
        stream: Stream,
        P_threshold: float = 0.3,
        S_threshold: float = 0.3,
        Detection_threshold: float = 0.3,
        **kwargs,
    ):
        bs = int(kwargs.pop("batch_size", self.annotate_batch_size))
        ov = int(kwargs.pop("overlap", self.overlap_samples))
        return run_precision_payload(
            self.models,
            stream,
            batch_size=bs,
            overlap_samples=ov,
            P_threshold=P_threshold,
            S_threshold=S_threshold,
            Detection_threshold=Detection_threshold,
            use_gpu=self.use_gpu,
        )


@ray.remote(max_calls=1)
def orch_ripper_classify(
    model_name: str,
    annotate_dtype: str,
    use_gpu: bool,
    annotate_batch_size: int,
    overlap_samples: int,
    stream: Stream,
    P_threshold: float = 0.3,
    S_threshold: float = 0.3,
    Detection_threshold: float = 0.3,
):
    """One-shot Ripper task: load model(s), classify, exit (max_calls=1)."""
    _ensure_rapid_on_path()
    import torch

    gpu = bool(use_gpu) and torch.cuda.is_available()
    device_str = "cuda:0" if gpu else "cpu"
    models = load_precision_models(model_name, annotate_dtype, device_str, gpu)
    return run_precision_payload(
        models,
        stream,
        batch_size=int(annotate_batch_size),
        overlap_samples=int(overlap_samples),
        P_threshold=P_threshold,
        S_threshold=S_threshold,
        Detection_threshold=Detection_threshold,
        use_gpu=gpu,
    )
