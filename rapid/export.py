"""Export SeisBench pretrained models to ONNX, and optionally compile to TensorRT.

Usage (from a Python script or REPL)::

    from rapid.export import to_onnx, build_trt_engine
    to_onnx("PhaseNet", "original", "models_exported/phasenet_original.onnx")
    build_trt_engine(
        "models_exported/phasenet_original.onnx",
        "models_exported/phasenet_original_fp16.plan",
        precision="fp16",
        min_batch=1, opt_batch=228, max_batch=1024,
        in_samples=3001,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np


LOG = logging.getLogger("rapid.export")


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def _infer_example_shape(parent: str, in_samples: int) -> tuple[int, int, int]:
    # All SeisBench pickers we benchmark are 3-component.
    return (1, 3, in_samples)


def to_onnx(
    parent: str,
    child: str,
    out_path: str | Path,
    opset: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Export a SeisBench pretrained model's forward pass to ONNX.

    Only the raw forward is traced — ``annotate_batch_pre`` / ``annotate_batch_post``
    stay in Python / PyTorch so per-model quirks (EQT's detection head stacking,
    PhaseNet's transpose + blinding, normalization) remain correct across backends.
    """
    import torch
    import seisbench.models as sbm

    model_cls = getattr(sbm, parent)
    model = model_cls.from_pretrained(child)
    model.eval()
    in_samples = getattr(model, "in_samples")
    example = torch.zeros(_infer_example_shape(parent, in_samples), dtype=torch.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"waveform": {0: "batch"}}

    torch.onnx.export(
        model,
        example,
        str(out_path),
        input_names=["waveform"],
        output_names=_infer_output_names(model),
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    LOG.info("Exported %s/%s -> %s", parent, child, out_path)
    return out_path


def _infer_output_names(model) -> list[str]:
    # EQTransformer returns (detection, P, S); others return a single tensor.
    name = type(model).__name__
    if name == "EQTransformer":
        return ["detection", "p_prob", "s_prob"]
    return ["prediction"]


# ---------------------------------------------------------------------------
# TensorRT engine build
# ---------------------------------------------------------------------------


def build_trt_engine(
    onnx_path: str | Path,
    out_path: str | Path,
    *,
    precision: Literal["fp32", "fp16"] = "fp16",
    min_batch: int = 1,
    opt_batch: int = 228,
    max_batch: int = 1024,
    in_samples: int = 3001,
    workspace_mb: int = 4096,
) -> Path:
    """Compile an ONNX file into a TensorRT engine (``.plan``).

    The engine uses an optimization profile so batch size can vary between
    ``min_batch`` and ``max_batch`` with ``opt_batch`` as the tuning target.
    """
    try:
        import tensorrt as trt  # type: ignore[import]
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "tensorrt is not installed; install nvidia-tensorrt matching your CUDA."
        ) from e

    onnx_path = Path(onnx_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                LOG.error("ONNX parse error: %s", parser.get_error(i))
            raise RuntimeError(f"Failed to parse {onnx_path}")

    config = builder.create_builder_config()
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))
    except Exception:  # pragma: no cover - older TRT APIs
        config.max_workspace_size = workspace_mb * (1 << 20)

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            LOG.warning("Platform does not advertise fast FP16; engine will still build.")
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    in_tensor = network.get_input(0)
    profile.set_shape(
        in_tensor.name,
        (min_batch, 3, in_samples),
        (opt_batch, 3, in_samples),
        (max_batch, 3, in_samples),
    )
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TensorRT engine build returned None.")
    out_path.write_bytes(bytes(engine_bytes))
    LOG.info("Built TRT engine %s (precision=%s)", out_path, precision)
    return out_path
