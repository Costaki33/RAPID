"""Inference backends with a common interface.

Every backend implements :class:`~rapid.backends.base.InferenceBackend`.

Available backends
------------------

- :class:`~rapid.backends.baseline.BaselineAnnotate` — unmodified SeisBench
  ``model.annotate(stream)`` call. End-to-end, no internal breakdown.
- :class:`~rapid.backends.lean_pytorch.LeanPyTorchBackend` — bypass
  SeisBench's asyncio pipeline. Run a single batched forward pass on a
  pre-built ``(B, C, T)`` tensor. Supports ``dtype`` ∈ {fp32, fp16, bf16}.
- :class:`~rapid.backends.onnx_rt.ONNXBackend` — ONNX Runtime CPU/GPU.
  Optional: only registered if ``onnxruntime`` is importable.
- :class:`~rapid.backends.tensorrt_rt.TensorRTBackend` — TensorRT engine runner.
  Optional: only registered if ``tensorrt`` and ``pycuda`` are importable.
"""

from .base import InferenceBackend, BackendError, BackendResult
from .baseline import BaselineAnnotate
from .lean_pytorch import LeanPyTorchBackend

_REGISTRY: dict[str, type[InferenceBackend]] = {
    BaselineAnnotate.name: BaselineAnnotate,
    LeanPyTorchBackend.name: LeanPyTorchBackend,
}

try:
    from .onnx_rt import ONNXBackend  # noqa: F401

    _REGISTRY[ONNXBackend.name] = ONNXBackend
except Exception:  # pragma: no cover - optional dep
    ONNXBackend = None  # type: ignore[assignment]

try:
    from .tensorrt_rt import TensorRTBackend  # noqa: F401

    _REGISTRY[TensorRTBackend.name] = TensorRTBackend
except Exception:  # pragma: no cover - optional dep
    TensorRTBackend = None  # type: ignore[assignment]


def get_backend(name: str) -> type[InferenceBackend]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown or unavailable backend {name!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "InferenceBackend",
    "BackendError",
    "BackendResult",
    "BaselineAnnotate",
    "LeanPyTorchBackend",
    "get_backend",
    "available_backends",
]
