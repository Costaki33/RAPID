"""Inference backends with a common interface.

Every backend implements :class:`~rapid.backends.base.InferenceBackend`.

- :class:`~rapid.backends.baseline.BaselineAnnotate` — SeisBench ``annotate()``
- :class:`~rapid.backends.lean_pytorch.LeanPyTorchBackend` — Slipstream lean forward
  (fp32 / fp16 / bf16, optional ``torch.compile``)
"""

from .base import InferenceBackend, BackendError, BackendResult
from .baseline import BaselineAnnotate
from .lean_pytorch import LeanPyTorchBackend

_REGISTRY: dict[str, type[InferenceBackend]] = {
    BaselineAnnotate.name: BaselineAnnotate,
    LeanPyTorchBackend.name: LeanPyTorchBackend,
}


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
