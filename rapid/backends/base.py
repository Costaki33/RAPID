"""Common backend interface.

A backend owns a single loaded model and answers two questions:

1. "Given this prepared ``(B, C, T)`` float32 array of windows, run the model
   and return ``(B, C_out, T_out)`` predictions.
2. Optionally, "load the pretrained model given (parent, child)" — the baseline
   backend keeps SeisBench's path, lean backends extract weights.

All wall-time measurement is captured by the caller using
:class:`rapid.timing.Timer`. Backends only expose hook methods the caller wraps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


class BackendError(RuntimeError):
    pass


@dataclass
class BackendResult:
    """Container for a single backend run.

    Attributes
    ----------
    predictions : np.ndarray | None
        Raw model output of shape ``(B, C_out, T_out)``, or ``None`` for
        end-to-end backends that only produce ObsPy streams (e.g. baseline).
    annotations_stream : Any
        ObsPy ``Stream`` of probability traces (only baseline / classify paths).
    stage_times : dict[str, float]
        Per-stage wall time in seconds.
    metadata : dict[str, Any]
        Arbitrary backend-specific metadata (e.g. dtype, compiled flag).
    """

    predictions: Optional[np.ndarray] = None
    annotations_stream: Any = None
    stage_times: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class InferenceBackend(ABC):
    """Abstract base class for all backends."""

    #: Unique registry name.
    name: str = "abstract"

    #: Human-readable description printed in reports.
    description: str = ""

    def __init__(
        self,
        parent_model: str,
        child_model: str,
        device: str = "cpu",
        dtype: str = "fp32",
        **kwargs: Any,
    ) -> None:
        self.parent_model = parent_model
        self.child_model = child_model
        self.device = device
        self.dtype = dtype
        self.extra = kwargs

        #: Set by ``load()``; exposed so callers can build a matching megabatch.
        self.in_samples: Optional[int] = None
        self.sampling_rate: Optional[float] = None
        self.component_order: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @abstractmethod
    def load(self) -> None:
        """Load the pretrained model onto ``self.device`` in ``self.dtype``."""

    def warmup(self, batch_size: int, n_iters: int = 3) -> None:
        """Feed dummy data through ``infer_batch`` to pay JIT/compile/alloc costs."""
        if self.in_samples is None:
            raise BackendError("Call load() before warmup().")
        x = np.zeros((batch_size, 3, self.in_samples), dtype=np.float32)
        for _ in range(n_iters):
            self.infer_batch(x)

    @abstractmethod
    def close(self) -> None:
        """Release the model and any device memory."""

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @abstractmethod
    def infer_batch(self, batch: np.ndarray) -> np.ndarray:
        """Run one forward pass on ``batch`` of shape ``(B, C, T)``.

        Returns a numpy array of shape ``(B, C_out, T_out)``.
        """

    def infer_chunked(self, mega: np.ndarray, batch_size: int) -> np.ndarray:
        """Split ``mega`` (N, C, T) into fixed-size sub-batches and concatenate."""
        n = mega.shape[0]
        if n == 0:
            return np.empty((0, 0, 0), dtype=np.float32)
        outs = []
        for i in range(0, n, batch_size):
            outs.append(self.infer_batch(mega[i : i + batch_size]))
        return np.concatenate(outs, axis=0)

    # ------------------------------------------------------------------
    # Context manager sugar
    # ------------------------------------------------------------------
    def __enter__(self) -> "InferenceBackend":
        self.load()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
