"""Baseline backend: unmodified SeisBench ``model.annotate(stream)``.

This is the reference we are trying to beat. It takes an ObsPy ``Stream`` in
and returns an ObsPy ``Stream`` of probability traces out — the exact public
API every SeisBench user runs today.

The baseline does *all* its preprocessing internally (filter/resample/taper)
because that's what real users do. For a fair comparison, lean backends use
SeisBench's own ``annotate_stream_pre`` (same filter/resample) on the way in.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import BackendError, InferenceBackend


class BaselineAnnotate(InferenceBackend):
    name = "baseline_annotate"
    description = "Plain SeisBench model.annotate(stream) — no modification."

    def __init__(
        self,
        parent_model: str,
        child_model: str,
        device: str = "cpu",
        dtype: str = "fp32",
        **kwargs: Any,
    ) -> None:
        if dtype not in ("fp32",):
            # SeisBench pretrained models are FP32 by default; we don't override
            # their dtype in the baseline so the baseline is truly the baseline.
            raise BackendError("Baseline backend only supports dtype='fp32'.")
        super().__init__(parent_model, child_model, device, dtype, **kwargs)
        self._model = None
        self._default_annotate_kwargs: Dict[str, Any] = kwargs.get(
            "annotate_kwargs", {}
        )

    def load(self) -> None:
        import seisbench.models as sbm
        import torch

        model_cls = getattr(sbm, self.parent_model)
        self._model = model_cls.from_pretrained(self.child_model)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            self._model.to(torch.device(self.device))
        self._model.eval()
        self.in_samples = getattr(self._model, "in_samples", None)
        self.sampling_rate = getattr(self._model, "sampling_rate", None)
        self.component_order = getattr(self._model, "component_order", None)

    def close(self) -> None:
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # The baseline is not a per-batch backend; it's end-to-end on Streams.
    def infer_batch(self, batch: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise BackendError(
            "BaselineAnnotate does not expose infer_batch(); call annotate_stream() instead."
        )

    def annotate_stream(
        self,
        stream,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Run ``model.annotate(stream)`` and return the resulting ObsPy Stream."""
        if self._model is None:
            raise BackendError("Call load() first.")
        kwargs = dict(self._default_annotate_kwargs)
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return self._model.annotate(stream, **kwargs)

    @property
    def model(self):
        """Underlying SeisBench model (for preprocessing-parity by lean backends)."""
        return self._model
