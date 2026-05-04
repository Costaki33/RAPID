"""ONNX Runtime backend.

Loads a pre-exported ``.onnx`` model file (see ``rapid.export.to_onnx``) and
runs inference via ``onnxruntime.InferenceSession``. Supports CPU and CUDA
execution providers.

Note: We apply the SeisBench model's ``annotate_batch_pre`` (normalization,
taper) and ``annotate_batch_post`` (transpose, blinding) in PyTorch on the
host; the exported ONNX graph only contains the raw forward pass. This keeps
per-model quirks correct without needing to trace them all into ONNX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import BackendError, InferenceBackend


class ONNXBackend(InferenceBackend):
    name = "onnx"
    description = "ONNX Runtime (CPU or CUDA) forward pass from a pre-exported .onnx file."

    def __init__(
        self,
        parent_model: str,
        child_model: str,
        device: str = "cpu",
        dtype: str = "fp32",
        onnx_path: Optional[str] = None,
        providers: Optional[list[str]] = None,
        inter_op_num_threads: Optional[int] = None,
        intra_op_num_threads: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent_model, child_model, device, dtype, **kwargs)
        if onnx_path is None:
            raise BackendError(
                "ONNXBackend requires `onnx_path`; run scripts/export_models.py first."
            )
        self.onnx_path = str(onnx_path)
        self.providers = providers
        self.inter_op = inter_op_num_threads
        self.intra_op = intra_op_num_threads
        self._session = None
        self._sb_model = None
        self._input_name: Optional[str] = None
        self._output_names: Optional[list[str]] = None

    def load(self) -> None:
        import onnxruntime as ort
        import seisbench.models as sbm

        if not Path(self.onnx_path).is_file():
            raise BackendError(f"ONNX file not found: {self.onnx_path}")

        providers = self.providers
        if providers is None:
            if self.device.startswith("cuda"):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        sess_opts = ort.SessionOptions()
        if self.inter_op is not None:
            sess_opts.inter_op_num_threads = self.inter_op
        if self.intra_op is not None:
            sess_opts.intra_op_num_threads = self.intra_op
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            self.onnx_path, sess_options=sess_opts, providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        # Keep a SeisBench model around just for pre/post-processing parity.
        model_cls = getattr(sbm, self.parent_model)
        self._sb_model = model_cls.from_pretrained(self.child_model)
        self._sb_model.eval()
        self.in_samples = getattr(self._sb_model, "in_samples", None)
        self.sampling_rate = getattr(self._sb_model, "sampling_rate", None)
        self.component_order = getattr(self._sb_model, "component_order", None)

    def close(self) -> None:
        self._session = None
        self._sb_model = None

    def infer_batch(self, batch: np.ndarray) -> np.ndarray:
        import torch

        if self._session is None or self._sb_model is None:
            raise BackendError("Call load() before infer_batch().")

        x = torch.from_numpy(np.ascontiguousarray(batch)).float()
        argdict: dict[str, Any] = {"sampling_rate": self.sampling_rate}
        with torch.inference_mode():
            pre = self._sb_model.annotate_batch_pre(x, argdict=argdict)
            if isinstance(pre, tuple):
                pre, piggyback = pre
            else:
                piggyback = None

        np_in = pre.detach().cpu().numpy()
        if self.dtype == "fp16":
            np_in = np_in.astype(np.float16)

        outs = self._session.run(self._output_names, {self._input_name: np_in})

        # Convert back into torch for the post-processing (keeps EQT's 3-head
        # stacking + blinding trivial).
        with torch.inference_mode():
            if len(outs) == 1:
                raw = torch.from_numpy(outs[0].astype(np.float32))
            else:
                raw = [torch.from_numpy(o.astype(np.float32)) for o in outs]
            post = self._sb_model.annotate_batch_post(
                raw, piggyback=piggyback, argdict=argdict
            )

        if isinstance(post, (list, tuple)):
            return np.stack([p.detach().cpu().numpy() for p in post], axis=-1)
        return post.detach().cpu().numpy()
