"""TensorRT backend.

Loads a prebuilt ``.plan`` engine (see ``rapid.export.build_trt_engine``) and
runs inference with pinned host/device buffers. Supports FP32 and FP16
engines; the specific precision is baked into the engine at build time.

Note: we do SeisBench's ``annotate_batch_pre`` on CPU (or via PyTorch on GPU
if a ``cuda`` device is used) then copy into the TRT input buffer. Post-
processing runs on the host.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .base import BackendError, InferenceBackend


class TensorRTBackend(InferenceBackend):
    name = "tensorrt"
    description = "TensorRT engine inference (FP16 or FP32, pre-exported .plan)."

    def __init__(
        self,
        parent_model: str,
        child_model: str,
        device: str = "cuda:0",
        dtype: str = "fp16",
        engine_path: Optional[str] = None,
        max_batch_size: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent_model, child_model, device, dtype, **kwargs)
        if engine_path is None:
            raise BackendError(
                "TensorRTBackend requires `engine_path`; run scripts/export_models.py first."
            )
        self.engine_path = str(engine_path)
        self.max_batch_size = int(max_batch_size)
        self._engine = None
        self._context = None
        self._stream = None
        self._sb_model = None
        self._trt = None
        self._cuda = None
        self._bindings: List[int] = []
        self._in_host: Optional[np.ndarray] = None
        self._in_device = None
        self._out_buffers: list[tuple[np.ndarray, Any]] = []
        self._in_shape: Optional[tuple[int, ...]] = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        import tensorrt as trt  # type: ignore[import]
        import pycuda.driver as cuda  # type: ignore[import]
        import pycuda.autoinit  # noqa: F401  # type: ignore[import]
        import seisbench.models as sbm

        if not Path(self.engine_path).is_file():
            raise BackendError(f"TensorRT engine not found: {self.engine_path}")

        self._trt = trt
        self._cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(self.engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise BackendError("Failed to deserialize TensorRT engine.")
        self._engine = engine
        self._context = engine.create_execution_context()
        self._stream = cuda.Stream()

        model_cls = getattr(sbm, self.parent_model)
        self._sb_model = model_cls.from_pretrained(self.child_model)
        self._sb_model.eval()
        self.in_samples = getattr(self._sb_model, "in_samples", None)
        self.sampling_rate = getattr(self._sb_model, "sampling_rate", None)
        self.component_order = getattr(self._sb_model, "component_order", None)

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._engine = None
        self._context = None
        self._stream = None
        self._sb_model = None
        self._bindings.clear()
        self._in_host = None
        self._in_device = None
        self._out_buffers.clear()

    # ------------------------------------------------------------------
    def _prepare_buffers(self, input_shape: tuple[int, ...]) -> None:
        """Allocate host/device buffers for a given input shape."""
        trt = self._trt
        cuda = self._cuda
        ctx = self._context
        engine = self._engine
        assert trt is not None and cuda is not None and ctx is not None and engine is not None

        in_name = engine.get_tensor_name(0)
        ctx.set_input_shape(in_name, tuple(input_shape))

        self._bindings = []
        self._out_buffers = []
        self._in_shape = input_shape

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(ctx.get_tensor_shape(name))
            trt_dtype = engine.get_tensor_dtype(name)
            np_dtype = np.dtype(trt.nptype(trt_dtype))
            host = np.empty(shape, dtype=np_dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self._bindings.append(int(dev))
            ctx.set_tensor_address(name, int(dev))
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._in_host = host
                self._in_device = dev
            else:
                self._out_buffers.append((host, dev))

    # ------------------------------------------------------------------
    def infer_batch(self, batch: np.ndarray) -> np.ndarray:
        import torch

        if self._engine is None or self._sb_model is None:
            raise BackendError("Call load() before infer_batch().")

        # SeisBench pre-processing on host.
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(batch)).float()
            argdict: dict[str, Any] = {"sampling_rate": self.sampling_rate}
            pre = self._sb_model.annotate_batch_pre(x, argdict=argdict)
            if isinstance(pre, tuple):
                pre, piggyback = pre
            else:
                piggyback = None
            np_in = pre.detach().cpu().numpy()

        if self.dtype == "fp16":
            np_in = np_in.astype(np.float16)
        else:
            np_in = np_in.astype(np.float32)

        if self._in_shape != np_in.shape:
            self._prepare_buffers(np_in.shape)

        assert self._in_host is not None and self._in_device is not None
        assert self._cuda is not None and self._context is not None and self._stream is not None

        np.copyto(self._in_host, np_in.astype(self._in_host.dtype, copy=False))
        self._cuda.memcpy_htod_async(self._in_device, self._in_host, self._stream)
        ok = self._context.execute_async_v3(stream_handle=self._stream.handle)
        if not ok:
            raise BackendError("TensorRT execute_async_v3 returned False.")
        host_outs = []
        for host, dev in self._out_buffers:
            self._cuda.memcpy_dtoh_async(host, dev, self._stream)
            host_outs.append(host)
        self._stream.synchronize()

        with torch.inference_mode():
            if len(host_outs) == 1:
                raw = torch.from_numpy(host_outs[0].astype(np.float32).copy())
            else:
                raw = [torch.from_numpy(h.astype(np.float32).copy()) for h in host_outs]
            post = self._sb_model.annotate_batch_post(
                raw, piggyback=piggyback, argdict=argdict
            )

        if isinstance(post, (list, tuple)):
            return np.stack([p.detach().cpu().numpy() for p in post], axis=-1)
        return post.detach().cpu().numpy()
