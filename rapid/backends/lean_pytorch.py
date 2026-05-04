"""Lean PyTorch backend.

Bypasses SeisBench's asyncio pipeline entirely: we load the pretrained weights
via ``from_pretrained``, then feed the forward pass a pre-built ``(B, C, T)``
float tensor. The backend can run in FP32, FP16, or BF16, and can optionally
be ``torch.compile``'d for extra graph-level fusion.

What this buys us (versus ``model.annotate(stream)``):

1. No asyncio queues, no per-fragment Python loop in ``_async_predict``.
2. No per-call ``annotate_stream_pre`` (caller does it once, shared across
   repeats).
3. Pre-built megabatch — one forward pass for all stations at once.
4. Optional ``.half()`` / ``.to(bfloat16)`` + ``torch.compile``.

The model classes in SeisBench wrap BatchNorm and convs, so we rely on
``.eval()`` + ``torch.no_grad()``. We call ``annotate_batch_pre`` and
``annotate_batch_post`` to preserve the model's expected normalization and
post-processing (e.g. PhaseNet mean/std, EQT detection head shaping).
"""

from __future__ import annotations

import contextlib
from typing import Any, Optional

import numpy as np

from .base import BackendError, InferenceBackend

# Re-used by ``rapid.matrix`` for driver-side skips so JSONL records the same
# message without instantiating the backend or Ray actors.
EQT_LEAN_FP16_MESSAGE = (
    "EQTransformer cannot run in fp16: it hard-codes -1e10 as a "
    "pooling pad sentinel, which overflows fp16. Use dtype='bf16' "
    "(same 16-bit throughput, full fp32 exponent range) or "
    "dtype='fp32'."
)


_DTYPE_MAP = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
}


class LeanPyTorchBackend(InferenceBackend):
    name = "lean_pytorch"
    description = (
        "Lean PyTorch forward. Bypasses SeisBench asyncio. "
        "Supports fp32 / fp16 / bf16 and optional torch.compile."
    )

    def __init__(
        self,
        parent_model: str,
        child_model: str,
        device: str = "cpu",
        dtype: str = "fp32",
        compile: bool = False,
        compile_mode: str = "reduce-overhead",
        channels_last: bool = False,
        cudnn_benchmark: bool = True,
        cast_weights: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        cast_weights : bool
            When ``dtype`` is fp16 or bf16 *and* ``cast_weights`` is True, the
            model's parameters are cast to the target dtype via
            ``model.to(dtype)``. This is fastest but can overflow for models
            that hard-code large sentinel constants (e.g. EQTransformer uses
            ``-1e10`` in an ``F.pad`` inside its pooling block, which is out
            of range for fp16).

            When ``cast_weights`` is False (default), parameters stay in FP32
            and the forward pass is wrapped in ``torch.autocast``, which casts
            individual ops at runtime and gracefully handles those sentinels.
            This is the safer default and still uses Tensor Cores on Ampere+.
        """
        if dtype not in _DTYPE_MAP:
            raise BackendError(
                f"dtype must be one of {list(_DTYPE_MAP)}, got {dtype!r}"
            )
        super().__init__(parent_model, child_model, device, dtype, **kwargs)
        self._raw_model = None  # SeisBench wrapper kept for preprocessing parity.
        self._fwd_model = None  # Possibly torch.compile'd + cast model.
        self._torch_dtype = None
        self._torch_device = None
        self.use_compile = compile
        self.compile_mode = compile_mode
        self.channels_last = channels_last
        self.cudnn_benchmark = cudnn_benchmark
        self.cast_weights = cast_weights

    # ------------------------------------------------------------------
    def load(self) -> None:
        import torch
        import seisbench.models as sbm

        # Known limitation: EQTransformer hard-codes a -1e10 sentinel inside
        # its encoder's F.pad(..., value=-1e10), which is out of fp16 range
        # (|-1e10| > 65504). Neither ``model.half()`` nor ``torch.autocast``
        # recovers from this because the constant is a Python literal that
        # gets cast by pad() to match the tensor's dtype. Use bf16 or fp32
        # for EQTransformer / EQT-NC.
        if self.parent_model == "EQTransformer" and self.dtype == "fp16":
            raise BackendError(EQT_LEAN_FP16_MESSAGE)

        model_cls = getattr(sbm, self.parent_model)
        model = model_cls.from_pretrained(self.child_model)
        model.eval()

        tdev = torch.device(self.device if self.device != "cpu" else "cpu")
        tdtype = getattr(torch, _DTYPE_MAP[self.dtype])

        model.to(tdev)
        if self.dtype in ("fp16", "bf16") and self.cast_weights:
            # Cast params (and therefore activations) to the target dtype.
            # Fastest path, but can overflow for models with large sentinels
            # (e.g. EQTransformer). Prefer autocast otherwise.
            model = model.to(tdtype)

        if self.cudnn_benchmark and self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True

        self._raw_model = model
        self._torch_device = tdev
        self._torch_dtype = tdtype
        self.in_samples = getattr(model, "in_samples", None)
        self.sampling_rate = getattr(model, "sampling_rate", None)
        self.component_order = getattr(model, "component_order", None)

        fwd = model
        if self.use_compile:
            try:
                fwd = torch.compile(fwd, mode=self.compile_mode)
            except Exception as e:  # pragma: no cover - torch<2.0 paths
                raise BackendError(f"torch.compile failed: {e}") from e
        self._fwd_model = fwd

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._raw_model = None
        self._fwd_model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def infer_batch(self, batch: np.ndarray) -> np.ndarray:
        import torch

        if self._fwd_model is None or self._raw_model is None:
            raise BackendError("Call load() before infer_batch().")

        if batch.ndim != 3:
            raise BackendError(f"batch must be (B, C, T); got {batch.shape}")

        # Host -> device in the *input* dtype. For autocast paths, keep FP32
        # on device; autocast handles op-level casts. For cast_weights paths,
        # feed the target dtype directly.
        in_dtype = self._torch_dtype if self.cast_weights else torch.float32
        x = torch.from_numpy(np.ascontiguousarray(batch)).to(
            self._torch_device,
            dtype=in_dtype,
            non_blocking=True,
        )

        if self.channels_last and x.dim() == 4:
            x = x.contiguous(memory_format=torch.channels_last)

        argdict: dict[str, Any] = {"sampling_rate": self.sampling_rate}
        use_autocast = (
            self.dtype in ("fp16", "bf16") and not self.cast_weights
        )
        dev_type = self._torch_device.type
        if use_autocast:
            autocast_ctx = torch.autocast(device_type=dev_type, dtype=self._torch_dtype)
        else:
            autocast_ctx = contextlib.nullcontext()

        with torch.inference_mode(), autocast_ctx:
            preprocessed = self._raw_model.annotate_batch_pre(x, argdict=argdict)
            if isinstance(preprocessed, tuple):
                preprocessed, piggyback = preprocessed
            else:
                piggyback = None
            preds = self._fwd_model(preprocessed)
            preds = self._raw_model.annotate_batch_post(
                preds, piggyback=piggyback, argdict=argdict
            )

        # annotate_batch_post returns a single tensor for supported pickers
        # (PhaseNet, EQTransformer, PhaseNetLight, etc.) in shape (B, T, C) or (B, T).
        if isinstance(preds, (list, tuple)):
            preds_np = np.stack(
                [p.detach().float().cpu().numpy() for p in preds], axis=-1
            )
        else:
            preds_np = preds.detach().float().cpu().numpy()
        return preds_np
