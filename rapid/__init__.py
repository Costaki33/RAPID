"""RAPID — Resource-Aware Parallel Inference Dispatcher benchmarking kit.

A minimal, backend-agnostic benchmarking framework for SeisBench pickers that
measures where SeisBench's ``annotate()`` spends wall time and lets us swap in
faster inference paths (lean PyTorch, FP16, BF16, ONNX Runtime, TensorRT).
"""

from importlib import metadata as _md

try:
    __version__ = _md.version("rapid")
except _md.PackageNotFoundError:
    __version__ = "0.0.0"
