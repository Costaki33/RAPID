"""RAPID — Resource-Aware Parallel Inference Dispatcher.

Lean inference (Slipstream), fair deployment benchmarks, and single-process
helpers for SeisBench ``annotate`` / ``classify``. Network-scale Ray
orchestration (Model-Actor / Ripper) is available via ``pick``.
"""

from importlib import metadata as _md

from .api import annotate, classify, slipstream

try:
    from eqcctpro.api import MODELS, load_network_meta, model_actor, pick, ripper
except ImportError:  # pragma: no cover - orchestration dep missing
    MODELS = None  # type: ignore[assignment]
    load_network_meta = None  # type: ignore[assignment]
    model_actor = None  # type: ignore[assignment]
    pick = None  # type: ignore[assignment]
    ripper = None  # type: ignore[assignment]

try:
    __version__ = _md.version("rapid")
except _md.PackageNotFoundError:
    __version__ = "1.0.0"

__all__ = [
    "annotate",
    "classify",
    "slipstream",
    "pick",
    "model_actor",
    "ripper",
    "load_network_meta",
    "MODELS",
    "__version__",
]
