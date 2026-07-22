"""RAPID — Resource-Aware Parallel Inference Dispatcher.

Lean inference (Slipstream), fair deployment benchmarks, single-process
SeisBench helpers, and Ray orchestration (Model-Actor / Ripper).
"""

from importlib import metadata as _md

from .api import annotate, classify, slipstream

try:
    from .orchestration import (
        MODELS,
        EvaluateSystem,
        OptimalCPUConfigurationFinder,
        OptimalGPUConfigurationFinder,
        RunEQCCTPro,
        load_network_meta,
        model_actor,
        pick,
        ripper,
    )
except ImportError:  # pragma: no cover - missing optional Ray stack
    MODELS = None  # type: ignore[assignment]
    EvaluateSystem = None  # type: ignore[assignment]
    OptimalCPUConfigurationFinder = None  # type: ignore[assignment]
    OptimalGPUConfigurationFinder = None  # type: ignore[assignment]
    RunEQCCTPro = None  # type: ignore[assignment]
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
    "RunEQCCTPro",
    "EvaluateSystem",
    "OptimalCPUConfigurationFinder",
    "OptimalGPUConfigurationFinder",
    "__version__",
]
