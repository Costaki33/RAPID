"""Ray orchestration: Model-Actor, Ripper, and Annotate-precision actors.

This package holds the production picking runtime that began as EQCCTPro and
now ships as part of RAPID. Importing the high-level helpers below requires
the optional ``orchestration`` extra (Ray).
"""

from .api import MODELS, load_network_meta, model_actor, pick, ripper

try:
    from .runtime.functionality import (
        EvaluateSystem,
        OptimalCPUConfigurationFinder,
        OptimalGPUConfigurationFinder,
        RunEQCCTPro,
    )
except ImportError:  # pragma: no cover - Ray not installed
    EvaluateSystem = None  # type: ignore[assignment]
    OptimalCPUConfigurationFinder = None  # type: ignore[assignment]
    OptimalGPUConfigurationFinder = None  # type: ignore[assignment]
    RunEQCCTPro = None  # type: ignore[assignment]

__all__ = [
    "pick",
    "model_actor",
    "ripper",
    "load_network_meta",
    "MODELS",
    "RunEQCCTPro",
    "EvaluateSystem",
    "OptimalCPUConfigurationFinder",
    "OptimalGPUConfigurationFinder",
]
