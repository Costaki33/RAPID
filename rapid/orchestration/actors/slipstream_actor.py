"""Compatibility shim: Slipstream actor moved to annotate_precision_actor.

Import ``AnnotatePrecisionModelActor`` / ``annotate_precision_classify_stream``
from ``rapid.orchestration.actors.annotate_precision_actor`` instead.
"""

from __future__ import annotations

from rapid.orchestration.actors.annotate_precision_actor import (  # noqa: F401
    AnnotatePrecisionModelActor,
    SlipstreamSeisBenchModelActor,
    annotate_precision_classify_stream,
    lean_classify_stream,
    _ensure_rapid_on_path,
)

__all__ = [
    "AnnotatePrecisionModelActor",
    "SlipstreamSeisBenchModelActor",
    "annotate_precision_classify_stream",
    "lean_classify_stream",
    "_ensure_rapid_on_path",
]
