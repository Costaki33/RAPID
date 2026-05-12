"""Shared colors, markers, and linestyles for SeisBench benchmark figures."""

from __future__ import annotations

MODEL_ORDER = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
MODEL_COLORS = {
    "PhaseNet": "#1f77b4",
    "PhaseNetLight": "#ff7f0e",
    "EQTransformer": "#2ca02c",
    "EQT-NC": "#d62728",
}

NST_ORDER = [64, 256, 580]
AFF_CPUS = [12, 16, 20]

# Station count → marker (used in every figure that encodes N_st).
NST_MARKERS = {64: "o", 256: "^", 580: "s"}

# Baseline SeisBench annotate() — use this linestyle everywhere annotate() appears.
LS_ANNOTATE = (0, (10, 3))

# Lean PyTorch: linestyle encodes dtype + torch.compile together.
# FP16 off: solid; FP16 on: loosely dashed; BF16 off: dashed; BF16 on: dash-dot.
def lean_linestyle(dtype: str, compiled: bool) -> str | tuple:
    d = str(dtype or "").lower()
    if d == "fp16":
        return (0, (6, 3)) if compiled else "-"
    if d == "bf16":
        return "-." if compiled else "--"
    return "-" if not compiled else (0, (3, 2, 1, 2))


# Speedup bar colors (condition × same across models)
BAR_ANNOTATE = "#4d4d4d"
BAR_FP16_OFF = "#6baed6"
BAR_FP16_ON = "#08519c"
BAR_BF16_OFF = "#74c476"
BAR_BF16_ON = "#006d2c"
