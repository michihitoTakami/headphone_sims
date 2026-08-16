"""Probe-cloud heatmaps: metric values over the pinna surface.

Probes are projected onto the x-y plane (the pinna faces the driver along z),
which preserves the anatomical layout for both mesh and planar probe arrays.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def save_probe_heatmap(
    positions: FloatArray,
    values: FloatArray,
    path: str | Path,
    title: str = "",
    label: str = "",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    sc = ax.scatter(
        positions[:, 0] * 1e3,
        positions[:, 1] * 1e3,
        c=values,
        s=22,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
    )
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label=label)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out
