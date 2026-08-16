"""Pressure-slice visualization from simulation snapshots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import animation
from matplotlib.artist import Artist

FloatArray = npt.NDArray[np.float64]


def save_slice_animation(
    snapshots: list[tuple[int, FloatArray]],
    dt: float,
    dx: float,
    path: str | Path,
    fps: int = 20,
    db_floor: float = -60.0,
) -> Path:
    """Animated GIF of |p| in dB (relative to the global peak) over time."""
    if not snapshots:
        raise ValueError("no snapshots recorded (set snapshot_every > 0)")
    peak = max(float(np.abs(s).max()) for _, s in snapshots) or 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    extent = (
        0.0,
        snapshots[0][1].shape[1] * dx * 1e3,
        0.0,
        snapshots[0][1].shape[0] * dx * 1e3,
    )

    def to_db(field: FloatArray) -> FloatArray:
        return 20.0 * np.log10(np.maximum(np.abs(field) / peak, 10 ** (db_floor / 20.0)))

    im = ax.imshow(
        to_db(snapshots[0][1]),
        origin="lower",
        extent=extent,
        vmin=db_floor,
        vmax=0.0,
        cmap="inferno",
        aspect="equal",
    )
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("|p| dB re peak")
    title = ax.set_title("")

    def update(frame: int) -> list[Artist]:
        it, field = snapshots[frame]
        im.set_data(to_db(field))
        title.set_text(f"t = {it * dt * 1e3:.3f} ms")
        return [im, title]

    anim = animation.FuncAnimation(fig, update, frames=len(snapshots), blit=False)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out


def save_slice_frame(
    snapshot: FloatArray,
    dx: float,
    path: str | Path,
    title: str = "",
) -> Path:
    """Single pressure-slice PNG (linear scale, symmetric colormap)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = float(np.abs(snapshot).max()) or 1.0
    extent = (0.0, snapshot.shape[1] * dx * 1e3, 0.0, snapshot.shape[0] * dx * 1e3)
    im = ax.imshow(
        snapshot,
        origin="lower",
        extent=extent,
        vmin=-vmax,
        vmax=vmax,
        cmap="RdBu_r",
        aspect="equal",
    )
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="p")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out
