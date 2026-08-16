"""Scene geometry views: what is actually being simulated.

Renders orthogonal mid-plane slices of the solid occupancy grid with the
driver disc and receiver probes overlaid, so a run report always shows the
voxelized geometry (baffle, filter, pinna, cup) as the FDTD grid sees it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import torch  # noqa: E402

FloatArray = npt.NDArray[np.float64]


def save_scene_views(
    solid: torch.Tensor,
    dx: float,
    driver_center: tuple[float, float, float],
    probe_positions: FloatArray,
    path: str | Path,
    slab_half_width: float = 1.5e-3,
) -> Path:
    """Three orthogonal slices through the driver axis with probes overlaid.

    Probes within ``slab_half_width`` of each slice plane are drawn on it.
    """
    occ = solid.cpu().numpy()
    center_idx = [int(round(c / dx)) for c in driver_center]
    shape = occ.shape

    k_front = min(int(round((driver_center[2] + 2e-3) / dx)), shape[2] - 1)
    views = [
        ("y-z (side)", occ[center_idx[0], :, :], (1, 2)),
        ("x-z (top)", occ[:, center_idx[1], :], (0, 2)),
        ("x-y (front, driver plane +2mm)", occ[:, :, k_front], (0, 1)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (title, sl, (a, b)) in zip(axes, views, strict=True):
        extent = (0.0, sl.shape[1] * dx * 1e3, 0.0, sl.shape[0] * dx * 1e3)
        ax.imshow(
            sl,
            origin="lower",
            extent=extent,
            cmap="gray_r",
            aspect="equal",
            interpolation="nearest",
        )
        plane_axis = 3 - a - b
        near = np.abs(probe_positions[:, plane_axis] - driver_center[plane_axis]) <= (
            slab_half_width if plane_axis != 2 else 1e9
        )
        if plane_axis == 2:
            near = np.ones(len(probe_positions), dtype=bool)
        ax.scatter(
            probe_positions[near, b] * 1e3,
            probe_positions[near, a] * 1e3,
            s=6,
            c="tab:orange",
            label="probes",
        )
        ax.plot(
            driver_center[b] * 1e3,
            driver_center[a] * 1e3,
            "b*",
            markersize=12,
            label="driver center",
        )
        ax.set_title(title)
        ax.set_xlabel("xyz"[b] + " (mm)")
        ax.set_ylabel("xyz"[a] + " (mm)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Voxelized scene geometry (gray = rigid solid)")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out
