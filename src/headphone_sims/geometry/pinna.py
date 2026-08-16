"""Pinna geometry: ear extraction from head meshes, placement, parametric fallback."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
import trimesh

from headphone_sims.geometry.mesh import clip_to_box, transform_mesh
from headphone_sims.grid import Grid

Vec3 = tuple[float, float, float]


def find_ear_center(
    mesh: trimesh.Trimesh, side: str = "left", lateral_axis: int = 0
) -> npt.NDArray[np.float64]:
    """Heuristic ear locator: the extreme vertex along the lateral axis.

    Ears are usually the laterally outermost points of a head scan. Verify
    visually for a new dataset (axis conventions differ between databases).
    """
    coords = mesh.vertices[:, lateral_axis]
    idx = int(np.argmin(coords)) if side == "left" else int(np.argmax(coords))
    return np.asarray(mesh.vertices[idx], dtype=np.float64)


def extract_ear_region(
    mesh: trimesh.Trimesh,
    side: str = "left",
    box_size: float = 90e-3,
    lateral_axis: int = 0,
    depth_fraction: float = 0.45,
) -> tuple[trimesh.Trimesh, npt.NDArray[np.float64]]:
    """Crop a head mesh to a box around the ear.

    The box spans ``box_size`` in the two tangential axes and
    ``depth_fraction * box_size`` inward from the ear tip along the lateral
    axis, keeping the pinna plus a patch of head surface behind it.
    Returns (cropped mesh, ear center in original coordinates).
    """
    ear = find_ear_center(mesh, side=side, lateral_axis=lateral_axis)
    lo = ear - box_size / 2.0
    hi = ear + box_size / 2.0
    depth = depth_fraction * box_size
    if side == "left":
        lo[lateral_axis] = ear[lateral_axis] - 2e-3
        hi[lateral_axis] = ear[lateral_axis] + depth
    else:
        lo[lateral_axis] = ear[lateral_axis] - depth
        hi[lateral_axis] = ear[lateral_axis] + 2e-3
    cropped = clip_to_box(mesh, (lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2]))
    return cropped, ear


def orient_pinna_to_scene(
    ear_mesh: trimesh.Trimesh,
    ear_center: npt.NDArray[np.float64],
    target_center: Vec3,
    side: str = "left",
    lateral_axis: int = 0,
    extra_rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
) -> trimesh.Trimesh:
    """Move an extracted ear so the pinna faces the scene's -z (driver) direction.

    Scene convention: the driver radiates along +z toward the pinna, so the
    pinna's outward (lateral) direction must map to -z, with the ear tip at
    ``target_center``.
    """
    out = ear_mesh.copy()
    out.apply_translation((-ear_center).tolist())
    if scale != 1.0:
        out.apply_scale(scale)
    # Rotate lateral axis onto -z (left ear points -lateral; right ear +lateral).
    outward = np.zeros(3)
    outward[lateral_axis] = -1.0 if side == "left" else 1.0
    target_dir = np.array([0.0, 0.0, -1.0])
    rot = trimesh.geometry.align_vectors(outward, target_dir)
    out.apply_transform(rot)
    out = transform_mesh(out, rotation_deg=extra_rotation_deg, translation=target_center)
    return out


def parametric_pinna(
    grid: Grid,
    center: Vec3,
    height: float = 60e-3,
    width: float = 35e-3,
    protrusion: float = 18e-3,
    shell_thickness: float = 3e-3,
    concha_radius: float = 10e-3,
) -> torch.Tensor:
    """Crude parametric pinna: half-ellipsoid shell with a concha opening.

    Scene convention: the driver sits at low z and radiates toward +z, so the
    pinna faces -z. ``center`` is the concha center on the head-surface plane;
    the shell bulges toward the driver (-z) by ``protrusion``.
    """
    ax = width / 2.0
    ay = height / 2.0
    az = protrusion
    half = max(ax, ay, az) + shell_thickness + 2 * grid.dx
    lo = [max(0, int((center[i] - half) / grid.dx) - 1) for i in range(3)]
    hi = [min(grid.shape[i], int((center[i] + half) / grid.dx) + 2) for i in range(3)]
    coords = [
        (torch.arange(lo[i], hi[i], dtype=torch.float64) + 0.5) * grid.dx - center[i]
        for i in range(3)
    ]
    gx, gy, gz = torch.meshgrid(coords[0], coords[1], coords[2], indexing="ij")

    def ellipsoid(sx: float, sy: float, sz: float) -> torch.Tensor:
        return (gx / sx) ** 2 + (gy / sy) ** 2 + (gz / sz) ** 2 <= 1.0

    outer = ellipsoid(ax, ay, az)
    inner = ellipsoid(
        max(ax - shell_thickness, 1e-4),
        max(ay - shell_thickness, 1e-4),
        max(az - shell_thickness, 1e-4),
    )
    shell = outer & ~inner & (gz <= 0.0)  # bulge toward the driver side (-z)
    # Concha opening: remove shell within a cylinder near the center.
    concha = (gx**2 + gy**2) <= concha_radius**2
    shell &= ~concha

    occ = torch.zeros(grid.shape, dtype=torch.bool)
    occ[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = shell
    return occ
