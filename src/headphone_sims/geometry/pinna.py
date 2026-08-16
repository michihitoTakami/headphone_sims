"""Pinna geometry: ear extraction from head meshes, placement, parametric fallback."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
import trimesh

from headphone_sims.geometry.mesh import clip_to_box, transform_mesh
from headphone_sims.grid import Grid

Vec3 = tuple[float, float, float]


def find_ear_tip(
    mesh: trimesh.Trimesh, side: str = "left", lateral_axis: int = 0
) -> npt.NDArray[np.float64]:
    """The laterally outermost vertex — the most protruding point of the pinna.

    On real head scans this lands on the helix rim, NOT at the ear canal;
    use it as a depth anchor only, never as an aiming point.
    """
    coords = mesh.vertices[:, lateral_axis]
    idx = int(np.argmin(coords)) if side == "left" else int(np.argmax(coords))
    return np.asarray(mesh.vertices[idx], dtype=np.float64)


def find_ear_canal_entrance(
    mesh: trimesh.Trimesh, side: str = "left", lateral_axis: int = 0
) -> npt.NDArray[np.float64]:
    """Ear-canal entrance: the surface point closest to the interaural axis.

    Relies on the HRTF-database convention (verified for HUTUBS) that the
    mesh is aligned with the interaural axis — the ``lateral_axis`` line
    through the origin — passing through both ear-canal entrances. For meshes
    without this alignment, pass an explicit canal position instead
    (``PinnaSpec.canal_hint``).
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    lateral = v[:, lateral_axis]
    on_side = lateral < 0.0 if side == "left" else lateral > 0.0
    if not on_side.any():
        raise ValueError(f"mesh has no vertices on the {side} side of the lateral axis")
    others = [a for a in range(3) if a != lateral_axis]
    d2 = v[on_side, others[0]] ** 2 + v[on_side, others[1]] ** 2
    out: npt.NDArray[np.float64] = v[on_side][int(np.argmin(d2))].copy()
    return out


def extract_ear_region(
    mesh: trimesh.Trimesh,
    center: npt.NDArray[np.float64],
    side: str = "left",
    box_size: float = 90e-3,
    lateral_axis: int = 0,
    depth_fraction: float = 0.45,
) -> trimesh.Trimesh:
    """Crop a head mesh to a box around ``center`` (the ear-canal entrance).

    The box spans ``box_size`` in the two tangential axes; along the lateral
    axis it runs from just outside the ear tip to ``depth_fraction * box_size``
    inward, keeping the pinna plus a patch of head surface behind it.
    """
    tip = find_ear_tip(mesh, side=side, lateral_axis=lateral_axis)
    lo = center - box_size / 2.0
    hi = center + box_size / 2.0
    depth = depth_fraction * box_size
    if side == "left":
        lo[lateral_axis] = tip[lateral_axis] - 2e-3
        hi[lateral_axis] = tip[lateral_axis] + depth
    else:
        lo[lateral_axis] = tip[lateral_axis] - depth
        hi[lateral_axis] = tip[lateral_axis] + 2e-3
    return clip_to_box(mesh, (lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2]))


def orient_pinna_to_scene(
    ear_mesh: trimesh.Trimesh,
    canal: npt.NDArray[np.float64],
    side: str = "left",
    lateral_axis: int = 0,
    extra_rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> trimesh.Trimesh:
    """Rotate an extracted ear into scene orientation, canal at the origin.

    Scene convention: the driver radiates along +z toward the pinna, so the
    pinna's outward (lateral) direction maps to -z. The ear-canal entrance
    ends up at the origin; the caller applies the final translation (aiming
    the driver axis at the canal).
    """
    out = ear_mesh.copy()
    out.apply_translation((-canal).tolist())
    # Rotate lateral axis onto -z (left ear points -lateral; right ear +lateral).
    outward = np.zeros(3)
    outward[lateral_axis] = -1.0 if side == "left" else 1.0
    target_dir = np.array([0.0, 0.0, -1.0])
    rot = trimesh.geometry.align_vectors(outward, target_dir)
    out.apply_transform(rot)
    return transform_mesh(out, rotation_deg=extra_rotation_deg)


def select_pinna_probes(
    mesh: trimesh.Trimesh,
    canal: npt.NDArray[np.float64],
    n_probes: int,
    offset: float = 1.5e-3,
    protrusion_threshold: float = 2.5e-3,
    concha_radius: float = 15e-3,
    pitch: float = 2e-3,
    max_normal_z: float = 0.3,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Probe positions on the pinna proper of a scene-oriented ear mesh.

    Works in scene coordinates (driver side = -z, canal near the origin):

    1. Rasterize the driver-facing surface depth z_min(x, y) on a ``pitch``
       grid and robustly fit a quadratic head-skin base surface (iteratively
       excluding protruding cells).
    2. Pinna mask = cells protruding more than ``protrusion_threshold`` in
       front of the base, restricted to the connected component nearest the
       canal (rejects fit artifacts at the crop border), dilated by one cell.
    3. Add the concha bowl (within ``concha_radius`` of the canal in x-y),
       which is recessed and therefore not caught by protrusion.
    4. Sample the mesh surface evenly, keep samples inside the mask whose
       face normal has z-component below ``max_normal_z`` (drops the back of
       the pinna and head-facing surfaces), offset along the face normal.
    """
    from scipy import ndimage

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    x0, y0 = float(bounds[0, 0]), float(bounds[0, 1])
    # Rasterize from dense surface samples, not vertices — cropped meshes can
    # have large sparsely-triangulated faces (e.g. slice caps).
    n_raster = max(20_000, int(4.0 * mesh.area / pitch**2))
    surf, _ = trimesh.sample.sample_surface(mesh, n_raster, seed=seed)
    surf = np.asarray(surf, dtype=np.float64)
    gx = np.floor((surf[:, 0] - x0) / pitch).astype(int)
    gy = np.floor((surf[:, 1] - y0) / pitch).astype(int)
    nx = int(np.ceil((bounds[1, 0] - x0) / pitch)) + 1
    ny = int(np.ceil((bounds[1, 1] - y0) / pitch)) + 1
    gx = np.clip(gx, 0, nx - 1)
    gy = np.clip(gy, 0, ny - 1)
    zmap = np.full((nx, ny), np.inf)
    np.minimum.at(zmap, (gx, gy), surf[:, 2])
    valid = np.isfinite(zmap)

    xs, ys = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float), indexing="ij")
    protruding = np.zeros_like(valid)
    base = np.zeros_like(zmap)
    for _ in range(3):
        fit = valid & ~protruding
        a_mat = np.stack(
            [np.ones(int(fit.sum())), xs[fit], ys[fit], xs[fit] ** 2, ys[fit] ** 2, (xs * ys)[fit]],
            axis=1,
        )
        coef, *_ = np.linalg.lstsq(a_mat, zmap[fit], rcond=None)
        base = (
            coef[0]
            + coef[1] * xs
            + coef[2] * ys
            + coef[3] * xs**2
            + coef[4] * ys**2
            + coef[5] * xs * ys
        )
        protruding = valid & (base - zmap > protrusion_threshold)

    canal_cell = np.array([(canal[0] - x0) / pitch, (canal[1] - y0) / pitch])
    labels, n_labels = ndimage.label(protruding)
    if n_labels == 0:
        raise ValueError("no protruding region found; is this a pinna mesh?")
    # Keep every component near the canal — the pinna can split into several
    # blobs (e.g. helix separated from the main body where the fold dips
    # below threshold) while crop-border fit artifacts sit far away.
    reach_cells = (concha_radius + 10e-3) / pitch
    dists = []
    for lab in range(1, n_labels + 1):
        cells = np.argwhere(labels == lab)
        dists.append(float(np.min(np.linalg.norm(cells - canal_cell, axis=1))))
    keep_labels = [
        lab for lab, d in zip(range(1, n_labels + 1), dists, strict=True) if d <= reach_cells
    ]
    if not keep_labels:
        keep_labels = [int(np.argmin(dists)) + 1]
    mask = ndimage.binary_dilation(np.isin(labels, keep_labels), iterations=1)

    cell_x = (xs + 0.5) * pitch + x0
    cell_y = (ys + 0.5) * pitch + y0
    concha = (cell_x - canal[0]) ** 2 + (cell_y - canal[1]) ** 2 <= concha_radius**2
    mask |= valid & concha

    points, face_idx = trimesh.sample.sample_surface_even(mesh, n_probes * 12, seed=seed)
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float64)
    front = normals[:, 2] < max_normal_z
    px = np.clip(np.floor((points[:, 0] - x0) / pitch).astype(int), 0, nx - 1)
    py = np.clip(np.floor((points[:, 1] - y0) / pitch).astype(int), 0, ny - 1)
    keep = front & mask[px, py]
    if not keep.any():
        raise ValueError("no surface samples on the detected pinna region")
    probes: npt.NDArray[np.float64] = points[keep] + offset * normals[keep]
    return probes[:n_probes]


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
