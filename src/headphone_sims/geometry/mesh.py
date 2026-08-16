"""Mesh loading, repair, and voxelization onto the FDTD grid.

Grid convention: cell (i, j, k) has its center at ((i+0.5)dx, (j+0.5)dx, (k+0.5)dx)
in domain coordinates (meters), matching :class:`headphone_sims.grid.Grid`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh

from headphone_sims.grid import Grid


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load an STL/PLY/OBJ mesh and apply light repairs."""
    loaded = trimesh.load(str(path), force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"{path} did not load as a single triangle mesh")
    loaded.merge_vertices()
    loaded.update_faces(loaded.nondegenerate_faces())
    loaded.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(loaded)
    return loaded


def transform_mesh(
    mesh: trimesh.Trimesh,
    scale: float = 1.0,
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> trimesh.Trimesh:
    """Scaled/rotated/translated copy (rotations are extrinsic XYZ, degrees)."""
    out = mesh.copy()
    if scale != 1.0:
        out.apply_scale(scale)
    for axis_vec, angle in zip(
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), rotation_deg, strict=True
    ):
        if angle:
            out.apply_transform(
                trimesh.transformations.rotation_matrix(np.deg2rad(angle), axis_vec)
            )
    out.apply_translation(list(translation))
    return out


def clip_to_box(
    mesh: trimesh.Trimesh,
    box_min: tuple[float, float, float],
    box_max: tuple[float, float, float],
) -> trimesh.Trimesh:
    """Intersection of the mesh with an axis-aligned box (capped, stays solid)."""
    out = mesh.copy()
    for axis in range(3):
        for sign, bound in ((1.0, box_min[axis]), (-1.0, box_max[axis])):
            normal = np.zeros(3)
            normal[axis] = sign
            origin = np.zeros(3)
            origin[axis] = bound
            out = out.slice_plane(origin, normal, cap=True)
            if out is None or len(out.faces) == 0:
                raise ValueError("mesh does not intersect the clipping box")
    return out


def voxelize(mesh: trimesh.Trimesh, grid: Grid, fill: bool = True) -> torch.Tensor:
    """Occupancy grid (bool, ``grid.shape``) of the mesh in domain coordinates.

    The mesh is clipped to the domain, surface-voxelized at pitch ``dx``, and
    (optionally) flood-filled to a solid interior. Robust to small mesh defects
    because voxelization is surface-based rather than winding-number-based.
    """
    dx = grid.dx
    ext = grid.extent
    clipped = clip_to_box(mesh, (0.0, 0.0, 0.0), ext)
    vg = clipped.voxelized(pitch=dx)
    if fill:
        vg = vg.fill()
    points = np.asarray(vg.points)  # world centers of filled voxels
    idx = np.floor(points / dx).astype(np.int64)
    keep = np.all((idx >= 0) & (idx < np.array(grid.shape)), axis=1)
    idx = idx[keep]
    occ = torch.zeros(grid.shape, dtype=torch.bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ


def surface_probes(
    mesh: trimesh.Trimesh,
    n_points: int,
    offset: float = 1.5e-3,
    direction: tuple[float, float, float] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Probe positions offset from the mesh surface along outward normals.

    If ``direction`` is given, only faces whose normal has a positive component
    along it are sampled (e.g., the driver-facing side of a pinna).
    """
    m = mesh
    if direction is not None:
        d = np.asarray(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        keep = (m.face_normals @ d) > 0.0
        if not keep.any():
            raise ValueError("no faces oriented along the requested direction")
        m = m.copy()
        m.update_faces(keep)
        m.remove_unreferenced_vertices()
    points, face_idx = trimesh.sample.sample_surface_even(m, n_points, seed=seed)
    normals = m.face_normals[face_idx]
    return np.asarray(points + offset * normals, dtype=np.float64)
