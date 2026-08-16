"""Parametric headphone parts as occupancy grids: baffles, cups, perforated plates.

All builders return a bool tensor of ``grid.shape`` (True = rigid solid). Parts
are computed only within their bounding sub-box for speed. Hole patterns are
defined in the plate's local (a, b) plane coordinates; True in a pattern mask
means OPEN (air).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from headphone_sims.grid import Grid

Vec3 = tuple[float, float, float]


def _local_frame(normal: Vec3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = torch.tensor(normal, dtype=torch.float64)
    n = n / torch.linalg.vector_norm(n)
    helper = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    if float(n[2].abs()) > 0.9:
        helper = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    u = torch.linalg.cross(n, helper)
    u = u / torch.linalg.vector_norm(u)
    w = torch.linalg.cross(n, u)
    return n, u, w


def _subbox(
    grid: Grid, center: Vec3, half_extent: float
) -> tuple[tuple[slice, slice, slice], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slices into the full grid plus cell-center coordinates of the sub-box."""
    slices = []
    coords = []
    for axis in range(3):
        lo = max(0, int((center[axis] - half_extent) / grid.dx) - 1)
        hi = min(grid.shape[axis], int((center[axis] + half_extent) / grid.dx) + 2)
        if lo >= hi:
            raise ValueError(f"part at {center} lies outside the grid along axis {axis}")
        slices.append(slice(lo, hi))
        coords.append((torch.arange(lo, hi, dtype=torch.float64) + 0.5) * grid.dx)
    gx, gy, gz = torch.meshgrid(coords[0], coords[1], coords[2], indexing="ij")
    return (slices[0], slices[1], slices[2]), gx, gy, gz


class HolePattern:
    """Base: subclasses mark where the plate is OPEN."""

    def open_mask(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def open_mask_3d(
        self, a: torch.Tensor, b: torch.Tensor, axial_frac: torch.Tensor
    ) -> torch.Tensor:
        """3D-aware openness; ``axial_frac`` in [-0.5, 0.5], +0.5 = exit (ear)
        face. Default: extrude the 2D pattern through the thickness."""
        return self.open_mask(a, b)


@dataclass(frozen=True)
class HexHoles(HolePattern):
    """Hexagonal lattice of circular holes (typical planar-magnetic grille)."""

    hole_radius: float
    pitch: float

    def open_mask(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        row_h = self.pitch * math.sqrt(3.0) / 2.0
        m0 = torch.round(b / row_h)
        best = torch.full_like(a, float("inf"))
        for dm in (-1.0, 0.0, 1.0):
            m = m0 + dm
            b_row = m * row_h
            offset = torch.where(m.to(torch.int64) % 2 == 0, 0.0, self.pitch / 2.0)
            a_near = torch.round((a - offset) / self.pitch) * self.pitch + offset
            d2 = (a - a_near) ** 2 + (b - b_row) ** 2
            best = torch.minimum(best, d2)
        return best <= self.hole_radius**2


@dataclass(frozen=True)
class Slots(HolePattern):
    """Parallel open slots of ``width`` spaced ``pitch`` apart along the a-axis."""

    width: float
    pitch: float

    def open_mask(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        frac = torch.remainder(a + self.pitch / 2.0, self.pitch) - self.pitch / 2.0
        return frac.abs() <= self.width / 2.0


@dataclass(frozen=True)
class ChamferedSlots(HolePattern):
    """Slots between magnet bars whose exit side is chamfered (Fazor-style).

    The gap is ``width`` for the straight throat (diaphragm side) and flares
    linearly to ``width + 2*chamfer_depth`` at the exit face over the last
    ``chamfer_fraction`` of the plate thickness — the bar cross-section
    becomes a trapezoid, shortening the acoustic neck and easing the exit
    discontinuity.
    """

    width: float
    pitch: float
    chamfer_depth: float
    chamfer_fraction: float = 0.66

    def open_mask(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # 2D fallback: the throat profile.
        frac = torch.remainder(a + self.pitch / 2.0, self.pitch) - self.pitch / 2.0
        return frac.abs() <= self.width / 2.0

    def open_mask_3d(
        self, a: torch.Tensor, b: torch.Tensor, axial_frac: torch.Tensor
    ) -> torch.Tensor:
        t_norm = (axial_frac + 0.5).clamp(0.0, 1.0)  # 0 = diaphragm side, 1 = exit
        start = 1.0 - self.chamfer_fraction
        flare = ((t_norm - start) / self.chamfer_fraction).clamp(min=0.0)
        half_gap = self.width / 2.0 + self.chamfer_depth * flare
        frac = torch.remainder(a + self.pitch / 2.0, self.pitch) - self.pitch / 2.0
        return frac.abs() <= half_gap


@dataclass(frozen=True)
class RingSlits(HolePattern):
    """Concentric open annular slits given as (r_inner, r_outer) pairs."""

    rings: tuple[tuple[float, float], ...]

    def open_mask(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt(a**2 + b**2)
        mask = torch.zeros_like(a, dtype=torch.bool)
        for r_in, r_out in self.rings:
            mask |= (r >= r_in) & (r <= r_out)
        return mask


@dataclass(frozen=True)
class FibonacciSpirals(HolePattern):
    """Crossing logarithmic-spiral ribs in Fibonacci counts (phyllotaxis
    parastichy) — the MDR-Z1R-style grille: ``m_cw`` clockwise and ``m_ccw``
    counterclockwise spirals of width ``rib_width`` leave open, outward-growing
    cells. Non-periodic spacing (no single lattice pitch), high open ratio.

    ``winding`` is the log-spiral slope b in theta = +/- b*ln(r); larger =
    more tightly wound. Inside ``hub_radius`` the plate is open (the spiral
    winding diverges toward the center).
    """

    rib_width: float = 1.0e-3
    m_cw: int = 8
    m_ccw: int = 13
    winding: float = 0.9
    hub_radius: float = 2.5e-3

    def open_mask(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt(a**2 + b**2).clamp(min=1e-9)
        theta = torch.atan2(b, a)
        log_r = torch.log(r / self.hub_radius)
        solid = torch.zeros_like(a, dtype=torch.bool)
        slant = math.sqrt(1.0 + self.winding**2)
        for m, sign in ((self.m_cw, 1.0), (self.m_ccw, -1.0)):
            period = 2.0 * math.pi / m
            phase = torch.remainder(theta - sign * self.winding * log_r, period)
            d_ang = torch.minimum(phase, period - phase)
            perp = d_ang * r / slant  # perpendicular distance to the nearest rib
            solid |= perp <= self.rib_width / 2.0
        solid &= r >= self.hub_radius
        return ~solid


def plate(
    grid: Grid,
    center: Vec3,
    normal: Vec3,
    radius: float,
    thickness: float,
    pattern: HolePattern | None = None,
    pattern_angle_deg: float = 0.0,
) -> tuple[torch.Tensor, float]:
    """Circular plate (optionally perforated). Returns (occupancy, open porosity).

    Porosity is the open-area fraction of the disc region, measured on the
    voxelized mid-slab — i.e., what the FDTD grid actually sees.
    """
    n, u, w = _local_frame(normal)
    (sx, sy, sz), gx, gy, gz = _subbox(grid, center, radius + thickness + 2 * grid.dx)
    dxv = gx - center[0]
    dyv = gy - center[1]
    dzv = gz - center[2]
    axial = dxv * n[0] + dyv * n[1] + dzv * n[2]
    a = dxv * u[0] + dyv * u[1] + dzv * u[2]
    b = dxv * w[0] + dyv * w[1] + dzv * w[2]
    radial2 = a**2 + b**2
    # Half-open slab interval with a sub-micron bias: when the plate mid-plane
    # falls exactly on a cell boundary, float rounding could otherwise select
    # zero layers (empty plate) or two. This picks exactly one, deterministically.
    bias = 1e-3 * grid.dx
    in_disc = (
        (axial > -thickness / 2.0 + bias)
        & (axial <= thickness / 2.0 + bias)
        & (radial2 <= radius**2)
    )
    solid_local = in_disc.clone()
    if pattern is not None:
        ang = math.radians(pattern_angle_deg)
        pa = a * math.cos(ang) + b * math.sin(ang)
        pb = -a * math.sin(ang) + b * math.cos(ang)
        solid_local &= ~pattern.open_mask_3d(pa, pb, axial / thickness)
    n_disc = int(in_disc.sum())
    porosity = 1.0 - int(solid_local.sum()) / n_disc if n_disc else 0.0

    occ = torch.zeros(grid.shape, dtype=torch.bool)
    occ[sx, sy, sz] = solid_local
    return occ, porosity


def rect_plate(
    grid: Grid,
    center: Vec3,
    normal: Vec3,
    width: float,
    height: float,
    thickness: float,
    pattern: HolePattern | None = None,
    pattern_angle_deg: float = 0.0,
) -> tuple[torch.Tensor, float]:
    """Rectangular plate (optionally perforated). Returns (occupancy, porosity).

    The height axis is the projection of scene +y onto the plate plane.
    ``pattern_angle_deg`` rotates the hole pattern in-plane (e.g. 90 to turn
    horizontal slots into vertical magnet-bar gaps).
    """
    n, u, w = _rect_frame(normal)
    half = max(width, height) / 2.0 + thickness + 2 * grid.dx
    (sx, sy, sz), gx, gy, gz = _subbox(grid, center, half)
    dxv, dyv, dzv = gx - center[0], gy - center[1], gz - center[2]
    axial = dxv * n[0] + dyv * n[1] + dzv * n[2]
    a = dxv * u[0] + dyv * u[1] + dzv * u[2]
    b = dxv * w[0] + dyv * w[1] + dzv * w[2]
    bias = 1e-3 * grid.dx
    in_rect = (
        (axial > -thickness / 2.0 + bias)
        & (axial <= thickness / 2.0 + bias)
        & (a.abs() <= height / 2.0)
        & (b.abs() <= width / 2.0)
    )
    solid_local = in_rect.clone()
    if pattern is not None:
        ang = math.radians(pattern_angle_deg)
        pa = a * math.cos(ang) + b * math.sin(ang)
        pb = -a * math.sin(ang) + b * math.cos(ang)
        solid_local &= ~pattern.open_mask_3d(pa, pb, axial / thickness)
    n_rect = int(in_rect.sum())
    porosity = 1.0 - int(solid_local.sum()) / n_rect if n_rect else 0.0
    occ = torch.zeros(grid.shape, dtype=torch.bool)
    occ[sx, sy, sz] = solid_local
    return occ, porosity


def rect_collar(
    grid: Grid,
    center: Vec3,
    normal: Vec3,
    width: float,
    height: float,
    length: float,
    thickness: float,
) -> torch.Tensor:
    """Rectangular side wall sealing a standoff cavity (rect_plate's rim)."""
    n, u, w = _rect_frame(normal)
    half = max(width, height) / 2.0 + thickness + length + 2 * grid.dx
    (sx, sy, sz), gx, gy, gz = _subbox(grid, center, half)
    dxv, dyv, dzv = gx - center[0], gy - center[1], gz - center[2]
    axial = dxv * n[0] + dyv * n[1] + dzv * n[2]
    a = (dxv * u[0] + dyv * u[1] + dzv * u[2]).abs()
    b = (dxv * w[0] + dyv * w[1] + dzv * w[2]).abs()
    inside = (a <= height / 2.0) & (b <= width / 2.0)
    outside_ring = (a <= height / 2.0 + thickness) & (b <= width / 2.0 + thickness)
    ring = (axial >= 0.0) & (axial <= length) & outside_ring & ~inside
    occ = torch.zeros(grid.shape, dtype=torch.bool)
    occ[sx, sy, sz] = ring
    return occ


def _rect_frame(normal: Vec3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frame whose u axis is the projection of +y (scene vertical) in-plane."""
    n = torch.tensor(normal, dtype=torch.float64)
    n = n / torch.linalg.vector_norm(n)
    helper = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    if float(n[1].abs()) > 0.9:
        helper = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    u = helper - n * torch.dot(helper, n)
    u = u / torch.linalg.vector_norm(u)
    w = torch.linalg.cross(n, u)
    return n, u, w


def annular_collar(
    grid: Grid,
    center: Vec3,
    normal: Vec3,
    radius: float,
    length: float,
    thickness: float,
) -> torch.Tensor:
    """Cylindrical side wall sealing a standoff cavity (e.g. driver-to-grille).

    Solid ring with inner radius ``radius`` extending ``length`` along
    ``normal`` from the plane of ``center``.
    """
    n, u, w = _local_frame(normal)
    (sx, sy, sz), gx, gy, gz = _subbox(grid, center, radius + thickness + length + 2 * grid.dx)
    dxv = gx - center[0]
    dyv = gy - center[1]
    dzv = gz - center[2]
    axial = dxv * n[0] + dyv * n[1] + dzv * n[2]
    a = dxv * u[0] + dyv * u[1] + dzv * u[2]
    b = dxv * w[0] + dyv * w[1] + dzv * w[2]
    r = torch.sqrt(a**2 + b**2)
    ring = (axial >= 0.0) & (axial <= length) & (r >= radius) & (r <= radius + thickness)
    occ = torch.zeros(grid.shape, dtype=torch.bool)
    occ[sx, sy, sz] = ring
    return occ


def cup_shell(
    grid: Grid,
    center: Vec3,
    axis: Vec3,
    inner_radius: float,
    depth: float,
    thickness: float,
) -> torch.Tensor:
    """Open cylindrical earcup: side wall plus closed back.

    ``center`` is the middle of the open rim plane; the cup extends ``depth``
    along ``axis`` (pointing from rim toward the back plate).
    """
    n, u, w = _local_frame(axis)
    half = inner_radius + thickness + depth + 2 * grid.dx
    (sx, sy, sz), gx, gy, gz = _subbox(grid, center, half)
    dxv = gx - center[0]
    dyv = gy - center[1]
    dzv = gz - center[2]
    axial = dxv * n[0] + dyv * n[1] + dzv * n[2]
    a = dxv * u[0] + dyv * u[1] + dzv * u[2]
    b = dxv * w[0] + dyv * w[1] + dzv * w[2]
    r = torch.sqrt(a**2 + b**2)
    in_length = (axial >= 0.0) & (axial <= depth + thickness)
    wall = in_length & (r >= inner_radius) & (r <= inner_radius + thickness)
    back = (axial >= depth) & (axial <= depth + thickness) & (r <= inner_radius + thickness)
    occ = torch.zeros(grid.shape, dtype=torch.bool)
    occ[sx, sy, sz] = wall | back
    return occ
