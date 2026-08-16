"""Excitation waveforms and source geometries.

Sources are "soft" (additive): they add to the field rather than overwrite it,
so scattered waves pass through the source region. Each source is *baked* onto
a specific grid/device into flat index/weight tensors for fast per-step
injection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from headphone_sims.grid import Grid

Vec3 = tuple[float, float, float]


def gaussian_modulated_sine(
    dt: float,
    n_steps: int,
    center_frequency: float = 10_000.0,
    bandwidth_frequency: float = 9_900.0,
    amplitude: float = 1.0,
) -> torch.Tensor:
    """Gaussian-windowed sine burst covering roughly fc +/- bandwidth.

    The Gaussian envelope std is chosen so the spectrum's -6 dB half-width is
    ``bandwidth_frequency``; the burst is centered late enough (4 sigma) that
    the leading truncation is negligible.
    """
    sigma_t = math.sqrt(math.log(2.0)) / (math.pi * bandwidth_frequency)
    t0 = 4.0 * sigma_t
    t = torch.arange(n_steps, dtype=torch.float64) * dt
    env = torch.exp(-0.5 * ((t - t0) / sigma_t) ** 2)
    return (amplitude * env * torch.sin(2.0 * math.pi * center_frequency * (t - t0))).to(
        torch.float32
    )


def ricker(dt: float, n_steps: int, peak_frequency: float, amplitude: float = 1.0) -> torch.Tensor:
    """Ricker wavelet (second derivative of a Gaussian): zero-DC broadband pulse."""
    t0 = 1.5 / peak_frequency
    t = torch.arange(n_steps, dtype=torch.float64) * dt - t0
    a = (math.pi * peak_frequency * t) ** 2
    return (amplitude * (1.0 - 2.0 * a) * torch.exp(-a)).to(torch.float32)


@dataclass
class BakedSource:
    """Grid-resident injection lists for one source.

    ``hard`` velocity sources overwrite the face values (a velocity boundary
    condition, e.g. a piston on a rigid wall); soft sources add to them.
    Velocity injection must run between the kernel's velocity and pressure
    half-steps so hard sources survive the solid face masks.
    """

    waveform: torch.Tensor  # (n_steps,) on CPU
    p_idx: torch.Tensor | None = None
    p_weight: torch.Tensor | None = None
    v_idx: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    v_weight: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    hard: bool = False
    _wave_dev: torch.Tensor = field(init=False)

    def to(self, device: torch.device) -> BakedSource:
        self._wave_dev = self.waveform.to(device)
        if self.p_idx is not None and self.p_weight is not None:
            self.p_idx = self.p_idx.to(device)
            self.p_weight = self.p_weight.to(device)
        if self.v_idx is not None and self.v_weight is not None:
            self.v_idx = tuple(i.to(device) for i in self.v_idx)  # type: ignore[assignment]
            self.v_weight = tuple(w.to(device) for w in self.v_weight)  # type: ignore[assignment]
        return self

    def inject_velocity(self, v: tuple[torch.Tensor, ...], it: int) -> None:
        if self.v_idx is None or self.v_weight is None:
            return
        if it >= int(self._wave_dev.shape[0]):
            if not self.hard:
                return
            amp = torch.zeros((), device=self._wave_dev.device, dtype=self._wave_dev.dtype)
        else:
            amp = self._wave_dev[it]
        for comp, idx, w in zip(v, self.v_idx, self.v_weight, strict=True):
            if not idx.numel():
                continue
            if self.hard:
                comp.view(-1).index_copy_(0, idx, w * amp)
            else:
                comp.view(-1).index_add_(0, idx, w * amp)

    def inject_pressure(self, p: torch.Tensor, it: int) -> None:
        if self.p_idx is None or self.p_weight is None:
            return
        if it >= int(self._wave_dev.shape[0]):
            return
        amp = self._wave_dev[it]
        p.view(-1).index_add_(0, self.p_idx, self.p_weight * amp)


@dataclass(frozen=True)
class PointSource:
    """Monopole: adds the waveform to pressure at the nearest cell center."""

    position: Vec3
    waveform: torch.Tensor

    def bake(self, grid: Grid, device: torch.device) -> BakedSource:
        i, j, k = grid.cell_index(self.position)
        flat = (i * grid.shape[1] + j) * grid.shape[2] + k
        baked = BakedSource(
            waveform=self.waveform,
            p_idx=torch.tensor([flat], dtype=torch.int64),
            p_weight=torch.tensor([1.0], dtype=torch.float32),
        )
        return baked.to(device)


def _face_coordinates(grid: Grid, axis: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Physical coordinates (meters) of every face carrying the ``axis`` velocity."""
    shape = grid.velocity_shape(axis)
    coords = []
    for a, n in enumerate(shape):
        c = torch.arange(n, dtype=torch.float64)
        c = (c + 1.0) * grid.dx if a == axis else (c + 0.5) * grid.dx
        coords.append(c)
    gx, gy, gz = torch.meshgrid(coords[0], coords[1], coords[2], indexing="ij")
    return gx, gy, gz


@dataclass(frozen=True)
class PistonSource:
    """Rigid vibrating piston: normal-velocity source over a (possibly annular,
    possibly tilted) disc.

    The waveform is the piston's normal velocity in m/s; each velocity
    component within one cell of the disc plane receives the corresponding
    normal-vector projection.

    With ``hard=True`` (default) the face values are overwritten each step —
    the correct model for a piston set into a rigid wall (place the disc on
    the wall's surface so the selected faces are the wall's boundary faces).
    A hard source in free air still radiates but is only approximate; set
    ``hard=False`` for a soft (additive) free-field source, which radiates a
    dipole-like cos(theta) pattern.
    """

    center: Vec3
    normal: Vec3
    radius: float
    waveform: torch.Tensor
    inner_radius: float = 0.0
    hard: bool = True

    def bake(self, grid: Grid, device: torch.device) -> BakedSource:
        n = torch.tensor(self.normal, dtype=torch.float64)
        norm = float(torch.linalg.vector_norm(n))
        if norm == 0.0:
            raise ValueError("piston normal must be non-zero")
        n = n / norm
        c = torch.tensor(self.center, dtype=torch.float64)

        idxs: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for axis in range(3):
            if abs(float(n[axis])) < 1e-12:
                idxs.append(torch.zeros(0, dtype=torch.int64))
                weights.append(torch.zeros(0, dtype=torch.float32))
                continue
            gx, gy, gz = _face_coordinates(grid, axis)
            dxv = gx - c[0]
            dyv = gy - c[1]
            dzv = gz - c[2]
            axial = dxv * n[0] + dyv * n[1] + dzv * n[2]
            radial_sq = dxv**2 + dyv**2 + dzv**2 - axial**2
            on_disc = (
                (axial.abs() <= 0.5 * grid.dx)
                & (radial_sq <= self.radius**2)
                & (radial_sq >= self.inner_radius**2)
            )
            flat = on_disc.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            idxs.append(flat)
            weights.append(torch.full((flat.shape[0],), float(n[axis]), dtype=torch.float32))
        if all(i.numel() == 0 for i in idxs):
            raise ValueError("piston disc does not intersect any velocity face on this grid")
        baked = BakedSource(
            waveform=self.waveform,
            v_idx=(idxs[0], idxs[1], idxs[2]),
            v_weight=(weights[0], weights[1], weights[2]),
            hard=self.hard,
        )
        return baked.to(device)


@dataclass(frozen=True)
class RectangularPistonSource:
    """Rectangular vibrating diaphragm (planar-magnetic style): hard/soft
    normal-velocity source over a width x height rectangle.

    The rectangle's height axis is the projection of the scene +y direction
    onto the source plane; width is perpendicular to it. Same hard-source
    semantics as :class:`PistonSource`.
    """

    center: Vec3
    normal: Vec3
    width: float
    height: float
    waveform: torch.Tensor
    hard: bool = True

    def bake(self, grid: Grid, device: torch.device) -> BakedSource:
        n = torch.tensor(self.normal, dtype=torch.float64)
        norm = float(torch.linalg.vector_norm(n))
        if norm == 0.0:
            raise ValueError("piston normal must be non-zero")
        n = n / norm
        helper = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        if abs(float(n[1])) > 0.9:
            helper = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        u = helper - n * float(torch.dot(helper, n))  # in-plane "height" axis
        u = u / torch.linalg.vector_norm(u)
        w = torch.linalg.cross(n, u)
        c = torch.tensor(self.center, dtype=torch.float64)

        idxs: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for axis in range(3):
            if abs(float(n[axis])) < 1e-12:
                idxs.append(torch.zeros(0, dtype=torch.int64))
                weights.append(torch.zeros(0, dtype=torch.float32))
                continue
            gx, gy, gz = _face_coordinates(grid, axis)
            dxv, dyv, dzv = gx - c[0], gy - c[1], gz - c[2]
            axial = dxv * n[0] + dyv * n[1] + dzv * n[2]
            a = dxv * u[0] + dyv * u[1] + dzv * u[2]
            b = dxv * w[0] + dyv * w[1] + dzv * w[2]
            on_rect = (
                (axial.abs() <= 0.5 * grid.dx)
                & (a.abs() <= self.height / 2.0)
                & (b.abs() <= self.width / 2.0)
            )
            flat = on_rect.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            idxs.append(flat)
            weights.append(torch.full((flat.shape[0],), float(n[axis]), dtype=torch.float32))
        if all(i.numel() == 0 for i in idxs):
            raise ValueError("rectangular piston does not intersect any velocity face")
        baked = BakedSource(
            waveform=self.waveform,
            v_idx=(idxs[0], idxs[1], idxs[2]),
            v_weight=(weights[0], weights[1], weights[2]),
            hard=self.hard,
        )
        return baked.to(device)


@dataclass(frozen=True)
class RigidBodySource:
    """Voxelized rigid body translating along ``direction``; the waveform is
    the body's speed in m/s (e.g. a dome diaphragm oscillating on its axis).

    For pure translation with velocity ``U(t) * d`` the boundary condition on
    every surface point is the body's own velocity, so the face carrying grid
    component ``axis`` takes the value ``U(t) * d[axis]`` — a uniform weight
    per component. The body's shape enters only through *which* faces are
    selected (the voxel staircase), never through per-face normal weights.

    Faces are the body's open boundary faces: one adjacent cell inside
    ``occupancy``, the other outside ``solid | occupancy`` — exactly the faces
    the solid masks zero each step, which the hard overwrite then re-drives
    (same mechanism as the flat piston-on-rigid-baffle model). With
    ``hard=True`` the faces clamp to zero after the waveform ends, leaving the
    body as a rigid scatterer.
    """

    occupancy: torch.Tensor  # bool, grid.shape: the moving body
    solid: torch.Tensor  # bool, grid.shape: all scene solids (may include the body)
    direction: Vec3  # motion axis (unnormalized ok)
    waveform: torch.Tensor
    hard: bool = True
    # Optional subset of ``occupancy`` that actually moves (e.g. the central
    # dome when the surround is decoupled at high frequency); the rest of the
    # body stays a static rigid scatterer. None = the whole body moves.
    drive_occupancy: torch.Tensor | None = None
    # Optional per-cell velocity amplitude (grid.shape, float): a flexing
    # surface such as a coil-driven dome with a rim-clamped surround whose
    # amplitude tapers to zero. Each face uses the moving-side cell's value;
    # None = uniform rigid motion.
    amplitude_field: torch.Tensor | None = None

    def bake(self, grid: Grid, device: torch.device) -> BakedSource:
        d = torch.tensor(self.direction, dtype=torch.float64)
        norm = float(torch.linalg.vector_norm(d))
        if norm == 0.0:
            raise ValueError("rigid body direction must be non-zero")
        d = d / norm
        if tuple(self.occupancy.shape) != grid.shape or tuple(self.solid.shape) != grid.shape:
            raise ValueError("occupancy and solid must match grid.shape")
        blocked = self.solid | self.occupancy
        drive = self.occupancy if self.drive_occupancy is None else self.drive_occupancy
        if tuple(drive.shape) != grid.shape:
            raise ValueError("drive_occupancy must match grid.shape")
        amp = self.amplitude_field
        if amp is not None and tuple(amp.shape) != grid.shape:
            raise ValueError("amplitude_field must match grid.shape")

        idxs: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for axis in range(3):
            if abs(float(d[axis])) < 1e-12:
                idxs.append(torch.zeros(0, dtype=torch.int64))
                weights.append(torch.zeros(0, dtype=torch.float32))
                continue
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis] = slice(None, -1)
            hi[axis] = slice(1, None)
            mov_lo = drive[tuple(lo)]
            mov_hi = drive[tuple(hi)]
            blk_lo = blocked[tuple(lo)]
            blk_hi = blocked[tuple(hi)]
            face = (mov_lo & ~blk_hi) | (mov_hi & ~blk_lo)  # shape == velocity_shape(axis)
            flat = face.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            idxs.append(flat)
            if amp is None:
                weights.append(
                    torch.full((flat.shape[0],), float(d[axis]), dtype=torch.float32)
                )
            else:
                # Amplitude of the moving-side cell (lo where the lo cell
                # drives the face, else hi).
                amp_lo = amp[tuple(lo)].reshape(-1)[flat]
                amp_hi = amp[tuple(hi)].reshape(-1)[flat]
                from_lo = (mov_lo & ~blk_hi).reshape(-1)[flat]
                a_face = torch.where(from_lo, amp_lo, amp_hi)
                weights.append((float(d[axis]) * a_face).to(torch.float32))
        if all(i.numel() == 0 for i in idxs):
            raise ValueError("rigid body has no open boundary faces on this grid")
        baked = BakedSource(
            waveform=self.waveform,
            v_idx=(idxs[0], idxs[1], idxs[2]),
            v_weight=(weights[0], weights[1], weights[2]),
            hard=self.hard,
        )
        return baked.to(device)


Source = PointSource | PistonSource | RectangularPistonSource | RigidBodySource
