"""Absorbing boundaries (graded sponge, C-PML) and damping-tensor assembly.

The default absorbing boundary is a graded sponge: an exponential damping
factor ``exp(-sigma * dt)`` applied to all fields each step, where ``sigma``
ramps polynomially from 0 at the inner sponge edge to ``sigma_max`` at the
domain edge (measured reflection < -40 dB). Lossy damping materials (earpads,
felt, viscous bore losses) contribute an additional local ``sigma`` on
velocity. Solid face masks are folded into the same velocity multipliers so
the kernel applies a single multiply per field per step.

The optional convolutional PML (:class:`CpmlConfig`) replaces the sponge with
Roden & Gedney recursive-convolution memory variables on the pressure
gradient and velocity divergence inside the boundary slabs — a substantially
cleaner floor (target reflection ~1e-6 by grading design, < -60 dB measured)
for late-window / fine-spectral-structure work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from headphone_sims.fdtd.kernel import FdtdState
from headphone_sims.grid import Grid


@dataclass(frozen=True)
class SpongeConfig:
    """Graded absorbing layer covering the outermost ``thickness`` cells."""

    thickness: int = 30
    sigma_max: float = 60_000.0  # 1/s at the domain edge
    order: float = 3.0  # polynomial grading exponent

    def profile(self, n: int, offset: float, device: torch.device) -> torch.Tensor:
        """1D sigma profile along an axis of ``n`` samples.

        ``offset`` is the staggered coordinate shift in cells (0.0 for cell
        centers, 0.5 for the face-normal velocity component).
        """
        coords = torch.arange(n, dtype=torch.float64, device=device) + offset
        far_edge = n - 1 + 2 * offset  # last sample position in cell units
        depth_lo = (self.thickness - coords).clamp(min=0.0) / self.thickness
        depth_hi = (coords - (far_edge - self.thickness)).clamp(min=0.0) / self.thickness
        depth = torch.maximum(depth_lo, depth_hi).clamp(max=1.0)
        return self.sigma_max * depth**self.order


def sigma_field(
    grid: Grid,
    sponge: SpongeConfig,
    field_shape: tuple[int, int, int],
    stagger_axis: int | None,
    device: torch.device,
) -> torch.Tensor:
    """3D sigma tensor for a (possibly staggered) field: sum of per-axis ramps."""
    sigma = torch.zeros(field_shape, dtype=torch.float64, device=device)
    for axis, n in enumerate(field_shape):
        offset = 0.5 if axis == stagger_axis else 0.0
        prof = sponge.profile(n, offset, device)
        shape = [1, 1, 1]
        shape[axis] = n
        sigma += prof.view(shape)
    return sigma


def attach_damping(
    state: FdtdState,
    sponge: SpongeConfig | None = None,
    solid: torch.Tensor | None = None,
    sigma_material: torch.Tensor | None = None,
) -> None:
    """Build and attach the per-field damping/mask multipliers to ``state``.

    - ``solid``: bool occupancy at cell centers ``grid.shape``; faces adjacent
      to a solid cell get zero normal velocity (exact rigid boundary).
    - ``sigma_material``: float damping rate (1/s) at cell centers, averaged
      onto faces, e.g. from
      :func:`headphone_sims.medium.flow_resistivity_to_sigma`.
    """
    grid = state.grid
    device = state.device
    dt = grid.dt

    def velocity_mult(axis: int) -> torch.Tensor | None:
        shape = grid.velocity_shape(axis)
        sigma = torch.zeros(shape, dtype=torch.float64, device=device)
        if sponge is not None:
            sigma += sigma_field(grid, sponge, shape, axis, device)
        if sigma_material is not None:
            sm = sigma_material.to(device=device, dtype=torch.float64)
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis] = slice(None, -1)
            hi[axis] = slice(1, None)
            sigma += 0.5 * (sm[tuple(lo)] + sm[tuple(hi)])
        if sponge is None and sigma_material is None and solid is None:
            return None
        mult = torch.exp(-sigma * dt)
        if solid is not None:
            s = solid.to(device=device, dtype=torch.bool)
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis] = slice(None, -1)
            hi[axis] = slice(1, None)
            open_face = ~(s[tuple(lo)] | s[tuple(hi)])
            mult = mult * open_face
        return mult.to(state.p.dtype)

    state.mult_vx = velocity_mult(0)
    state.mult_vy = velocity_mult(1)
    state.mult_vz = velocity_mult(2)

    if sponge is not None:
        sigma_p = sigma_field(grid, sponge, grid.shape, None, device)
        state.damp_p = torch.exp(-sigma_p * dt).to(state.p.dtype)
    else:
        state.damp_p = None


@dataclass(frozen=True)
class CpmlConfig:
    """Convolutional PML (Roden & Gedney recursive convolution, kappa = 1).

    ``sigma_max`` follows the standard polynomial-grading design
    ``-(order+1) c ln(R) / (2 L)`` for target reflection ``R`` over depth
    ``L = thickness * dx``; ``alpha`` ramps linearly from ``alpha_max`` at the
    inner PML edge to 0 at the outer edge (grazing/evanescent stabilization).
    The outer domain boundary stays rigid — the returning wave is absorbed
    again on the way out.
    """

    thickness: int = 15
    target_reflection: float = 1e-6
    order: float = 3.0
    alpha_max: float = 600.0  # ~ pi * (lowest frequency of interest ~200 Hz)

    def coeffs(
        self, u: torch.Tensor, n_cells: int, c: float, dx: float, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(b, a) recursion coefficients at coordinates ``u`` (cell units,
        domain [0, n_cells]): psi <- b * psi + a * g."""
        sigma_max = -(self.order + 1.0) * c * math.log(self.target_reflection)
        sigma_max /= 2.0 * self.thickness * dx
        depth_lo = ((self.thickness - u) / self.thickness).clamp(min=0.0)
        depth_hi = ((u - (n_cells - self.thickness)) / self.thickness).clamp(min=0.0)
        depth = torch.maximum(depth_lo, depth_hi).clamp(max=1.0)
        sigma = sigma_max * depth**self.order
        alpha = self.alpha_max * (1.0 - depth)
        b = torch.exp(-(sigma + alpha) * dt)
        denom = sigma + alpha
        a = torch.where(denom > 0.0, sigma / denom.clamp(min=1e-30) * (b - 1.0), sigma * 0.0)
        return b, a


@dataclass
class _CpmlSlab:
    """One boundary slab of one axis: memory variable + broadcast coefficients."""

    axis: int
    lo: bool  # True = low-index side
    psi: torch.Tensor
    b: torch.Tensor  # broadcastable over the slab
    a: torch.Tensor


@dataclass
class CpmlState:
    """Baked C-PML slabs. ``v_slabs`` correct the pressure-gradient term of
    the velocity update; ``p_slabs`` correct the velocity-divergence term of
    the pressure update. All quantities are in the kernel's unscaled
    difference units, so the kernel applies them with its own cv/cp factors.
    """

    thickness: int
    v_slabs: list[_CpmlSlab] = field(default_factory=list)
    p_slabs: list[_CpmlSlab] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        grid: Grid,
        config: CpmlConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> CpmlState:
        t = config.thickness
        c = grid.medium.sound_speed
        state = cls(thickness=t)
        for axis in range(3):
            n = grid.shape[axis]
            if n < 2 * t + 4:
                raise ValueError(f"grid axis {axis} too small for a {t}-cell CPML")
            v_shape = list(grid.velocity_shape(axis))
            p_shape = list(grid.shape)
            for lo in (True, False):
                # Velocity slab: faces at u = i + 1 (i local face index).
                face_idx = torch.arange(t, dtype=torch.float64)
                u_face = (face_idx + 1.0) if lo else (v_shape[axis] - t + face_idx + 1.0)
                b, a = config.coeffs(u_face, n, c, grid.dx, grid.dt)
                shape = [1, 1, 1]
                shape[axis] = t
                sl = v_shape.copy()
                sl[axis] = t
                state.v_slabs.append(
                    _CpmlSlab(
                        axis=axis,
                        lo=lo,
                        psi=torch.zeros(sl, device=device, dtype=dtype),
                        b=b.view(shape).to(device=device, dtype=dtype),
                        a=a.view(shape).to(device=device, dtype=dtype),
                    )
                )
                # Pressure slab: cells at u = i + 0.5.
                cell_idx = torch.arange(t, dtype=torch.float64)
                u_cell = (cell_idx + 0.5) if lo else (n - t + cell_idx + 0.5)
                b, a = config.coeffs(u_cell, n, c, grid.dx, grid.dt)
                sl = p_shape.copy()
                sl[axis] = t
                state.p_slabs.append(
                    _CpmlSlab(
                        axis=axis,
                        lo=lo,
                        psi=torch.zeros(sl, device=device, dtype=dtype),
                        b=b.view(shape).to(device=device, dtype=dtype),
                        a=a.view(shape).to(device=device, dtype=dtype),
                    )
                )
        return state

    @staticmethod
    def _slicer(axis: int, sl: slice) -> tuple[slice, slice, slice]:
        s: list[slice] = [slice(None)] * 3
        s[axis] = sl
        return (s[0], s[1], s[2])

    def apply_velocity(self, state: FdtdState, cv: float) -> None:
        """Add the memory-variable correction to the velocity update.

        Call after the plain gradient update and before the masks: the same
        pressure field feeds both, so psi sees this step's gradient.
        """
        p = state.p
        vs = (state.vx, state.vy, state.vz)
        t = self.thickness
        for slab in self.v_slabs:
            ax = slab.axis
            v = vs[ax]
            nf = v.shape[ax]
            f = slice(0, t) if slab.lo else slice(nf - t, nf)
            # grad slab: p[i+1] - p[i] for faces i in f (p index = face index).
            g = p[self._slicer(ax, slice(f.start + 1, f.stop + 1))] - p[self._slicer(ax, f)]
            slab.psi.mul_(slab.b).add_(g * slab.a)
            v[self._slicer(ax, f)].sub_(slab.psi, alpha=cv)

    def apply_pressure(self, state: FdtdState, cp: float) -> None:
        """Add the memory-variable correction to the pressure update (same
        one-sided edge handling as the kernel: domain-edge faces are rigid)."""
        vs = (state.vx, state.vy, state.vz)
        t = self.thickness
        for slab in self.p_slabs:
            ax = slab.axis
            v = vs[ax]
            n = state.p.shape[ax]
            if slab.lo:
                # cells 0..t-1: g_i = v[i] - v[i-1], v[-1] = 0 (rigid edge)
                g = v[self._slicer(ax, slice(0, t))].clone()
                g[self._slicer(ax, slice(1, t))] -= v[self._slicer(ax, slice(0, t - 1))]
                cells = slice(0, t)
            else:
                # cells n-t..n-1: g_i = v[i] - v[i-1], v[n-1] = 0 (rigid edge)
                c0 = n - t
                g = -v[self._slicer(ax, slice(c0 - 1, n - 1))]
                g[self._slicer(ax, slice(0, t - 1))] += v[self._slicer(ax, slice(c0, n - 1))]
                cells = slice(c0, n)
            slab.psi.mul_(slab.b).add_(g * slab.a)
            state.p[self._slicer(ax, cells)].sub_(slab.psi, alpha=cp)
