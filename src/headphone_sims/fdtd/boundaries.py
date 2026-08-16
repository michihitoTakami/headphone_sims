"""Absorbing sponge boundary and damping-tensor assembly.

The outer absorbing boundary is a graded sponge: an exponential damping factor
``exp(-sigma * dt)`` applied to all fields each step, where ``sigma`` ramps
polynomially from 0 at the inner sponge edge to ``sigma_max`` at the domain
edge. Lossy damping materials (earpads, felt) contribute an additional local
``sigma`` on velocity. Solid face masks are folded into the same velocity
multipliers so the kernel applies a single multiply per field per step.
"""

from __future__ import annotations

from dataclasses import dataclass

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
