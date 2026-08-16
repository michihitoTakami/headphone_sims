"""Leapfrog update kernel for the staggered pressure-velocity FDTD scheme.

Update equations (linear acoustics):
    v_n += -(dt / (rho * dx)) * dp/dn        on interior cell faces
    p   += -(rho * c^2 * dt / dx) * div(v)   at cell centers

Rigid solids are enforced by zeroing the normal velocity on any face adjacent
to a solid cell; this is exact and airtight for voxelized geometry. Absorption
(sponge boundary and lossy damping materials) is folded into per-field
multiplicative damping tensors applied once per step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from headphone_sims.grid import Grid


@dataclass
class FdtdState:
    """All per-step tensors of a simulation.

    ``mult_v*`` combine the solid face masks with velocity damping
    (``mask * exp(-sigma*dt)``); ``damp_p`` is the pressure damping. Any of
    them may be ``None`` when there is nothing to mask or damp.
    """

    grid: Grid
    p: torch.Tensor
    vx: torch.Tensor
    vy: torch.Tensor
    vz: torch.Tensor
    mult_vx: torch.Tensor | None = None
    mult_vy: torch.Tensor | None = None
    mult_vz: torch.Tensor | None = None
    damp_p: torch.Tensor | None = None

    @classmethod
    def zeros(
        cls,
        grid: Grid,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> FdtdState:
        dev = torch.device(device)
        return cls(
            grid=grid,
            p=torch.zeros(grid.shape, device=dev, dtype=dtype),
            vx=torch.zeros(grid.velocity_shape(0), device=dev, dtype=dtype),
            vy=torch.zeros(grid.velocity_shape(1), device=dev, dtype=dtype),
            vz=torch.zeros(grid.velocity_shape(2), device=dev, dtype=dtype),
        )

    @property
    def device(self) -> torch.device:
        return self.p.device


def step_velocity(state: FdtdState) -> None:
    """Half of the leapfrog: velocity update from the pressure gradient + masks.

    Velocity sources must be injected *after* this call (so hard sources can
    overwrite the masked boundary faces) and before :func:`step_pressure`.
    """
    grid = state.grid
    cv = grid.dt / (grid.medium.density * grid.dx)
    p, vx, vy, vz = state.p, state.vx, state.vy, state.vz

    # Velocity update from pressure gradient (interior faces only).
    vx.sub_(p[1:, :, :] - p[:-1, :, :], alpha=cv)
    vy.sub_(p[:, 1:, :] - p[:, :-1, :], alpha=cv)
    vz.sub_(p[:, :, 1:] - p[:, :, :-1], alpha=cv)

    if state.mult_vx is not None:
        vx.mul_(state.mult_vx)
    if state.mult_vy is not None:
        vy.mul_(state.mult_vy)
    if state.mult_vz is not None:
        vz.mul_(state.mult_vz)


def step_pressure(state: FdtdState) -> None:
    """Other half of the leapfrog: pressure update from velocity divergence."""
    grid = state.grid
    c = grid.medium.sound_speed
    cp = grid.medium.density * c * c * grid.dt / grid.dx
    p, vx, vy, vz = state.p, state.vx, state.vy, state.vz

    # Domain-edge faces are rigid (no velocity component), so edge cells only
    # see their interior faces.
    p[:-1, :, :].sub_(vx, alpha=cp)
    p[1:, :, :].add_(vx, alpha=cp)
    p[:, :-1, :].sub_(vy, alpha=cp)
    p[:, 1:, :].add_(vy, alpha=cp)
    p[:, :, :-1].sub_(vz, alpha=cp)
    p[:, :, 1:].add_(vz, alpha=cp)

    if state.damp_p is not None:
        p.mul_(state.damp_p)


def step(state: FdtdState) -> None:
    """Advance the state by one full leapfrog time step (no source injection)."""
    step_velocity(state)
    step_pressure(state)


def field_energy(state: FdtdState) -> float:
    """Total acoustic energy (J per unit cell volume factor) — for conservation tests.

    E = sum(p^2 / (2 rho c^2)) + sum(rho |v|^2 / 2), times cell volume dx^3.
    Velocity components live on faces; the sum is a first-order approximation
    adequate for monotone-decay checks.
    """
    grid = state.grid
    rho = grid.medium.density
    c = grid.medium.sound_speed
    e_p = float((state.p**2).sum()) / (2.0 * rho * c * c)
    e_v = (
        0.5
        * rho
        * (float((state.vx**2).sum()) + float((state.vy**2).sum()) + float((state.vz**2).sum()))
    )
    return (e_p + e_v) * grid.dx**3
