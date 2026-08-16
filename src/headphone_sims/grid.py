"""Uniform staggered FDTD grid definition.

Field layout (staggered, Yee-like):
- pressure ``p`` at cell centers, shape ``(nx, ny, nz)``
- particle velocity on interior cell faces:
  ``vx`` at ``(nx-1, ny, nz)``, ``vy`` at ``(nx, ny-1, nz)``, ``vz`` at ``(nx, ny, nz-1)``

Domain-edge faces carry no velocity component, i.e. the outer boundary is rigid;
an absorbing sponge layer (see :mod:`headphone_sims.fdtd.boundaries`) prevents
reflections from reaching the region of interest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from headphone_sims.medium import AIR, Medium


@dataclass(frozen=True)
class Grid:
    """Uniform Cartesian grid with CFL-stable time step."""

    shape: tuple[int, int, int]
    dx: float
    dt: float
    medium: Medium = AIR

    @classmethod
    def create(
        cls,
        shape: tuple[int, int, int],
        dx: float,
        medium: Medium = AIR,
        courant: float = 0.9,
    ) -> Grid:
        """Build a grid with dt set from the 3D CFL limit dt <= dx / (c*sqrt(3))."""
        if not 0.0 < courant <= 1.0:
            raise ValueError(f"courant must be in (0, 1], got {courant}")
        if dx <= 0.0:
            raise ValueError(f"dx must be positive, got {dx}")
        if any(n < 3 for n in shape):
            raise ValueError(f"grid shape must be at least 3 cells per axis, got {shape}")
        dt = courant * dx / (medium.sound_speed * math.sqrt(3.0))
        return cls(shape=shape, dx=dx, dt=dt, medium=medium)

    @property
    def sample_rate(self) -> float:
        return 1.0 / self.dt

    @property
    def extent(self) -> tuple[float, float, float]:
        """Physical size of the domain in meters."""
        return (self.shape[0] * self.dx, self.shape[1] * self.dx, self.shape[2] * self.dx)

    @property
    def n_cells(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    def velocity_shape(self, axis: int) -> tuple[int, int, int]:
        """Shape of the staggered velocity array along ``axis``."""
        s = list(self.shape)
        s[axis] -= 1
        return (s[0], s[1], s[2])

    def points_per_wavelength(self, frequency: float) -> float:
        return self.medium.sound_speed / (frequency * self.dx)

    def cell_index(self, position_m: tuple[float, float, float]) -> tuple[int, int, int]:
        """Nearest cell-center index for a physical position (meters)."""
        idx = tuple(round(x / self.dx - 0.5) for x in position_m)
        for i, (j, n) in enumerate(zip(idx, self.shape, strict=True)):
            if not 0 <= j < n:
                raise ValueError(
                    f"position {position_m} maps to index {idx}, outside grid axis {i} (n={n})"
                )
        return (idx[0], idx[1], idx[2])
