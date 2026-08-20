"""Probe receiver-stencil air checks and placement clearance (issue #6)."""

import numpy as np
import torch

from headphone_sims.geometry.scene import probe_solid_clearance, probe_stencil_air_mask
from headphone_sims.grid import Grid


def _slab_grid() -> tuple[Grid, torch.Tensor]:
    """40mm cube at dx=1mm with a solid slab filling z-cells 0..9 (surface at
    z = 10 mm; topmost solid cell center at 9.5 mm)."""
    grid = Grid.create((40, 40, 40), dx=1e-3)
    solid = torch.zeros(grid.shape, dtype=torch.bool)
    solid[:, :, :10] = True
    return grid, solid


def test_clear_probe_passes() -> None:
    grid, solid = _slab_grid()
    probes = np.array([[20e-3, 20e-3, 12.6e-3]])
    assert probe_stencil_air_mask(probes, solid, grid).all()


def test_velocity_stencil_touching_solid_fails() -> None:
    # At z = 10.8 mm the PRESSURE stencil (cells 10, 11) is clear of the slab,
    # but the vz stencil (faces 9, 10 -> cells 9..11) touches solid cell 9 —
    # the face the rigid BC forces to zero. The old pressure-only check
    # accepted this probe; the stencil-aware check must reject it.
    grid, solid = _slab_grid()
    probes = np.array([[20e-3, 20e-3, 10.8e-3]])
    assert not probe_stencil_air_mask(probes, solid, grid).any()


def test_probe_inside_solid_fails() -> None:
    grid, solid = _slab_grid()
    probes = np.array([[20e-3, 20e-3, 5e-3]])
    assert not probe_stencil_air_mask(probes, solid, grid).any()


def test_out_of_domain_stencil_rejected_not_clamped() -> None:
    grid, solid = _slab_grid()
    # In air but so close to the domain edge that interpolation stencils
    # would be clamped/shifted — must be rejected.
    probes = np.array(
        [
            [20e-3, 20e-3, 39.9e-3],
            [0.2e-3, 20e-3, 20e-3],
            [20e-3, 20e-3, -1e-3],
        ]
    )
    assert not probe_stencil_air_mask(probes, solid, grid).any()


def test_lateral_velocity_stencil_detects_side_wall() -> None:
    # A solid column beside the probe: pressure corners are air but the vx
    # stencil (3 cells along x) reaches into the column.
    grid, solid = _slab_grid()
    solid[:, :, :] = False
    solid[22, :, :] = True
    probes = np.array([[20.6e-3, 20e-3, 20e-3]])  # p cells x=20,21; vx cells 19..21? -> see below
    # u_x = 20.6: p base 20 (cells 20,21 clear); vx face base floor(19.6)=19 ->
    # cells 19..21 clear of x=22 -> passes.
    assert probe_stencil_air_mask(probes, solid, grid).all()
    probes = np.array([[21.4e-3, 20e-3, 20e-3]])
    # u_x = 21.4: p base 20 (cells 20,21 clear of the x=22 column); vx face
    # base floor(20.4)=20 -> cells 20..22 include the column -> must fail.
    assert not probe_stencil_air_mask(probes, solid, grid).any()


def test_clearance_measures_distance_to_voxel_centers() -> None:
    grid, solid = _slab_grid()
    # xy on a cell center so the nearest solid center (20.5, 20.5, 9.5mm) is
    # directly below the first probe.
    probes = np.array([[20.5e-3, 20.5e-3, 12.6e-3], [20.5e-3, 20.5e-3, 30e-3]])
    clearance = probe_solid_clearance(probes, solid, grid)
    np.testing.assert_allclose(clearance[0], 3.1e-3, atol=1e-6)
    assert np.isinf(clearance[1])  # beyond the 5mm search window
