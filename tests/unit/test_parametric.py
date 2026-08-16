import math

import pytest

from headphone_sims.geometry.parametric import HexHoles, RingSlits, Slots, cup_shell, plate
from headphone_sims.grid import Grid


def _grid(n: int = 80, dx: float = 0.5e-3) -> Grid:
    return Grid.create((n, n, n), dx=dx)


def test_solid_plate_volume() -> None:
    grid = _grid()
    radius, thickness = 10e-3, 1e-3
    occ, porosity = plate(grid, (20e-3, 20e-3, 20e-3), (0.0, 0.0, 1.0), radius, thickness)
    assert porosity == 0.0
    expected_cells = math.pi * radius**2 * thickness / grid.dx**3
    assert int(occ.sum()) == pytest.approx(expected_cells, rel=0.10)


def test_hex_porosity_matches_analytic() -> None:
    grid = _grid(120)
    hole_r, pitch = 1e-3, 3e-3
    occ, porosity = plate(
        grid,
        (30e-3, 30e-3, 30e-3),
        (0.0, 0.0, 1.0),
        radius=25e-3,
        thickness=1e-3,
        pattern=HexHoles(hole_radius=hole_r, pitch=pitch),
    )
    # Hex lattice open fraction = (2*pi/sqrt(3)) * (r/pitch)^2.
    analytic = (2.0 * math.pi / math.sqrt(3.0)) * (hole_r / pitch) ** 2
    assert porosity == pytest.approx(analytic, rel=0.15)
    assert int(occ.sum()) > 0


def test_slots_porosity() -> None:
    grid = _grid(120)
    _occ, porosity = plate(
        grid,
        (30e-3, 30e-3, 30e-3),
        (0.0, 0.0, 1.0),
        radius=25e-3,
        thickness=1e-3,
        pattern=Slots(width=1e-3, pitch=4e-3),
    )
    assert porosity == pytest.approx(0.25, abs=0.05)


def test_ring_slits_open_annulus() -> None:
    grid = _grid(120)
    _occ, porosity = plate(
        grid,
        (30e-3, 30e-3, 30e-3),
        (0.0, 0.0, 1.0),
        radius=20e-3,
        thickness=1e-3,
        pattern=RingSlits(rings=((5e-3, 8e-3),)),
    )
    analytic = (8e-3**2 - 5e-3**2) / 20e-3**2
    assert porosity == pytest.approx(analytic, rel=0.10)


def test_tilted_plate_volume_near_analytic() -> None:
    # Center deliberately off the cell-boundary lattice: an exactly aligned
    # thin plate quantizes its thickness by up to +/- one cell layer.
    grid = _grid()
    radius, thickness = 8e-3, 1.5e-3
    analytic_cells = math.pi * radius**2 * thickness / grid.dx**3
    for normal in ((0.0, 0.0, 1.0), (0.0, 1.0, 2.0)):
        occ, _ = plate(grid, (20.1e-3, 20.2e-3, 20.3e-3), normal, radius, thickness)
        assert int(occ.sum()) == pytest.approx(analytic_cells, rel=0.15)


def test_cup_shell_is_open_on_rim_side() -> None:
    grid = _grid(100)
    occ = cup_shell(
        grid,
        center=(25e-3, 25e-3, 25e-3),
        axis=(0.0, 0.0, -1.0),
        inner_radius=15e-3,
        depth=12e-3,
        thickness=2e-3,
    )
    assert int(occ.sum()) > 0
    # Interior of the cup (on-axis, inside depth) stays open.
    i = int(25e-3 / grid.dx)
    k_inside = int((25e-3 - 6e-3) / grid.dx)
    assert not bool(occ[i, i, k_inside])
    # Back plate is solid.
    k_back = int((25e-3 - 13e-3) / grid.dx)
    assert bool(occ[i, i, k_back])
