import math

import pytest

from headphone_sims.grid import Grid


def test_cfl_time_step() -> None:
    g = Grid.create((100, 100, 100), dx=0.5e-3, courant=0.9)
    assert g.dt == pytest.approx(0.9 * 0.5e-3 / (343.0 * math.sqrt(3.0)))
    assert g.sample_rate > 1e6


def test_points_per_wavelength() -> None:
    g = Grid.create((10, 10, 10), dx=0.5e-3)
    assert g.points_per_wavelength(20_000.0) == pytest.approx(343.0 / (20_000.0 * 0.5e-3))


def test_velocity_shapes() -> None:
    g = Grid.create((10, 12, 14), dx=1e-3)
    assert g.velocity_shape(0) == (9, 12, 14)
    assert g.velocity_shape(1) == (10, 11, 14)
    assert g.velocity_shape(2) == (10, 12, 13)


def test_cell_index_round_trip() -> None:
    g = Grid.create((20, 20, 20), dx=1e-3)
    # Center of cell (5, 6, 7) is at ((5.5, 6.5, 7.5) * dx).
    assert g.cell_index((5.5e-3, 6.5e-3, 7.5e-3)) == (5, 6, 7)


def test_cell_index_out_of_bounds() -> None:
    g = Grid.create((10, 10, 10), dx=1e-3)
    with pytest.raises(ValueError, match="outside grid"):
        g.cell_index((100e-3, 5e-3, 5e-3))


def test_invalid_args() -> None:
    with pytest.raises(ValueError):
        Grid.create((10, 10, 10), dx=-1.0)
    with pytest.raises(ValueError):
        Grid.create((10, 10, 10), dx=1e-3, courant=1.5)
    with pytest.raises(ValueError):
        Grid.create((2, 10, 10), dx=1e-3)
