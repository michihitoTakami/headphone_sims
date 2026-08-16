import math

import pytest
import torch

from headphone_sims.fdtd.kernel import FdtdState
from headphone_sims.fdtd.sources import (
    PistonSource,
    PointSource,
    gaussian_modulated_sine,
    ricker,
)
from headphone_sims.grid import Grid


def test_ricker_zero_dc() -> None:
    w = ricker(dt=1e-6, n_steps=4000, peak_frequency=5000.0)
    assert abs(float(w.sum())) < 1e-3 * float(w.abs().max())


def test_gaussian_burst_peak_near_center_frequency() -> None:
    dt = 1e-6
    w = gaussian_modulated_sine(dt, 4096, center_frequency=10_000.0, bandwidth_frequency=5000.0)
    spec = torch.fft.rfft(w.to(torch.float64)).abs()
    freqs = torch.fft.rfftfreq(w.shape[0], dt)
    peak = float(freqs[int(spec.argmax())])
    assert peak == pytest.approx(10_000.0, rel=0.05)


def test_point_source_injects_at_cell() -> None:
    grid = Grid.create((16, 16, 16), dx=1e-3)
    state = FdtdState.zeros(grid)
    wf = torch.tensor([2.5, 0.0])
    baked = PointSource(position=(8.5e-3, 8.5e-3, 8.5e-3), waveform=wf).bake(
        grid, torch.device("cpu")
    )
    baked.inject_pressure(state.p, 0)
    assert float(state.p[8, 8, 8]) == pytest.approx(2.5)
    assert float(state.p.abs().sum()) == pytest.approx(2.5)


def test_hard_piston_overwrites_velocity() -> None:
    grid = Grid.create((32, 32, 32), dx=1e-3)
    state = FdtdState.zeros(grid)
    wf = torch.tensor([3.0])
    baked = PistonSource(
        center=(16e-3, 16e-3, 16e-3), normal=(0.0, 0.0, 1.0), radius=5e-3, waveform=wf
    ).bake(grid, torch.device("cpu"))
    v = (state.vx, state.vy, state.vz)
    state.vz.fill_(99.0)
    baked.inject_velocity(v, 0)
    assert baked.v_idx is not None
    touched = state.vz.view(-1)[baked.v_idx[2]]
    assert torch.allclose(touched, torch.full_like(touched, 3.0))
    # Past the end of the waveform a hard source clamps its faces to zero.
    baked.inject_velocity(v, 5)
    touched = state.vz.view(-1)[baked.v_idx[2]]
    assert torch.allclose(touched, torch.zeros_like(touched))


def test_piston_face_count_matches_disc_area() -> None:
    grid = Grid.create((64, 64, 64), dx=1e-3)
    wf = torch.zeros(4)
    radius = 10e-3
    baked = PistonSource(
        center=(32e-3, 32e-3, 10e-3), normal=(0.0, 0.0, 1.0), radius=radius, waveform=wf
    ).bake(grid, torch.device("cpu"))
    assert baked.v_idx is not None
    # Axis-aligned piston excites only the z-component.
    assert baked.v_idx[0].numel() == 0
    assert baked.v_idx[1].numel() == 0
    n_faces = baked.v_idx[2].numel()
    expected = math.pi * radius**2 / grid.dx**2
    assert n_faces == pytest.approx(expected, rel=0.05)


def test_tilted_piston_excites_multiple_components() -> None:
    grid = Grid.create((64, 64, 64), dx=1e-3)
    wf = torch.zeros(4)
    baked = PistonSource(
        center=(32e-3, 32e-3, 16e-3),
        normal=(0.0, 1.0, 2.0),
        radius=8e-3,
        waveform=wf,
    ).bake(grid, torch.device("cpu"))
    assert baked.v_idx is not None
    assert baked.v_idx[1].numel() > 0
    assert baked.v_idx[2].numel() > 0
    # Weights carry the normal projection.
    assert baked.v_weight is not None
    ny = 1.0 / math.sqrt(5.0)
    assert float(baked.v_weight[1][0]) == pytest.approx(ny, rel=1e-5)


def test_annular_piston_excludes_center() -> None:
    grid = Grid.create((64, 64, 64), dx=1e-3)
    wf = torch.zeros(4)
    full = PistonSource(
        center=(32e-3, 32e-3, 10e-3), normal=(0.0, 0.0, 1.0), radius=10e-3, waveform=wf
    ).bake(grid, torch.device("cpu"))
    ring = PistonSource(
        center=(32e-3, 32e-3, 10e-3),
        normal=(0.0, 0.0, 1.0),
        radius=10e-3,
        inner_radius=5e-3,
        waveform=wf,
    ).bake(grid, torch.device("cpu"))
    assert full.v_idx is not None and ring.v_idx is not None
    assert 0 < ring.v_idx[2].numel() < full.v_idx[2].numel()


def test_piston_missing_grid_raises() -> None:
    grid = Grid.create((16, 16, 16), dx=1e-3)
    with pytest.raises(ValueError, match="does not intersect"):
        PistonSource(
            center=(8e-3, 8e-3, 8e-3), normal=(0.0, 0.0, 1.0), radius=1e-5, waveform=torch.zeros(2)
        ).bake(grid, torch.device("cpu"))
