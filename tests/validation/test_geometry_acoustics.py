"""Acoustic validation of voxelized geometry: airtightness and piston directivity."""

import numpy as np
import pytest
import torch
import trimesh
from scipy.special import j1

from headphone_sims.fdtd.boundaries import SpongeConfig
from headphone_sims.fdtd.receivers import ReceiverArray
from headphone_sims.fdtd.simulation import Simulation
from headphone_sims.fdtd.sources import PistonSource, PointSource, gaussian_modulated_sine, ricker
from headphone_sims.geometry.mesh import voxelize
from headphone_sims.geometry.parametric import plate
from headphone_sims.grid import Grid


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_voxelized_shell_is_airtight() -> None:
    """A hollow voxelized sphere with a source inside must leak nothing."""
    grid = Grid.create((60, 60, 60), dx=1e-3)
    center = (30e-3, 30e-3, 30e-3)
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=10e-3)
    sphere.apply_translation(center)
    shell = voxelize(sphere, grid, fill=False)

    n_steps = 250
    wf = ricker(grid.dt, n_steps, peak_frequency=10_000.0)
    probe_out = np.array([[52e-3, 30e-3, 30e-3]])
    sim = Simulation(
        grid=grid,
        sources=[PointSource(position=center, waveform=wf)],
        receivers=ReceiverArray(probe_out),
        n_steps=n_steps,
        solid=shell,
        sponge=None,
        device=_device(),
    )
    res = sim.run()
    assert float(np.abs(res.p).max()) == 0.0


@pytest.mark.slow
def test_baffled_piston_directivity() -> None:
    """Far-field directivity of a baffled piston vs. analytic 2*J1(x)/x."""
    dx = 1e-3
    grid = Grid.create((220, 220, 200), dx=dx)
    n_steps = 560
    freq = 10_000.0
    a = 15e-3  # piston radius

    wf = gaussian_modulated_sine(
        grid.dt, n_steps, center_frequency=freq, bandwidth_frequency=4000.0
    )
    cx = 110 * dx
    z0 = 45e-3
    driver_center = (cx, cx, z0)

    # Solid wall whose front surface lies exactly on the source plane z0, so
    # the (hard) piston faces are the wall's own boundary faces.
    baffle, _ = plate(
        grid,
        (cx, cx, z0 - dx),
        (0.0, 0.0, 1.0),
        radius=95e-3,
        thickness=2 * dx,
    )

    r = 100e-3
    angles = np.deg2rad(np.arange(0.0, 45.0, 5.0))
    probes = np.stack(
        [
            cx + r * np.sin(angles),
            np.full_like(angles, cx),
            z0 + r * np.cos(angles),
        ],
        axis=1,
    )
    sim = Simulation(
        grid=grid,
        sources=[PistonSource(center=driver_center, normal=(0.0, 0.0, 1.0), radius=a, waveform=wf)],
        receivers=ReceiverArray(probes),
        n_steps=n_steps,
        solid=baffle,
        sponge=SpongeConfig(thickness=25),
        device=_device(),
    )
    res = sim.run()

    spec = np.abs(np.fft.rfft(res.p, axis=0))
    freqs = np.fft.rfftfreq(res.p.shape[0], res.dt)
    bin_f = int(np.argmin(np.abs(freqs - freq)))
    magnitude = spec[bin_f, :]
    measured = magnitude / magnitude[0]

    k = 2.0 * np.pi * freq / grid.medium.sound_speed
    x = k * a * np.sin(angles)
    with np.errstate(invalid="ignore", divide="ignore"):
        analytic = np.where(x == 0.0, 1.0, 2.0 * j1(x) / np.where(x == 0.0, 1.0, x))

    np.testing.assert_allclose(measured, np.abs(analytic), atol=0.10)
