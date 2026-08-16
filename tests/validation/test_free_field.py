"""Analytic free-field validation: 1/r decay, propagation delay, boundary reflections."""

import numpy as np
import pytest
import torch

from headphone_sims.fdtd.boundaries import SpongeConfig
from headphone_sims.fdtd.receivers import ReceiverArray
from headphone_sims.fdtd.simulation import Simulation
from headphone_sims.fdtd.sources import PointSource, ricker
from headphone_sims.grid import Grid


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.slow
def test_monopole_amplitude_and_delay() -> None:
    n = 140
    dx = 2e-3
    grid = Grid.create((n, n, n), dx=dx)
    n_steps = 320
    wf = ricker(grid.dt, n_steps, peak_frequency=6000.0)
    center = ((n / 2) * dx, (n / 2) * dx, (n / 2) * dx)
    r1, r2 = 40e-3, 80e-3
    probes = np.array(
        [
            [center[0] + r1, center[1], center[2]],
            [center[0] + r2, center[1], center[2]],
        ]
    )
    sim = Simulation(
        grid=grid,
        sources=[PointSource(position=center, waveform=wf)],
        receivers=ReceiverArray(probes),
        n_steps=n_steps,
        sponge=SpongeConfig(thickness=25),
        device=_device(),
    )
    res = sim.run()

    a1 = float(np.abs(res.p[:, 0]).max())
    a2 = float(np.abs(res.p[:, 1]).max())
    assert a1 / a2 == pytest.approx(r2 / r1, rel=0.05)

    # Delay between the two probes via cross-correlation peak.
    xc = np.correlate(res.p[:, 1], res.p[:, 0], mode="full")
    lag = int(xc.argmax()) - (res.p.shape[0] - 1)
    expected_lag = (r2 - r1) / grid.medium.sound_speed / grid.dt
    assert lag == pytest.approx(expected_lag, abs=2.0)


@pytest.mark.slow
def test_sponge_reflection_below_minus_40db() -> None:
    n = 120
    dx = 2e-3
    grid = Grid.create((n, n, n), dx=dx)
    # Long enough for the boundary round trip to return to the probe.
    n_steps = 700
    wf = ricker(grid.dt, n_steps, peak_frequency=8000.0)
    center = ((n / 2) * dx, (n / 2) * dx, (n / 2) * dx)
    probe = np.array([[center[0] + 20e-3, center[1], center[2]]])
    sim = Simulation(
        grid=grid,
        sources=[PointSource(position=center, waveform=wf)],
        receivers=ReceiverArray(probe),
        n_steps=n_steps,
        sponge=SpongeConfig(thickness=30),
        device=_device(),
    )
    res = sim.run()
    t = res.times
    direct_window = t < 0.45e-3
    late_window = t > 0.7e-3
    direct = float(np.abs(res.p[direct_window, 0]).max())
    late = float(np.abs(res.p[late_window, 0]).max())
    assert 20.0 * np.log10(late / direct) < -40.0


@pytest.mark.gpu
def test_cpu_gpu_parity_small_grid() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    n = 48
    grid = Grid.create((n, n, n), dx=2e-3)
    n_steps = 80
    wf = ricker(grid.dt, n_steps, peak_frequency=8000.0)
    center = ((n / 2) * grid.dx,) * 3
    probes = np.array([[center[0] + 15e-3, center[1], center[2]]])

    results = []
    for device in ("cpu", "cuda"):
        sim = Simulation(
            grid=grid,
            sources=[PointSource(position=center, waveform=wf)],
            receivers=ReceiverArray(probes),
            n_steps=n_steps,
            sponge=SpongeConfig(thickness=10),
            device=device,
        )
        results.append(sim.run().p)
    np.testing.assert_allclose(results[0], results[1], atol=1e-6)
