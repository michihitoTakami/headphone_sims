"""Analytic validation of the viscous bore-loss model (issue #6 / B1).

Two independent checks:
1. The FDTD momentum-sink implementation: a plane-ish wave crossing a uniform
   sigma slab must attenuate by exp(-sigma L / (2 c)) — the small-loss plane
   wave solution of dv/dt = -grad p / rho - sigma v.
2. The Maa/Crandall resistance formula itself is unit-tested against its
   Poiseuille and boundary-layer limits in tests/unit/test_viscous.py.
Together they validate sigma = Phi_maa / rho as an equivalent-fluid bore loss.
"""

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
def test_sigma_slab_attenuation_matches_plane_wave_theory() -> None:
    n = 140
    dx = 2e-3
    grid = Grid.create((n, n, n), dx=dx)
    n_steps = 340
    wf = ricker(grid.dt, n_steps, peak_frequency=6000.0)
    center = ((n / 2) * dx, (n / 2) * dx, (n / 2) * dx)
    # Probes bracketing a sigma slab along +x, far enough from the source
    # that the wavefront is locally plane-ish over the slab.
    x_a, x_b = center[0] + 30e-3, center[0] + 100e-3
    probes = np.array([[x_a, center[1], center[2]], [x_b, center[1], center[2]]])
    slab_lo, slab_hi = center[0] + 40e-3, center[0] + 90e-3
    sigma_val = 4000.0  # 1/s, comfortably << omega
    xs = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    in_slab = (xs > slab_lo) & (xs <= slab_hi)
    sigma = torch.zeros(grid.shape, dtype=torch.float64)
    sigma[in_slab, :, :] = sigma_val
    length = float(in_slab.sum()) * dx  # realized slab thickness on the grid

    amps = {}
    for name, sm in (("lossless", None), ("lossy", sigma)):
        sim = Simulation(
            grid=grid,
            sources=[PointSource(position=center, waveform=wf)],
            receivers=ReceiverArray(probes),
            n_steps=n_steps,
            sigma_material=sm,
            sponge=SpongeConfig(thickness=25),
            device=_device(),
        )
        res = sim.run()
        amps[name] = (
            float(np.abs(res.p[:, 0]).max()),
            float(np.abs(res.p[:, 1]).max()),
        )

    # Transmission ratio relative to the lossless run cancels 1/r spreading.
    ratio = (amps["lossy"][1] / amps["lossy"][0]) / (amps["lossless"][1] / amps["lossless"][0])
    c = grid.medium.sound_speed
    expected = float(np.exp(-sigma_val * length / (2.0 * c)))
    assert expected < 0.85  # the test must exercise a non-trivial attenuation
    assert ratio == pytest.approx(expected, rel=0.05)
