import numpy as np
import torch

from headphone_sims.fdtd.kernel import FdtdState
from headphone_sims.fdtd.receivers import ReceiverArray
from headphone_sims.grid import Grid


def test_trilinear_exact_on_linear_pressure_field() -> None:
    grid = Grid.create((16, 16, 16), dx=1e-3)
    state = FdtdState.zeros(grid)
    # p(x, y, z) = 2x + 3y + 5z (in meters) sampled at cell centers.
    idx = [torch.arange(n, dtype=torch.float32) for n in grid.shape]
    gx, gy, gz = torch.meshgrid(*idx, indexing="ij")
    dx = grid.dx
    state.p = 2.0 * (gx + 0.5) * dx + 3.0 * (gy + 0.5) * dx + 5.0 * (gz + 0.5) * dx

    positions = np.array([[5.3e-3, 7.9e-3, 4.4e-3], [8.0e-3, 8.0e-3, 8.0e-3]])
    baked = ReceiverArray(positions).bake(grid, n_steps=1, device=torch.device("cpu"))
    baked.sample(state.p, (state.vx, state.vy, state.vz), 0)

    expected = 2.0 * positions[:, 0] + 3.0 * positions[:, 1] + 5.0 * positions[:, 2]
    np.testing.assert_allclose(baked.rec_p[0].numpy(), expected, rtol=1e-5)


def test_trilinear_exact_on_linear_velocity_field() -> None:
    grid = Grid.create((16, 16, 16), dx=1e-3)
    state = FdtdState.zeros(grid)
    # vx(x) = 10x sampled on x-faces at ((i+1)*dx, ...).
    nx = grid.velocity_shape(0)[0]
    xs = (torch.arange(nx, dtype=torch.float32) + 1.0) * grid.dx
    state.vx = xs.view(-1, 1, 1).expand(grid.velocity_shape(0)).contiguous() * 10.0

    positions = np.array([[6.7e-3, 7.5e-3, 7.5e-3]])
    baked = ReceiverArray(positions).bake(grid, n_steps=1, device=torch.device("cpu"))
    baked.sample(state.p, (state.vx, state.vy, state.vz), 0)
    np.testing.assert_allclose(baked.rec_v[0][0].numpy(), [10.0 * 6.7e-3], rtol=1e-5)


def test_positions_shape_validated() -> None:
    grid = Grid.create((8, 8, 8), dx=1e-3)
    import pytest

    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        ReceiverArray(np.zeros((3, 2))).bake(grid, 1, torch.device("cpu"))
