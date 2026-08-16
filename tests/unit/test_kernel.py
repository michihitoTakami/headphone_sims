import torch

from headphone_sims.fdtd.boundaries import attach_damping
from headphone_sims.fdtd.kernel import FdtdState, field_energy, step
from headphone_sims.grid import Grid


def _excited_state(n: int = 32) -> FdtdState:
    grid = Grid.create((n, n, n), dx=2e-3)
    state = FdtdState.zeros(grid)
    # Smooth Gaussian pressure blob at the center (avoids single-cell checkerboarding).
    coords = [torch.arange(n, dtype=torch.float32) - (n - 1) / 2 for _ in range(3)]
    gx, gy, gz = torch.meshgrid(*coords, indexing="ij")
    state.p = torch.exp(-(gx**2 + gy**2 + gz**2) / (2.0 * 3.0**2))
    return state


def test_energy_bounded_in_sealed_box() -> None:
    # Leapfrog samples p and v half a step apart, so the naive energy sum
    # oscillates a few percent; stability means it stays bounded (no growth).
    state = _excited_state()
    e0 = field_energy(state)
    for it in range(1, 601):
        step(state)
        if it % 100 == 0:
            assert abs(field_energy(state) - e0) / e0 < 0.10


def test_symmetry_preserved() -> None:
    state = _excited_state()
    for _ in range(40):
        step(state)
    p = state.p
    for axis in range(3):
        assert torch.allclose(p, p.flip(axis), atol=1e-6)


def test_solid_face_masks_block_flow() -> None:
    n = 24
    grid = Grid.create((n, n, n), dx=2e-3)
    state = FdtdState.zeros(grid)
    # Solid wall splitting the domain at x = n//2; pressure blob on the left side.
    solid = torch.zeros(grid.shape, dtype=torch.bool)
    solid[n // 2, :, :] = True
    attach_damping(state, sponge=None, solid=solid)
    state.p[n // 4, n // 2, n // 2] = 1.0
    for _ in range(100):
        step(state)
    right = state.p[n // 2 + 1 :, :, :]
    assert float(right.abs().max()) == 0.0
    # Wave should still exist on the source side.
    assert float(state.p[: n // 2, :, :].abs().max()) > 0.0


def test_sponge_absorbs_energy() -> None:
    from headphone_sims.fdtd.boundaries import SpongeConfig

    state = _excited_state()
    attach_damping(state, sponge=SpongeConfig(thickness=8, sigma_max=60_000.0))
    e0 = field_energy(state)
    for _ in range(400):
        step(state)
    e1 = field_energy(state)
    assert e1 < 0.01 * e0
