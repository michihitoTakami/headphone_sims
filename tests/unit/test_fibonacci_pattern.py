import pytest
import torch

from headphone_sims.geometry.parametric import FibonacciSpirals, plate
from headphone_sims.grid import Grid


def test_fibonacci_plate_porosity_and_structure() -> None:
    grid = Grid.create((160, 160, 24), dx=0.5e-3)
    occ, porosity = plate(
        grid,
        (40e-3, 40e-3, 6e-3),
        (0.0, 0.0, 1.0),
        radius=35e-3,
        thickness=0.5e-3,
        pattern=FibonacciSpirals(rib_width=1.2e-3, winding=1.2),
    )
    assert 0.55 < porosity < 0.85
    assert int(occ.sum()) > 0


def test_hub_is_open_and_ribs_are_solid() -> None:
    pat = FibonacciSpirals(rib_width=1.2e-3, winding=1.2, hub_radius=2.5e-3)
    a = torch.tensor([0.0, 1.0e-3])
    b = torch.tensor([0.0, 0.0])
    open_m = pat.open_mask(a, b)
    assert bool(open_m[0]) and bool(open_m[1])  # hub region open
    # A dense ring at r=20mm must contain both rib (solid) and cell (open) points.
    theta = torch.linspace(0, 2 * torch.pi, 2000)
    ring_open = pat.open_mask(20e-3 * torch.cos(theta), 20e-3 * torch.sin(theta))
    frac_open = float(ring_open.float().mean())
    assert 0.4 < frac_open < 0.95


def test_spiral_counts_change_pattern() -> None:
    theta = torch.linspace(0, 2 * torch.pi, 4000)
    a, b = 30e-3 * torch.cos(theta), 30e-3 * torch.sin(theta)
    open_8_13 = FibonacciSpirals(rib_width=1.0e-3).open_mask(a, b)
    open_5_8 = FibonacciSpirals(rib_width=1.0e-3, m_cw=5, m_ccw=8).open_mask(a, b)
    # Fewer spirals -> fewer rib crossings on the ring -> more open.
    assert float(open_5_8.float().mean()) > float(open_8_13.float().mean())
    # Number of solid segments on the ring ~ m_cw + m_ccw.
    solid = ~open_8_13
    transitions = int((solid[1:] & ~solid[:-1]).sum())
    assert transitions == pytest.approx(8 + 13, abs=3)
