"""Rectangular diaphragm, plate, and collar for planar-magnetic scenes."""

import numpy as np
import pytest
import torch

from headphone_sims.fdtd.sources import RectangularPistonSource
from headphone_sims.geometry.parametric import Slots, rect_collar, rect_plate
from headphone_sims.geometry.scene import (
    DriverSpec,
    FilterSpec,
    PinnaSpec,
    SceneConfig,
    build_scene,
)
from headphone_sims.grid import Grid


def test_rect_piston_face_count() -> None:
    grid = Grid.create((80, 100, 40), dx=1e-3)
    src = RectangularPistonSource(
        center=(40e-3, 50e-3, 20e-3),
        normal=(0.0, 0.0, 1.0),
        width=30e-3,
        height=60e-3,
        waveform=torch.zeros(4),
    )
    baked = src.bake(grid, torch.device("cpu"))
    assert baked.v_idx is not None
    n_faces = baked.v_idx[2].numel()
    assert n_faces == pytest.approx(30e-3 * 60e-3 / grid.dx**2, rel=0.05)
    assert baked.v_idx[0].numel() == 0 and baked.v_idx[1].numel() == 0


def test_rect_slot_plate_porosity() -> None:
    grid = Grid.create((160, 220, 40), dx=0.5e-3)
    occ, porosity = rect_plate(
        grid,
        (40e-3, 55e-3, 10e-3),
        (0.0, 0.0, 1.0),
        width=65e-3,
        height=90e-3,
        thickness=3e-3,
        pattern=Slots(width=3e-3, pitch=7e-3),
        pattern_angle_deg=90.0,
    )
    assert porosity == pytest.approx(3.0 / 7.0, abs=0.05)
    # Bars run along y: occupancy varies along x, constant along y (interior).
    sl = occ[:, 60:160, 20]  # z slice through the plate mid-plane (10mm / 0.5mm)
    assert int(occ.sum()) > 0
    x_profile = sl.any(dim=1)
    y_profile = sl.any(dim=0)
    assert bool(y_profile.all())  # every y row crosses some bar
    assert not bool(x_profile.all())  # gaps exist along x


def test_sealed_rect_solid_plate_is_airtight() -> None:
    config = SceneConfig(
        dx=1.0e-3,
        distance=14e-3,
        record_ms=0.3,
        sponge_thickness=8,
        lateral_margin=6e-3,
        axial_margin=8e-3,
        n_probes=30,
        driver=DriverSpec(shape="rect", width=30e-3, height=40e-3),
        filters=(
            FilterSpec(
                kind="solid",
                shape="rect",
                width=30e-3,
                height=40e-3,
                standoff=4e-3,
                thickness=2e-3,
            ),
        ),
        pinna=PinnaSpec(kind="none"),
    )
    built = build_scene(config, device="cpu")
    result = built.simulation.run()
    assert float(np.abs(result.p).max()) == 0.0


def test_rect_collar_is_hollow_frame() -> None:
    grid = Grid.create((80, 100, 40), dx=1e-3)
    occ = rect_collar(
        grid,
        (40e-3, 50e-3, 10e-3),
        (0.0, 0.0, 1.0),
        width=30e-3,
        height=50e-3,
        length=5e-3,
        thickness=2e-3,
    )
    assert int(occ.sum()) > 0
    # Interior stays open.
    assert not bool(occ[40, 50, 12])
