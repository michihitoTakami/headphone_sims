"""Transparent-driver reference: aperture monopole sheet, solid-free scene."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

from headphone_sims.fdtd.sources import ApertureMonopoleSource
from headphone_sims.geometry.scene import DriverSpec, PinnaSpec, SceneConfig, build_scene
from headphone_sims.grid import Grid


def test_disc_aperture_cells_and_taper() -> None:
    grid = Grid.create((60, 60, 20), dx=1e-3)
    src = ApertureMonopoleSource(
        center=(30e-3, 30e-3, 5e-3),
        normal=(0.0, 0.0, 1.0),
        waveform=torch.zeros(8),
        shape="disc",
        radius=20e-3,
        taper_start=15e-3,
    )
    baked = src.bake(grid, torch.device("cpu"))
    assert baked.p_idx is not None and baked.p_weight is not None
    assert baked.v_idx is None  # pressure-only: nothing overwrites velocities
    n = int(baked.p_idx.numel())
    expected = math.pi * (20e-3 / 1e-3) ** 2
    assert abs(n - expected) / expected < 0.05
    w = baked.p_weight
    assert float(w.max()) == pytest.approx(1.0)
    assert float(w.min()) < 0.15  # cos^2 roll-off reaches near zero at the rim
    # inner disc (r < taper_start) stays at full amplitude
    assert int((w > 0.999).sum()) >= int(math.pi * (15e-3 / 1e-3) ** 2 * 0.9)


def test_rect_aperture_cell_count() -> None:
    grid = Grid.create((80, 100, 20), dx=1e-3)
    src = ApertureMonopoleSource(
        center=(40e-3, 50e-3, 5e-3),
        normal=(0.0, 0.0, 1.0),
        waveform=torch.zeros(8),
        shape="rect",
        width=30e-3,
        height=60e-3,
    )
    baked = src.bake(grid, torch.device("cpu"))
    assert baked.p_idx is not None
    n = int(baked.p_idx.numel())
    assert abs(n - 30 * 60) / (30 * 60) < 0.08
    assert baked.p_weight is not None and torch.all(baked.p_weight == 1.0)


def _synthetic_head(path: Path) -> str:
    skin = trimesh.creation.box(extents=[0.12, 0.004, 0.12])
    skin.apply_translation([0.0, -0.083, 0.0])
    bump = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    bump.apply_scale([0.015, 0.012, 0.025])
    bump.apply_translation([0.0, -0.085, 0.0])
    mesh = trimesh.util.concatenate([skin, bump])
    assert isinstance(mesh, trimesh.Trimesh)
    out = path / "head.ply"
    mesh.export(out)
    return str(out)


def test_transparent_scene_contains_only_the_pinna(tmp_path: Path) -> None:
    cfg = SceneConfig(
        dx=1e-3,
        distance=12e-3,
        driver=DriverSpec(diameter=40e-3),
        filters=(),
        baffle=False,
        transparent_driver=True,
        pinna=PinnaSpec(kind="mesh", mesh_path=_synthetic_head(tmp_path), side="left"),
        record_ms=0.1,
        n_probes=100,
    )
    built = build_scene(cfg, device="cpu")
    assert [name for name, _ in built.parts] == ["pinna"]
    assert bool((built.solid == dict(built.parts)["pinna"]).all())
    baked = built.simulation.baked_sources[0]
    assert baked.p_idx is not None and baked.v_idx is None
    # probes derived from the pinna exactly as in the structured scene
    assert len(built.probe_positions) > 30
    np.testing.assert_array_less(0, built.reference_index + 1)


def test_transparent_rejects_structure() -> None:
    with pytest.raises(ValueError, match="transparent_driver"):
        build_scene(SceneConfig(transparent_driver=True, baffle=True), device="cpu")
