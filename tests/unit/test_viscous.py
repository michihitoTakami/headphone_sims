"""Viscous bore-loss model: Maa resistance, bore masks, scene wiring (B1)."""

import dataclasses
import math

import pytest

from headphone_sims.geometry import parametric
from headphone_sims.geometry.parametric import AmtsHexCells, HexHoles
from headphone_sims.geometry.scene import (
    DriverSpec,
    FilterSpec,
    PinnaSpec,
    SceneConfig,
    build_scene,
)
from headphone_sims.grid import Grid
from headphone_sims.medium import AIR, maa_tube_flow_resistivity


def test_maa_resistivity_poiseuille_limit() -> None:
    # s << 1: Phi -> 8 eta / a^2 (steady Poiseuille).
    a = 20e-6
    phi = maa_tube_flow_resistivity(a, frequency=10.0)
    assert phi == pytest.approx(8.0 * AIR.dynamic_viscosity / a**2, rel=0.01)


def test_maa_resistivity_boundary_layer_limit() -> None:
    # s >> 1: Phi -> sqrt(2 eta rho omega) / a. The AMTS bore (a = 1.1 mm) at
    # 7 kHz sits at s ~ 60, firmly in this regime.
    a = 1.1e-3
    f = 7000.0
    omega = 2.0 * math.pi * f
    s = a * math.sqrt(AIR.density * omega / AIR.dynamic_viscosity)
    assert s > 30.0
    phi = maa_tube_flow_resistivity(a, f)
    bl = math.sqrt(2.0 * AIR.dynamic_viscosity * AIR.density * omega) / a
    assert phi == pytest.approx(bl, rel=0.05)


def test_flat_plate_bore_equals_open_slab_cells() -> None:
    grid = Grid.create((60, 60, 30), dx=0.5e-3)
    kwargs = dict(
        center=(15e-3, 15e-3, 7e-3),
        normal=(0.0, 0.0, 1.0),
        radius=10e-3,
        thickness=1e-3,
        pattern=HexHoles(hole_radius=0.5e-3, pitch=2e-3),
    )
    occ, porosity = parametric.plate(grid, **kwargs)  # type: ignore[arg-type]
    bore = parametric.plate_bore(grid, **kwargs)  # type: ignore[arg-type]
    assert 0.1 < porosity < 0.5
    assert bool((bore & occ).any()) is False  # bores are air, never solid
    # For a flat plate the slab is solid + bore exactly.
    n_slab = int(occ.sum()) + int(bore.sum())
    assert int(bore.sum()) == pytest.approx(n_slab * porosity, rel=0.05)


def test_amts_bore_excludes_air_above_slope() -> None:
    grid = Grid.create((80, 120, 60), dx=0.5e-3)
    pat = AmtsHexCells(
        tube_radius=1.1e-3,
        pitch=3.2e-3,
        slope_length=40e-3,
        min_thickness=3e-3,
        max_thickness=15e-3,
        plug_cells=((0, 0),),
        plug_wall=1e-3,
    )
    kwargs = dict(
        center=(20e-3, 30e-3, 15e-3),
        normal=(0.0, 0.0, 1.0),
        width=30e-3,
        height=40e-3,
        thickness=15e-3,
        pattern=pat,
    )
    occ, _ = parametric.rect_plate(grid, **kwargs)  # type: ignore[arg-type]
    bore = parametric.rect_plate_bore(grid, **kwargs)  # type: ignore[arg-type]
    assert bool((bore & occ).any()) is False
    # The slab holds solid + bores + carved-away free air above the slope;
    # bores must be a small fraction of the slab air, not all of it.
    slab_box, _ = parametric.rect_plate(
        grid,
        **{**kwargs, "pattern": None},  # type: ignore[arg-type]
    )
    n_air_in_box = int((slab_box & ~occ).sum())
    assert 0 < int(bore.sum()) < 0.5 * n_air_in_box
    # Every bore cell has solid nearby at the same z (it is inside a tube):
    idx = bore.nonzero(as_tuple=False)
    sample = idx[:: max(1, len(idx) // 50)]
    for i, j, k in sample.tolist():
        lo_i, hi_i = max(i - 4, 0), min(i + 5, grid.shape[0])
        lo_j, hi_j = max(j - 4, 0), min(j + 5, grid.shape[1])
        assert bool(occ[lo_i:hi_i, lo_j:hi_j, k].any())


def test_scene_viscous_filter_damps_bore_faces() -> None:
    cfg = SceneConfig(
        dx=1e-3,
        distance=15e-3,
        driver=DriverSpec(diameter=20e-3),
        filters=(
            FilterSpec(
                kind="hex",
                hole_radius=1.5e-3,
                pitch=5e-3,
                standoff=5e-3,
                thickness=2e-3,
                sealed=False,
                viscous_losses=True,
            ),
        ),
        pinna=PinnaSpec(kind="none"),
        baffle=True,
        record_ms=0.05,
        n_probes=16,
    )
    built = build_scene(cfg, device="cpu")
    mult = built.simulation.state.mult_vz
    assert mult is not None
    # Interior faces away from the sponge are exactly 1.0 (lossless) or 0.0
    # (solid mask) except inside the bores, where 0 < mult < 1.
    sp = cfg.sponge_thickness + 2
    inner = mult[sp:-sp, sp:-sp, sp:-sp]
    partial = (inner > 0.0) & (inner < 1.0)
    assert bool(partial.any())
    vals = inner[partial]
    # sigma ~ 8.5e2 1/s at a=1.5mm/7kHz, dt~1.5us -> mult ~ exp(-1.3e-3)
    assert float(vals.min()) > 0.99
    assert float(vals.max()) < 1.0

    # Rigid control: no partial damping anywhere in the interior.
    cfg_rigid = SceneConfig(
        dx=cfg.dx,
        distance=cfg.distance,
        driver=cfg.driver,
        filters=(dataclasses.replace(cfg.filters[0], viscous_losses=False),),
        pinna=cfg.pinna,
        baffle=True,
        record_ms=cfg.record_ms,
        n_probes=cfg.n_probes,
    )
    built_rigid = build_scene(cfg_rigid, device="cpu")
    mult_r = built_rigid.simulation.state.mult_vz
    assert mult_r is not None
    inner_r = mult_r[sp:-sp, sp:-sp, sp:-sp]
    assert not bool(((inner_r > 0.0) & (inner_r < 1.0)).any())


def test_viscous_losses_rejected_for_unsupported_kinds() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        FilterSpec(kind="fib", viscous_losses=True).viscous_radius()
    cfg = SceneConfig(
        driver=DriverSpec(diameter=20e-3),
        filters=(FilterSpec(kind="fib", viscous_losses=True, sealed=False),),
        pinna=PinnaSpec(kind="none"),
        record_ms=0.02,
    )
    with pytest.raises(ValueError, match="unsupported"):
        build_scene(cfg, device="cpu")
