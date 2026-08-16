"""Dome diaphragm: profile, voxelization, rigid-body source, scene wiring."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from headphone_sims.experiments.config import load_config
from headphone_sims.fdtd.boundaries import attach_damping
from headphone_sims.fdtd.kernel import FdtdState, step_velocity
from headphone_sims.fdtd.receivers import ReceiverArray
from headphone_sims.fdtd.simulation import Simulation
from headphone_sims.fdtd.sources import PistonSource, PointSource, RigidBodySource, ricker
from headphone_sims.geometry.parametric import DomeProfile, dome_solid, plate
from headphone_sims.geometry.scene import (
    DriverSpec,
    FilterSpec,
    PinnaSpec,
    SceneConfig,
    build_scene,
)
from headphone_sims.grid import Grid


def _dome_on_baffle(
    grid: Grid,
    center: tuple[float, float, float],
    normal: tuple[float, float, float],
    profile: DomeProfile,
    baffle_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(dome occupancy, full solid) with the standard baffle placement."""
    n = np.asarray(normal) / np.linalg.norm(normal)
    b_center = tuple(float(c) for c in np.asarray(center) - grid.dx * n)
    baffle, _ = plate(
        grid,
        (b_center[0], b_center[1], b_center[2]),
        normal,
        radius=baffle_radius,
        thickness=2 * grid.dx,
    )
    dome = dome_solid(grid, center, normal, profile, base_depth=grid.dx)
    return dome, baffle | dome


def test_profile_endpoints_and_roll_bump() -> None:
    p = DomeProfile(
        radius=20e-3, dome_fraction=0.85, dome_depth=5e-3, edge_height=1e-3, surround="roll"
    )
    r_d = 0.85 * 20e-3
    span = 20e-3 - r_d
    h = p.height(torch.tensor([0.0, r_d, 20e-3, 25e-3], dtype=torch.float64))
    assert float(h[0]) == pytest.approx(5e-3)
    assert float(h[1]) == pytest.approx(1e-3)
    assert float(h[2]) == pytest.approx(0.0, abs=1e-9)
    assert float(h[3]) == 0.0
    mid = p.height(torch.tensor([r_d + span / 2.0], dtype=torch.float64))
    assert float(mid[0]) == pytest.approx(1e-3 / 2.0 + span / 2.0, rel=1e-6)
    cone = DomeProfile(
        radius=20e-3, dome_fraction=0.85, dome_depth=5e-3, edge_height=1e-3, surround="cone"
    )
    mid_c = cone.height(torch.tensor([r_d + span / 2.0], dtype=torch.float64))
    assert float(mid_c[0]) == pytest.approx(0.5e-3, rel=1e-6)
    full = p.height(torch.linspace(0.0, 20e-3, 512, dtype=torch.float64))
    assert bool((full >= 0.0).all())


def test_profile_validation() -> None:
    with pytest.raises(ValueError, match="dome_fraction"):
        DomeProfile(radius=20e-3, dome_fraction=1.2)
    with pytest.raises(ValueError, match="edge_height"):
        DomeProfile(radius=20e-3, dome_depth=2e-3, edge_height=3e-3)
    with pytest.raises(ValueError, match="surround"):
        DomeProfile(radius=20e-3, surround="flat")


def test_dome_solid_volume_matches_analytic() -> None:
    grid = Grid.create((128, 128, 64), dx=0.5e-3)
    profile = DomeProfile(
        radius=20e-3, dome_fraction=0.85, dome_depth=5e-3, edge_height=1e-3, surround="roll"
    )
    center = (32e-3, 32e-3, 8e-3)
    occ = dome_solid(grid, center, (0.0, 0.0, 1.0), profile, base_depth=grid.dx)
    r = torch.linspace(0.0, profile.radius, 20001, dtype=torch.float64)
    v_profile = float(2.0 * math.pi * torch.trapezoid(r * profile.height(r), r))
    v_base = math.pi * profile.radius**2 * grid.dx
    expected = v_profile + v_base
    measured = int(occ.sum()) * grid.dx**3
    assert measured == pytest.approx(expected, rel=0.1)


def test_dome_face_count_equals_projected_disc_area() -> None:
    """Open boundary z-face count == projected pi R^2 / dx^2, independent of
    dome depth — the volume-velocity invariance vs the flat piston."""
    grid = Grid.create((64, 64, 64), dx=1e-3)
    center = (32e-3, 32e-3, 16.3e-3)
    radius = 15e-3
    counts = []
    for depth in (2e-3, 5e-3, 8e-3):
        profile = DomeProfile(
            radius=radius,
            dome_fraction=0.85,
            dome_depth=depth,
            edge_height=depth / 5.0,
            surround="roll",
        )
        dome, solid = _dome_on_baffle(grid, center, (0.0, 0.0, 1.0), profile, 25e-3)
        baked = RigidBodySource(
            occupancy=dome, solid=solid, direction=(0.0, 0.0, 1.0), waveform=torch.zeros(4)
        ).bake(grid, torch.device("cpu"))
        assert baked.v_idx is not None and baked.v_weight is not None
        assert baked.v_idx[0].numel() == 0
        assert baked.v_idx[1].numel() == 0
        assert torch.all(baked.v_weight[2] == 1.0)
        counts.append(baked.v_idx[2].numel())
    expected = math.pi * radius**2 / grid.dx**2
    assert counts[0] == pytest.approx(expected, rel=0.05)
    # The projected area does not depend on the profile at all.
    assert counts[0] == counts[1] == counts[2]


def test_zero_depth_dome_reduces_to_flat_piston() -> None:
    """A 1-layer flat body on the baffle selects exactly the flat hard-piston
    faces (regression anchor for dome_depth -> 0)."""
    grid = Grid.create((64, 64, 64), dx=1e-3)
    radius = 10e-3
    # One cell layer k=20 (centers on the half-lattice): cells z in (20, 21] mm.
    flat_body, _ = plate(
        grid, (32e-3, 32e-3, 20.5e-3), (0.0, 0.0, 1.0), radius=radius, thickness=grid.dx
    )
    baffle, _ = plate(
        grid, (32e-3, 32e-3, 19.5e-3), (0.0, 0.0, 1.0), radius=20e-3, thickness=2 * grid.dx
    )
    body_baked = RigidBodySource(
        occupancy=flat_body, solid=baffle, direction=(0.0, 0.0, 1.0), waveform=torch.zeros(2)
    ).bake(grid, torch.device("cpu"))
    piston_baked = PistonSource(
        center=(32e-3, 32e-3, 21e-3),
        normal=(0.0, 0.0, 1.0),
        radius=radius,
        waveform=torch.zeros(2),
    ).bake(grid, torch.device("cpu"))
    assert body_baked.v_idx is not None and piston_baked.v_idx is not None
    assert torch.equal(
        torch.sort(body_baked.v_idx[2]).values, torch.sort(piston_baked.v_idx[2]).values
    )


def test_tilted_dome_weights_are_axis_components() -> None:
    tau = math.radians(15.0)
    normal = (0.0, math.sin(tau), math.cos(tau))
    grid = Grid.create((80, 80, 80), dx=1e-3)
    profile = DomeProfile(
        radius=15e-3, dome_fraction=0.85, dome_depth=5e-3, edge_height=1e-3, surround="roll"
    )
    dome, solid = _dome_on_baffle(grid, (40e-3, 40e-3, 20e-3), normal, profile, 25e-3)
    baked = RigidBodySource(
        occupancy=dome, solid=solid, direction=normal, waveform=torch.zeros(4)
    ).bake(grid, torch.device("cpu"))
    assert baked.v_idx is not None and baked.v_weight is not None
    assert baked.v_idx[0].numel() == 0
    assert baked.v_idx[1].numel() > 0
    assert baked.v_idx[2].numel() > 0
    assert torch.allclose(baked.v_weight[1], torch.full_like(baked.v_weight[1], math.sin(tau)))
    assert torch.allclose(baked.v_weight[2], torch.full_like(baked.v_weight[2], math.cos(tau)))


def test_hard_dome_source_survives_solid_masks() -> None:
    grid = Grid.create((64, 64, 64), dx=1e-3)
    profile = DomeProfile(
        radius=15e-3, dome_fraction=0.85, dome_depth=5e-3, edge_height=1e-3, surround="roll"
    )
    dome, solid = _dome_on_baffle(grid, (32e-3, 32e-3, 16.3e-3), (0.0, 0.0, 1.0), profile, 25e-3)
    baked = RigidBodySource(
        occupancy=dome, solid=solid, direction=(0.0, 0.0, 1.0), waveform=torch.tensor([3.0])
    ).bake(grid, torch.device("cpu"))
    state = FdtdState.zeros(grid)
    attach_damping(state, sponge=None, solid=solid)
    state.vz.fill_(99.0)
    step_velocity(state)  # masks zero the boundary faces...
    v = (state.vx, state.vy, state.vz)
    baked.inject_velocity(v, 0)  # ...then the hard source re-drives them
    assert baked.v_idx is not None
    touched = state.vz.view(-1)[baked.v_idx[2]]
    assert torch.allclose(touched, torch.full_like(touched, 3.0))
    # Past the waveform the body clamps to a rigid scatterer.
    baked.inject_velocity(v, 5)
    touched = state.vz.view(-1)[baked.v_idx[2]]
    assert torch.allclose(touched, torch.zeros_like(touched))


def test_dome_on_baffle_is_airtight() -> None:
    """No leak through the filled dome + baffle: a source in front must not
    reach a probe behind the baffle."""
    grid = Grid.create((60, 60, 60), dx=1e-3)
    profile = DomeProfile(
        radius=15e-3, dome_fraction=0.85, dome_depth=5e-3, edge_height=1e-3, surround="roll"
    )
    _, solid = _dome_on_baffle(
        grid, (30e-3, 30e-3, 20.3e-3), (0.0, 0.0, 1.0), profile, 60e-3
    )  # baffle seals wall to wall
    n_steps = 250
    wf = ricker(grid.dt, n_steps, peak_frequency=10_000.0)
    sim = Simulation(
        grid=grid,
        sources=[PointSource(position=(30e-3, 30e-3, 40e-3), waveform=wf)],
        receivers=ReceiverArray(np.array([[30e-3, 30e-3, 8e-3]])),
        n_steps=n_steps,
        solid=solid,
        sponge=None,
        device="cpu",
    )
    res = sim.run()
    assert float(np.abs(res.p).max()) == 0.0


def test_dome_yaml_loads_and_rejects_typos(tmp_path: Path) -> None:
    good = tmp_path / "dome.yaml"
    good.write_text(
        """
name: dome_test
scene:
  dx: 1.0e-3
  driver:
    diameter: 40.0e-3
    shape: dome
    dome_fraction: 0.85
    dome_depth: "5.0e-3"
    edge_height: 1.0e-3
    surround: roll
  pinna: { kind: none }
""",
        encoding="utf-8",
    )
    cfg = load_config(good)
    assert cfg.scene.driver.shape == "dome"
    assert cfg.scene.driver.dome_depth == pytest.approx(5e-3)  # str coerced to float
    assert cfg.scene.driver.surround == "roll"
    bad = tmp_path / "typo.yaml"
    bad.write_text(
        "name: t\nscene: { driver: { shape: dome, dome_dept: 5.0e-3 } }\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown DriverSpec keys"):
        load_config(bad)


def _small_scene(driver: DriverSpec, **overrides: object) -> SceneConfig:
    defaults: dict[str, object] = {
        "dx": 1e-3,
        "driver": driver,
        "pinna": PinnaSpec(kind="none"),
        "lateral_margin": 2e-3,
        "axial_margin": 5e-3,
        "sponge_thickness": 8,
        "n_probes": 16,
        "record_ms": 0.1,
    }
    defaults.update(overrides)
    return SceneConfig(**defaults)  # type: ignore[arg-type]


def test_filter_standoff_must_clear_dome() -> None:
    config = _small_scene(
        DriverSpec(diameter=40e-3, shape="dome", dome_depth=5e-3),
        filters=(FilterSpec(kind="hex", standoff=3e-3, thickness=0.5e-3),),
    )
    with pytest.raises(ValueError, match="intersects the dome"):
        build_scene(config, device="cpu")


def test_dome_triggers_pinna_collision_guard() -> None:
    config = _small_scene(
        DriverSpec(diameter=40e-3, shape="dome", dome_depth=5e-3),
        pinna=PinnaSpec(kind="parametric"),
        distance=6e-3,
        baffle=False,
    )
    with pytest.raises(ValueError, match="intersects the pinna"):
        build_scene(config, device="cpu")


def test_dome_scene_builds_and_registers_driver_part() -> None:
    config = _small_scene(
        DriverSpec(diameter=40e-3, shape="dome", dome_depth=5e-3),
        filters=(FilterSpec(kind="hex", standoff=7e-3, thickness=0.5e-3),),
    )
    built = build_scene(config, device="cpu")
    names = [name for name, _ in built.parts]
    assert names[0] == "driver"
    assert "baffle" in names
    driver_occ = dict(built.parts)["driver"]
    assert int(driver_occ.sum()) > 0


def test_dome_only_drive_restricts_faces() -> None:
    """dome_drive="dome" drives fewer faces, all within the dome rim radius."""
    import dataclasses

    from headphone_sims.experiments.config import load_config

    base = load_config("configs/hutubs_70mm_z1r_dome.yaml").scene
    small = dataclasses.replace(
        base,
        dx=1.0e-3,
        record_ms=0.1,
        sponge_thickness=8,
        lateral_margin=6e-3,
        axial_margin=8e-3,
        n_probes=20,
        pinna=dataclasses.replace(base.pinna, kind="none", mesh_path=None),
    )
    full = build_scene(small, device="cpu")
    dome_only = build_scene(
        dataclasses.replace(small, driver=dataclasses.replace(small.driver, dome_drive="dome")),
        device="cpu",
    )
    n_full = sum(i.numel() for i in full.simulation.baked_sources[0].v_idx or ())
    n_dome = sum(i.numel() for i in dome_only.simulation.baked_sources[0].v_idx or ())
    assert 0 < n_dome < n_full
    # Roughly the dome's projected-area share of the total driven z-faces.
    assert n_dome < 0.5 * n_full


def test_tapered_drive_amplitude_profile() -> None:
    """Tapered drive: full amplitude on the dome, monotonic decay on the
    surround, ~0 at the rim, and volume velocity between dome-only and full."""
    import dataclasses

    import numpy as np

    from headphone_sims.experiments.config import load_config

    base = load_config("configs/hutubs_70mm_z1r_dome.yaml").scene
    small = dataclasses.replace(
        base,
        dx=1.0e-3,
        record_ms=0.1,
        sponge_thickness=8,
        lateral_margin=6e-3,
        axial_margin=8e-3,
        n_probes=20,
        pinna=dataclasses.replace(base.pinna, kind="none", mesh_path=None),
    )

    from typing import Literal

    from headphone_sims.fdtd.sources import BakedSource
    from headphone_sims.geometry.scene import BuiltScene

    def baked(drive: 'Literal["tapered", "full", "dome"]') -> tuple[BuiltScene, BakedSource]:
        scene = dataclasses.replace(
            small, driver=dataclasses.replace(small.driver, dome_drive=drive)
        )
        built = build_scene(scene, device="cpu")
        return built, built.simulation.baked_sources[0]

    built, src = baked("tapered")
    assert src.v_idx is not None and src.v_weight is not None
    # z-face weights vs lateral radius of the face.
    grid = built.grid
    idx = src.v_idx[2].numpy()
    w = src.v_weight[2].numpy()
    shape = grid.velocity_shape(2)
    j = (idx // shape[2]) % shape[1]
    i = idx // (shape[1] * shape[2])
    x = (i + 0.5) * grid.dx
    y = (j + 0.5) * grid.dx
    dc = np.asarray(built.driver_center)
    lat = np.hypot(x - dc[0], y - dc[1])
    r_dome = small.driver.dome_fraction * small.driver.diameter / 2.0
    on_dome = lat <= r_dome - 1e-3
    near_rim = lat >= small.driver.diameter / 2.0 - 2e-3
    assert w[on_dome].min() > 0.9  # full amplitude on the dome
    assert w[near_rim].max() < 0.2  # ~zero at the clamped rim
    mid = (lat > r_dome + 2e-3) & (lat < small.driver.diameter / 2.0 - 4e-3)
    assert 0.05 < w[mid].mean() < 0.95  # smooth taper in between

    _, src_dome = baked("dome")
    _, src_full = baked("full")

    def vv(s: BakedSource) -> float:
        return float(sum(wi.abs().sum() for wi in (s.v_weight or ())))

    assert vv(src_dome) < vv(src) < vv(src_full)


def test_roll_height_makes_donut_edge() -> None:
    from headphone_sims.geometry.parametric import DomeProfile

    p = DomeProfile(
        radius=35e-3,
        dome_fraction=0.43,
        dome_depth=6e-3,
        edge_height=3e-3,
        surround="roll",
        roll_height=3e-3,
    )
    r = torch.tensor([0.0, 15e-3, 25e-3, 34.9e-3])
    h = p.height(r)
    assert float(h[0]) == pytest.approx(6e-3, abs=1e-4)  # dome apex
    crest = float(h[2])
    assert crest > float(h[1]) - 1e-4 or crest > 4e-3  # rounded bump mid-edge
    assert 4e-3 < crest < 5e-3
    assert float(h[3]) < 1e-3  # falls to ~0 at the rim
    # Backward compatibility: None keeps legacy amplitudes.
    legacy = DomeProfile(
        radius=35e-3, dome_fraction=0.43, dome_depth=6e-3, edge_height=3e-3, surround="cone"
    )
    assert float(legacy.height(torch.tensor([25e-3]))[0]) < 2e-3
