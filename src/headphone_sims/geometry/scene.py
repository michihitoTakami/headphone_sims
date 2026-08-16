"""High-level headphone scene assembly.

Scene coordinate convention: the driver piston sits near the low-z face and
radiates toward +z; the pinna faces it (outward normal toward -z). The x/y
domain is sized from the largest part, plus wave-travel margin and the sponge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch

from headphone_sims.fdtd.boundaries import SpongeConfig
from headphone_sims.fdtd.receivers import ReceiverArray
from headphone_sims.fdtd.simulation import Simulation, SnapshotConfig
from headphone_sims.fdtd.sources import PistonSource, Source, gaussian_modulated_sine
from headphone_sims.geometry import parametric
from headphone_sims.geometry.mesh import load_mesh, surface_probes, voxelize
from headphone_sims.geometry.parametric import HexHoles, HolePattern, RingSlits, Slots
from headphone_sims.geometry.pinna import extract_ear_region, orient_pinna_to_scene
from headphone_sims.grid import Grid

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class DriverSpec:
    diameter: float = 40e-3
    inner_diameter: float = 0.0  # >0 for an annular (planar-magnetic ring) source
    tilt_deg: float = 0.0  # rotation about the x axis; positive aims toward +y
    center_frequency: float = 10_000.0
    bandwidth_frequency: float = 9_900.0


@dataclass(frozen=True)
class FilterSpec:
    """Perforated plate between driver and ear."""

    kind: Literal["hex", "slots", "rings", "solid"]
    standoff: float = 3e-3  # distance from driver plane toward the ear
    thickness: float = 1e-3
    radius: float | None = None  # default: driver radius + 2 mm
    hole_radius: float = 0.5e-3  # hex
    pitch: float = 2e-3  # hex, slots
    slot_width: float = 1e-3  # slots
    rings: tuple[tuple[float, float], ...] = ()  # ring slits (r_in, r_out)

    def pattern(self) -> HolePattern | None:
        if self.kind == "hex":
            return HexHoles(hole_radius=self.hole_radius, pitch=self.pitch)
        if self.kind == "slots":
            return Slots(width=self.slot_width, pitch=self.pitch)
        if self.kind == "rings":
            return RingSlits(rings=self.rings)
        return None  # solid


@dataclass(frozen=True)
class PinnaSpec:
    kind: Literal["mesh", "parametric", "none"] = "parametric"
    mesh_path: str | None = None
    side: Literal["left", "right"] = "left"
    lateral_axis: int = 0
    scale: float = 1.0  # e.g. 1e-3 if the mesh is in millimeters
    extra_rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    region_size: float = 90e-3


@dataclass(frozen=True)
class SceneConfig:
    dx: float = 0.5e-3
    driver: DriverSpec = DriverSpec()
    filters: tuple[FilterSpec, ...] = ()
    pinna: PinnaSpec = PinnaSpec()
    distance: float = 15e-3  # driver plane to pinna reference plane
    baffle: bool = True
    baffle_radius: float | None = None  # default: driver radius + 10 mm
    cup_depth: float = 0.0  # >0 adds a cylindrical cup behind the driver
    lateral_margin: float = 15e-3
    axial_margin: float = 20e-3
    sponge_thickness: int = 30
    n_probes: int = 400
    probe_offset: float = 1.5e-3
    record_ms: float = 2.0
    snapshot_every: int = 0  # 0 = no snapshots


@dataclass
class BuiltScene:
    """Everything needed to run and analyze one scene."""

    grid: Grid
    simulation: Simulation
    driver_center: Vec3
    probe_positions: npt.NDArray[np.float64]
    reference_index: int  # probe used as the ear-canal-entrance reference
    porosities: list[float] = field(default_factory=list)


def _domain_grid(config: SceneConfig) -> tuple[Grid, Vec3, float]:
    """Grid, driver center, and pinna reference z."""
    r_driver = config.driver.diameter / 2.0
    r_baffle = config.baffle_radius or (r_driver + 10e-3)
    r_pinna = 45e-3 if config.pinna.kind != "none" else 30e-3
    r_lateral = max(r_baffle, r_pinna) + config.lateral_margin
    sponge = config.sponge_thickness * config.dx

    z_driver = config.axial_margin + sponge
    z_pinna = z_driver + config.distance
    pinna_depth = 45e-3 if config.pinna.kind == "mesh" else 25e-3
    z_max = z_pinna + pinna_depth + config.axial_margin + sponge

    nx = int(np.ceil(2.0 * (r_lateral + sponge) / config.dx))
    nz = int(np.ceil(z_max / config.dx))
    shape = (nx, nx, nz)
    grid = Grid.create(shape, dx=config.dx)
    cx = nx * config.dx / 2.0
    return grid, (cx, cx, z_driver), z_pinna


def build_scene(config: SceneConfig, device: str | None = None) -> BuiltScene:
    grid, driver_center, z_pinna = _domain_grid(config)
    n_steps = int(np.ceil(config.record_ms * 1e-3 / grid.dt))
    r_driver = config.driver.diameter / 2.0

    tilt = np.deg2rad(config.driver.tilt_deg)
    normal = (0.0, float(np.sin(tilt)), float(np.cos(tilt)))

    waveform = gaussian_modulated_sine(
        grid.dt,
        n_steps,
        center_frequency=config.driver.center_frequency,
        bandwidth_frequency=config.driver.bandwidth_frequency,
    )
    sources: list[Source] = [
        PistonSource(
            center=driver_center,
            normal=normal,
            radius=r_driver,
            inner_radius=config.driver.inner_diameter / 2.0,
            waveform=waveform,
        )
    ]

    solid = torch.zeros(grid.shape, dtype=torch.bool)
    porosities: list[float] = []

    # The baffle, cup, and filters tilt together with the driver (one assembly).
    # The baffle sits far enough behind the source plane that its face masks
    # cannot swallow the injected piston velocity (injection happens on open
    # faces only).
    nvec = np.asarray(normal)

    def along_normal(base: Vec3, offset: float) -> Vec3:
        p = np.asarray(base) + offset * nvec
        return (float(p[0]), float(p[1]), float(p[2]))

    if config.baffle:
        # Wall front surface on the driver plane: the hard piston faces are
        # then the wall's own boundary faces (piston-on-rigid-baffle model).
        r_baffle = config.baffle_radius or (r_driver + 10e-3)
        occ, _ = parametric.plate(
            grid,
            along_normal(driver_center, -1.0 * config.dx),
            normal,
            radius=r_baffle,
            thickness=2 * config.dx,
        )
        solid |= occ

    if config.cup_depth > 0.0:
        r_baffle = config.baffle_radius or (r_driver + 10e-3)
        solid |= parametric.cup_shell(
            grid,
            center=along_normal(driver_center, -1.0 * config.dx),
            axis=(-normal[0], -normal[1], -normal[2]),
            inner_radius=r_baffle,
            depth=config.cup_depth,
            thickness=2e-3,
        )

    for spec in config.filters:
        r_filter = spec.radius or (r_driver + 2e-3)
        occ, porosity = parametric.plate(
            grid,
            along_normal(driver_center, spec.standoff),
            normal,
            radius=r_filter,
            thickness=spec.thickness,
            pattern=spec.pattern(),
        )
        solid |= occ
        porosities.append(porosity)

    pinna_center = (driver_center[0], driver_center[1], z_pinna)
    probes: npt.NDArray[np.float64]
    if config.pinna.kind == "mesh":
        if config.pinna.mesh_path is None:
            raise ValueError("pinna.kind='mesh' requires mesh_path")
        head = load_mesh(Path(config.pinna.mesh_path))
        if config.pinna.scale != 1.0:
            head.apply_scale(config.pinna.scale)
        ear_mesh, ear_center = extract_ear_region(
            head,
            side=config.pinna.side,
            box_size=config.pinna.region_size,
            lateral_axis=config.pinna.lateral_axis,
        )
        placed = orient_pinna_to_scene(
            ear_mesh,
            ear_center,
            pinna_center,
            side=config.pinna.side,
            lateral_axis=config.pinna.lateral_axis,
            extra_rotation_deg=config.pinna.extra_rotation_deg,
        )
        solid |= voxelize(placed, grid)
        probes = surface_probes(
            placed,
            config.n_probes,
            offset=config.probe_offset,
            direction=(0.0, 0.0, -1.0),
        )
    elif config.pinna.kind == "parametric":
        from headphone_sims.geometry.pinna import parametric_pinna

        solid |= parametric_pinna(grid, pinna_center)
        probes = _disc_probes(pinna_center, radius=30e-3, n=config.n_probes)
    else:
        probes = _disc_probes(pinna_center, radius=30e-3, n=config.n_probes)

    # Reference probe: closest to the nominal ear-canal entrance (pinna center).
    ref = int(np.argmin(np.linalg.norm(probes - np.asarray(pinna_center), axis=1)))

    snapshot = (
        SnapshotConfig(every=config.snapshot_every, axis=0) if config.snapshot_every > 0 else None
    )
    sim = Simulation(
        grid=grid,
        sources=sources,
        receivers=ReceiverArray(probes),
        n_steps=n_steps,
        solid=solid,
        sponge=SpongeConfig(thickness=config.sponge_thickness),
        device=device,
        snapshot=snapshot,
    )
    return BuiltScene(
        grid=grid,
        simulation=sim,
        driver_center=driver_center,
        probe_positions=probes,
        reference_index=ref,
        porosities=porosities,
    )


def _disc_probes(center: Vec3, radius: float, n: int) -> npt.NDArray[np.float64]:
    """Sunflower-spiral probe layout on the pinna reference plane."""
    k = np.arange(n, dtype=np.float64) + 0.5
    r = radius * np.sqrt(k / n)
    theta = k * 2.399963229728653  # golden angle
    pts = np.zeros((n, 3))
    pts[:, 0] = center[0] + r * np.cos(theta)
    pts[:, 1] = center[1] + r * np.sin(theta)
    pts[:, 2] = center[2]
    return pts
