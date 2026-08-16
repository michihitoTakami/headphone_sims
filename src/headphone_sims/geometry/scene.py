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
from headphone_sims.fdtd.sources import (
    PistonSource,
    RectangularPistonSource,
    Source,
    gaussian_modulated_sine,
)
from headphone_sims.geometry import parametric
from headphone_sims.geometry.mesh import load_mesh, voxelize
from headphone_sims.geometry.parametric import (
    ChamferedSlots,
    FibonacciSpirals,
    HexHoles,
    HolePattern,
    RingSlits,
    Slots,
)
from headphone_sims.geometry.pinna import (
    extract_ear_region,
    find_ear_canal_entrance,
    orient_pinna_to_scene,
    select_pinna_probes,
)
from headphone_sims.grid import Grid

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class DriverSpec:
    diameter: float = 40e-3
    inner_diameter: float = 0.0  # >0 for an annular (planar-magnetic ring) source
    tilt_deg: float = 0.0  # rotation about the x axis; positive aims toward +y
    center_frequency: float = 10_000.0
    bandwidth_frequency: float = 9_900.0
    # Rectangular (planar-magnetic) diaphragm: shape="rect" uses width x height
    # (height runs along scene y, i.e. along the pinna) and ignores diameter.
    shape: Literal["disc", "rect"] = "disc"
    width: float = 65e-3
    height: float = 90e-3

    def half_extents(self) -> tuple[float, float]:
        """Lateral half extents (x, y) of the radiating surface."""
        if self.shape == "rect":
            return (self.width / 2.0, self.height / 2.0)
        return (self.diameter / 2.0, self.diameter / 2.0)


@dataclass(frozen=True)
class FilterSpec:
    """Perforated plate between driver and ear."""

    kind: Literal["hex", "slots", "rings", "fib", "solid"]
    standoff: float = 3e-3  # distance from driver plane toward the ear
    thickness: float = 1e-3
    radius: float | None = None  # default: driver radius + 2 mm
    # Rectangular plate (e.g. a planar-magnetic magnet-bar array): shape="rect"
    # uses width x height; pattern_angle_deg rotates the hole pattern in-plane
    # (Slots at 90 deg = vertical bars).
    shape: Literal["disc", "rect"] = "disc"
    width: float = 65e-3
    height: float = 90e-3
    pattern_angle_deg: float = 0.0
    # Fazor-style chamfer for kind="slots": >0 flares the gap toward the ear
    # over the exit `chamfer_fraction` of the thickness (bar cross-section
    # becomes trapezoidal), shortening the acoustic neck.
    chamfer_depth: float = 0.0
    chamfer_fraction: float = 0.66
    # fib (Fibonacci/phyllotaxis spiral web, MDR-Z1R style):
    rib_width: float = 1.0e-3
    spirals_cw: int = 8
    spirals_ccw: int = 13
    winding: float = 0.9
    # Seal the rim to the baffle with a cylindrical collar (realistic mounting:
    # all sound must pass through the holes). False = free-floating disc,
    # which lets sound diffract around the rim through the standoff gap.
    sealed: bool = True
    hole_radius: float = 0.5e-3  # hex
    pitch: float = 2e-3  # hex, slots
    slot_width: float = 1e-3  # slots
    rings: tuple[tuple[float, float], ...] = ()  # ring slits (r_in, r_out)

    def pattern(self) -> HolePattern | None:
        if self.kind == "hex":
            return HexHoles(hole_radius=self.hole_radius, pitch=self.pitch)
        if self.kind == "slots":
            if self.chamfer_depth > 0.0:
                return ChamferedSlots(
                    width=self.slot_width,
                    pitch=self.pitch,
                    chamfer_depth=self.chamfer_depth,
                    chamfer_fraction=self.chamfer_fraction,
                )
            return Slots(width=self.slot_width, pitch=self.pitch)
        if self.kind == "rings":
            return RingSlits(rings=self.rings)
        if self.kind == "fib":
            return FibonacciSpirals(
                rib_width=self.rib_width,
                m_cw=self.spirals_cw,
                m_ccw=self.spirals_ccw,
                winding=self.winding,
            )
        return None  # solid


@dataclass(frozen=True)
class PinnaSpec:
    kind: Literal["mesh", "parametric", "none"] = "parametric"
    mesh_path: str | None = None
    side: Literal["left", "right"] = "left"  # left = negative end of lateral_axis
    lateral_axis: int = 1  # interaural axis of the mesh (HUTUBS: y)
    scale: float = 1.0  # e.g. 1e-3 if the mesh is in millimeters
    extra_rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    region_size: float = 90e-3
    # Ear-canal entrance in ORIGINAL mesh coordinates (after `scale`); None =
    # auto-detect via the interaural-axis convention (HUTUBS-style alignment).
    canal_hint: tuple[float, float, float] | None = None
    # Pinna probe selection: cells protruding more than this beyond the fitted
    # head-skin surface count as pinna; the concha bowl (recessed, so never
    # "protruding") is added within concha_radius of the canal.
    probe_protrusion: float = 2.5e-3
    concha_radius: float = 15e-3


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
    solid: torch.Tensor  # bool occupancy (CPU), for geometry visualization
    canal_position: Vec3 = (0.0, 0.0, 0.0)  # ear-canal entrance in scene coords
    porosities: list[float] = field(default_factory=list)
    # Named per-part occupancy grids (baffle, filter_1, pinna, ...) for
    # color-coded geometry visualization; their union equals ``solid``.
    parts: list[tuple[str, torch.Tensor]] = field(default_factory=list)


def _domain_grid(config: SceneConfig) -> tuple[Grid, Vec3, float]:
    """Grid, driver center, and pinna reference z."""
    hx, hy = config.driver.half_extents()
    r_driver = max(hx, hy)
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


def build_scene(
    config: SceneConfig,
    device: str | None = None,
    probes_override: npt.NDArray[np.float64] | None = None,
) -> BuiltScene:
    """Assemble a scene. ``probes_override`` reuses an existing probe layout
    verbatim (for paired with/without-filter runs, which must share probes)."""
    grid, driver_center, z_pinna = _domain_grid(config)
    n_steps = int(np.ceil(config.record_ms * 1e-3 / grid.dt))
    drv_hx, drv_hy = config.driver.half_extents()
    r_driver = max(drv_hx, drv_hy)

    tilt = np.deg2rad(config.driver.tilt_deg)
    normal = (0.0, float(np.sin(tilt)), float(np.cos(tilt)))

    waveform = gaussian_modulated_sine(
        grid.dt,
        n_steps,
        center_frequency=config.driver.center_frequency,
        bandwidth_frequency=config.driver.bandwidth_frequency,
    )
    sources: list[Source]
    if config.driver.shape == "rect":
        sources = [
            RectangularPistonSource(
                center=driver_center,
                normal=normal,
                width=config.driver.width,
                height=config.driver.height,
                waveform=waveform,
            )
        ]
    else:
        sources = [
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
    parts: list[tuple[str, torch.Tensor]] = []

    def add_part(name: str, occ_part: torch.Tensor) -> None:
        nonlocal solid
        parts.append((name, occ_part))
        solid |= occ_part

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
        add_part("baffle", occ)

    if config.cup_depth > 0.0:
        r_baffle = config.baffle_radius or (r_driver + 10e-3)
        add_part(
            "cup",
            parametric.cup_shell(
                grid,
                center=along_normal(driver_center, -1.0 * config.dx),
                axis=(-normal[0], -normal[1], -normal[2]),
                inner_radius=r_baffle,
                depth=config.cup_depth,
                thickness=2e-3,
            ),
        )

    for i, spec in enumerate(config.filters):
        if spec.shape == "rect":
            occ, porosity = parametric.rect_plate(
                grid,
                along_normal(driver_center, spec.standoff),
                normal,
                width=spec.width,
                height=spec.height,
                thickness=spec.thickness,
                pattern=spec.pattern(),
                pattern_angle_deg=spec.pattern_angle_deg,
            )
            if spec.sealed:
                occ = occ | parametric.rect_collar(
                    grid,
                    driver_center,
                    normal,
                    width=spec.width,
                    height=spec.height,
                    length=spec.standoff + spec.thickness / 2.0,
                    thickness=2 * config.dx,
                )
        else:
            r_filter = spec.radius or (r_driver + 2e-3)
            occ, porosity = parametric.plate(
                grid,
                along_normal(driver_center, spec.standoff),
                normal,
                radius=r_filter,
                thickness=spec.thickness,
                pattern=spec.pattern(),
                pattern_angle_deg=spec.pattern_angle_deg,
            )
            if spec.sealed:
                occ = occ | parametric.annular_collar(
                    grid,
                    driver_center,
                    normal,
                    radius=r_filter,
                    length=spec.standoff + spec.thickness / 2.0,
                    thickness=2 * config.dx,
                )
        add_part(f"filter_{i + 1}", occ)
        porosities.append(porosity)

    pinna_center = (driver_center[0], driver_center[1], z_pinna)
    canal_position = pinna_center  # refined below for mesh pinnae
    probes: npt.NDArray[np.float64]
    if config.pinna.kind == "mesh":
        if config.pinna.mesh_path is None:
            raise ValueError("pinna.kind='mesh' requires mesh_path")
        head = load_mesh(Path(config.pinna.mesh_path))
        if config.pinna.scale != 1.0:
            head.apply_scale(config.pinna.scale)
        if config.pinna.canal_hint is not None:
            canal = np.asarray(config.pinna.canal_hint, dtype=np.float64)
        else:
            canal = find_ear_canal_entrance(
                head, side=config.pinna.side, lateral_axis=config.pinna.lateral_axis
            )
        ear_mesh = extract_ear_region(
            head,
            canal,
            side=config.pinna.side,
            box_size=config.pinna.region_size,
            lateral_axis=config.pinna.lateral_axis,
        )
        placed = orient_pinna_to_scene(
            ear_mesh,
            canal,
            side=config.pinna.side,
            lateral_axis=config.pinna.lateral_axis,
            extra_rotation_deg=config.pinna.extra_rotation_deg,
        )
        # Aim the driver axis at the ear canal (x, y) and put the pinna's most
        # protruding point on the z_pinna plane, so `distance` is the closest
        # driver-to-pinna gap; the recessed canal lies deeper.
        z_tip = float(placed.vertices[:, 2].min())
        placed.apply_translation([pinna_center[0], pinna_center[1], z_pinna - z_tip])
        canal_position = (pinna_center[0], pinna_center[1], z_pinna - z_tip)
        pinna_occ = voxelize(placed, grid)
        add_part("pinna", pinna_occ)
        probes = select_pinna_probes(
            placed,
            np.asarray(canal_position),
            config.n_probes,
            offset=config.probe_offset,
            protrusion_threshold=config.pinna.probe_protrusion,
            concha_radius=config.pinna.concha_radius,
        )
        # The 2D pinna mask cannot tell front from back surfaces sharing the
        # same (x, y); drop probes the pinna itself hides from the driver.
        probes = _remove_driver_occluded_probes(probes, pinna_occ, grid, driver_center)
    elif config.pinna.kind == "parametric":
        from headphone_sims.geometry.pinna import parametric_pinna

        add_part("pinna", parametric_pinna(grid, pinna_center))
        # Head-surface plate behind the pinna (front face on the pinna plane).
        head_plate, _ = parametric.plate(
            grid,
            (pinna_center[0], pinna_center[1], z_pinna + config.dx),
            (0.0, 0.0, 1.0),
            radius=45e-3,
            thickness=2 * config.dx,
        )
        add_part("head", head_plate)
        probes = _parametric_pinna_probes(
            pinna_center, n=config.n_probes, offset=config.probe_offset
        )
    else:
        probes = _disc_probes(
            pinna_center, radius=30e-3, n=config.n_probes, offset=config.probe_offset
        )

    # A tilted/close driver assembly must not intersect the pinna or head.
    if config.pinna.kind != "none":
        pinna_solid = dict(parts).get("pinna")
        if pinna_solid is not None:
            for name, occ_part in parts:
                if name == "pinna":
                    continue
                n_overlap = int((occ_part & pinna_solid).sum())
                if n_overlap:
                    raise ValueError(
                        f"part '{name}' intersects the pinna/head in {n_overlap} cells — "
                        "increase distance or reduce tilt"
                    )

    # Drop probes that ended up inside (or trilinearly touching) solid voxels —
    # staircased surfaces can swallow surface-hugging probes.
    if probes_override is not None:
        probes = probes_override
    else:
        probes = _remove_probes_in_solid(probes, solid, grid)

    # Reference probe: closest to the ear-canal entrance.
    ref = int(np.argmin(np.linalg.norm(probes - np.asarray(canal_position), axis=1)))

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
        solid=solid,
        canal_position=canal_position,
        porosities=porosities,
        parts=parts,
    )


def _remove_driver_occluded_probes(
    probes: npt.NDArray[np.float64],
    occupancy: torch.Tensor,
    grid: Grid,
    driver_center: Vec3,
    skip: float = 2.5e-3,
) -> npt.NDArray[np.float64]:
    """Keep only probes with a clear line of sight to the driver center.

    Marches each probe->driver segment through the voxel occupancy (half-cell
    steps). This removes points wrapped around the pinna silhouette — which a
    straight -z column test lets through — while the concha stays (visible
    through its opening). The first ``skip`` of the segment is ignored so a
    probe is not blocked by the staircase of its own surface.
    """
    occ = occupancy.numpy()
    target = np.asarray(driver_center, dtype=np.float64)
    delta = target[None, :] - probes
    length = np.linalg.norm(delta, axis=1)
    direction = delta / length[:, None]
    step = 0.5 * grid.dx
    visible = np.ones(len(probes), dtype=bool)
    n_steps = int(np.ceil((float(length.max()) - skip) / step))
    shape = np.array(grid.shape)
    for s in range(n_steps):
        t = skip + s * step
        active = visible & (t < length - step)
        if not active.any():
            break
        pts = probes[active] + direction[active] * t
        idx = np.floor(pts / grid.dx).astype(int)
        inside = np.all((idx >= 0) & (idx < shape), axis=1)
        blocked = np.zeros(int(active.sum()), dtype=bool)
        ii = idx[inside]
        blocked[inside] = occ[ii[:, 0], ii[:, 1], ii[:, 2]]
        active_idx = np.flatnonzero(active)
        visible[active_idx[blocked]] = False
    if not visible.any():
        raise ValueError("all probes are occluded from the driver; check the geometry")
    n_dropped = int((~visible).sum())
    if n_dropped:
        print(f"dropped {n_dropped} probes without line of sight to the driver")
    out: npt.NDArray[np.float64] = probes[visible]
    return out


def _remove_probes_in_solid(
    probes: npt.NDArray[np.float64], solid: torch.Tensor, grid: Grid
) -> npt.NDArray[np.float64]:
    """Keep only probes whose 8 trilinear corner cells are all air."""
    solid_np = solid.numpy()
    keep = np.ones(len(probes), dtype=bool)
    for i, pos in enumerate(probes):
        base = np.floor(pos / grid.dx - 0.5).astype(int)
        for corner in range(8):
            idx = base + np.array([(corner >> 2) & 1, (corner >> 1) & 1, corner & 1])
            idx = np.clip(idx, 0, np.array(grid.shape) - 1)
            if solid_np[idx[0], idx[1], idx[2]]:
                keep[i] = False
                break
    if not keep.any():
        raise ValueError("all probes fall inside solid geometry")
    if not keep.all():
        print(f"dropped {int((~keep).sum())} probes inside solid geometry")
    return probes[keep]


def _disc_probes(
    center: Vec3, radius: float, n: int, offset: float = 0.0
) -> npt.NDArray[np.float64]:
    """Sunflower-spiral probe layout on the pinna reference plane, shifted
    ``offset`` toward the driver."""
    k = np.arange(n, dtype=np.float64) + 0.5
    r = radius * np.sqrt(k / n)
    theta = k * 2.399963229728653  # golden angle
    pts = np.zeros((n, 3))
    pts[:, 0] = center[0] + r * np.cos(theta)
    pts[:, 1] = center[1] + r * np.sin(theta)
    pts[:, 2] = center[2] - offset
    return pts


def _parametric_pinna_probes(
    center: Vec3,
    n: int,
    offset: float,
    height: float = 60e-3,
    width: float = 35e-3,
    protrusion: float = 18e-3,
) -> npt.NDArray[np.float64]:
    """Probes hugging the parametric pinna's driver-facing ellipsoid surface.

    Must match the default geometry of
    :func:`headphone_sims.geometry.pinna.parametric_pinna`. Points outside the
    ellipse footprint sit just off the head-surface plane.
    """
    ax, ay, az = width / 2.0, height / 2.0, protrusion
    k = np.arange(n, dtype=np.float64) + 0.5
    r = np.sqrt(k / n)
    theta = k * 2.399963229728653
    # Probes stay on the pinna footprint itself (evaluation is pinna-only).
    u = r * np.cos(theta)
    v = r * np.sin(theta)
    inside = u**2 + v**2 < 1.0
    z_surf = np.where(inside, -az * np.sqrt(np.clip(1.0 - u**2 - v**2, 0.0, 1.0)), 0.0)
    pts = np.zeros((n, 3))
    pts[:, 0] = u * ax
    pts[:, 1] = v * ay
    pts[:, 2] = z_surf
    # Offset along the outward ellipsoid normal (on steep flanks a pure -z
    # offset would stay inside the staircased shell), or -z off the footprint.
    grad = np.stack([pts[:, 0] / ax**2, pts[:, 1] / ay**2, pts[:, 2] / az**2], axis=1)
    grad[~inside] = [0.0, 0.0, -1.0]
    grad /= np.linalg.norm(grad, axis=1, keepdims=True)
    pts += offset * grad
    return np.asarray(pts + np.asarray(center), dtype=np.float64)
