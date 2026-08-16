"""AMTS-style metamaterial insert: sloped top, through-tubes, Helmholtz cells."""

from pathlib import Path

import pytest
import torch

from headphone_sims.experiments.config import load_config
from headphone_sims.geometry.parametric import AmtsHexCells, rect_plate
from headphone_sims.geometry.scene import (
    DriverSpec,
    FilterSpec,
    PinnaSpec,
    SceneConfig,
    build_scene,
)
from headphone_sims.grid import Grid

PATTERN = AmtsHexCells(
    tube_radius=1.5e-3,
    pitch=4.5e-3,
    slope_length=30e-3,
    min_thickness=3e-3,
    max_thickness=15e-3,
    neck_radius=1.2e-3,
    neck_length=1e-3,
    bottom_wall=1e-3,
)


def _open_at(a_mm: float, b_mm: float, h_mm: float) -> bool:
    """Openness at pattern coords (a, b) and height h above the bottom face."""
    a = torch.tensor([a_mm * 1e-3], dtype=torch.float64)
    b = torch.tensor([b_mm * 1e-3], dtype=torch.float64)
    frac = torch.tensor([h_mm * 1e-3 / PATTERN.max_thickness - 0.5], dtype=torch.float64)
    return bool(PATTERN.open_mask_3d(a, b, frac)[0])


def _z_top_mm(a_mm: float) -> float:
    t = min(max(a_mm / 30.0 + 0.5, 0.0), 1.0)
    return 3.0 + 12.0 * t


def test_amts_through_tube_open_full_depth() -> None:
    # Cell (m=0, n=0) at the pattern origin has even parity: a through tube.
    for h in (0.25, 4.0, 8.0):  # z_top(0) = 9mm
        assert _open_at(0.0, 0.0, h)
    assert _open_at(0.0, 0.0, 9.5)  # above the sloped top: carved air


def test_amts_helmholtz_cell_layers() -> None:
    # Cell (m=0, n=1) at a=4.5mm has odd parity: a Helmholtz cell.
    # z_top(4.5) = 9.75mm, neck region above 8.75mm.
    assert not _open_at(4.5, 0.0, 0.5)  # closed bottom wall (1mm)
    assert _open_at(4.5, 0.0, 5.0)  # cavity (full tube radius)
    assert _open_at(4.5, 0.0, 9.3)  # neck at the cell center
    # Off-center point (r=1.3mm > neck radius): the top surface slopes, so use
    # the LOCAL top height there — z_top(5.8) = 11.32mm, neck above 10.32mm.
    assert _open_at(4.5 + 1.3, 0.0, 5.0)  # cavity is wider than the neck...
    assert not _open_at(4.5 + 1.3, 0.0, 10.8)  # ...which is closed off-center
    assert _open_at(4.5 + 1.3, 0.0, 11.8)  # above the local top: open


def test_amts_web_is_solid_below_sloped_top() -> None:
    # Wall point between cells (0,0) and (4.5,0): solid below z_top, open above.
    z_top = _z_top_mm(2.25)  # = 9.9mm
    assert not _open_at(2.25, 0.0, z_top - 1.0)
    assert not _open_at(2.25, 0.0, 0.5)
    assert _open_at(2.25, 0.0, z_top + 1.0)


def test_amts_plate_local_thickness_follows_slope() -> None:
    """Voxelized web-column heights match the sloped top at both ends."""
    grid = Grid.create((64, 96, 64), dx=0.5e-3)
    center = (16e-3, 24e-3, 12e-3)
    occ, _ = rect_plate(
        grid,
        center,
        (0.0, 0.0, 1.0),
        width=20e-3,
        height=30e-3,
        thickness=15e-3,
        pattern=PATTERN,
    )
    dx = grid.dx

    def solid_height_mm(a_mm: float, b_mm: float) -> float:
        # _rect_frame with +z normal: a = y - cy, b = -(x - cx).
        j = int((24.0 + a_mm) * 1e-3 / dx)
        i = int((16.0 - b_mm) * 1e-3 / dx)
        return float(occ[i, j, :].sum()) * dx * 1e3

    # Web (wall) columns: full local height, thick vs thin end.
    thick = solid_height_mm(11.25, 0.0)  # wall point, z_top = 9.9... use formula
    assert thick == pytest.approx(_z_top_mm(11.25), abs=2 * dx * 1e3)
    thin = solid_height_mm(-11.25, 0.0)
    assert thin == pytest.approx(_z_top_mm(-11.25), abs=2 * dx * 1e3)
    assert thick > thin + 4.0  # the slope is really there (~7.2mm difference)

    # Through-tube center column: fully open.
    assert solid_height_mm(0.0, 0.0) == 0.0
    # Helmholtz center column: only the 1mm bottom wall is solid (neck is
    # open at the cell center).
    assert solid_height_mm(4.5, 0.0) == pytest.approx(1.0, abs=2 * dx * 1e3)


def test_amts_scene_builds_with_housing() -> None:
    config = SceneConfig(
        dx=1e-3,
        driver=DriverSpec(shape="rect", width=20e-3, height=30e-3),
        filters=(
            FilterSpec(
                kind="amts",
                shape="rect",
                width=20e-3,
                height=30e-3,
                hole_radius=1.5e-3,
                pitch=4.5e-3,
                thickness=8e-3,
                amts_min_thickness=2e-3,
                standoff=6e-3,
            ),
        ),
        pinna=PinnaSpec(kind="none"),
        distance=18e-3,
        lateral_margin=2e-3,
        axial_margin=5e-3,
        sponge_thickness=8,
        n_probes=16,
        record_ms=0.1,
    )
    built = build_scene(config, device="cpu")
    names = [name for name, _ in built.parts]
    assert "filter_1" in names
    assert "housing_1" in names
    assert int(dict(built.parts)["filter_1"].sum()) > 0


def test_amts_yaml_loads_and_rejects_typos(tmp_path: Path) -> None:
    good = tmp_path / "amts.yaml"
    good.write_text(
        """
name: amts_test
scene:
  driver: { shape: rect, width: 45.0e-3, height: 72.0e-3 }
  filters:
    - kind: amts
      shape: rect
      width: 45.0e-3
      height: 72.0e-3
      thickness: 15.0e-3
      amts_min_thickness: 3.0e-3
      amts_neck_radius: 1.2e-3
      standoff: 13.5e-3
  pinna: { kind: none }
""",
        encoding="utf-8",
    )
    cfg = load_config(good)
    spec = cfg.scene.filters[0]
    assert spec.kind == "amts"
    assert spec.amts_min_thickness == pytest.approx(3e-3)
    pattern = spec.pattern()
    assert isinstance(pattern, AmtsHexCells)
    assert pattern.slope_length == pytest.approx(72e-3)
    assert pattern.max_thickness == pytest.approx(15e-3)
    bad = tmp_path / "typo.yaml"
    bad.write_text(
        "name: t\nscene: { filters: [ { kind: amts, amts_min_thicknes: 3.0e-3 } ] }\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown FilterSpec keys"):
        load_config(bad)
