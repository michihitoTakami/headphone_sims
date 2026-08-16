from pathlib import Path

import numpy as np
import pytest
import trimesh

from headphone_sims.geometry.mesh import clip_to_box, load_mesh, surface_probes, voxelize
from headphone_sims.grid import Grid


def _sphere(
    radius: float = 8e-3, center: tuple[float, float, float] = (16e-3, 16e-3, 16e-3)
) -> trimesh.Trimesh:
    m: trimesh.Trimesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    m.apply_translation(center)
    return m


def test_voxelize_sphere_volume() -> None:
    grid = Grid.create((64, 64, 64), dx=0.5e-3)
    radius = 8e-3
    occ = voxelize(_sphere(radius), grid)
    volume = int(occ.sum()) * grid.dx**3
    analytic = 4.0 / 3.0 * np.pi * radius**3
    assert volume == pytest.approx(analytic, rel=0.10)


def test_voxelize_without_fill_is_hollow() -> None:
    grid = Grid.create((64, 64, 64), dx=0.5e-3)
    shell = voxelize(_sphere(), grid, fill=False)
    solid = voxelize(_sphere(), grid, fill=True)
    assert int(shell.sum()) < int(solid.sum())
    # Center cell open in the shell, filled in the solid.
    c = grid.cell_index((16e-3, 16e-3, 16e-3))
    assert not bool(shell[c])
    assert bool(solid[c])


def test_clip_to_box() -> None:
    m = _sphere(radius=8e-3, center=(0.0, 0.0, 0.0))
    clipped = clip_to_box(m, (-10e-3, -10e-3, 0.0), (10e-3, 10e-3, 10e-3))
    assert clipped.vertices[:, 2].min() >= -1e-9
    with pytest.raises(ValueError, match="does not intersect"):
        clip_to_box(m, (1.0, 1.0, 1.0), (2.0, 2.0, 2.0))


def test_surface_probes_offset_and_direction() -> None:
    radius = 8e-3
    center = np.array([16e-3, 16e-3, 16e-3])
    probes = surface_probes(_sphere(radius), n_points=200, offset=2e-3, direction=(0, 0, -1))
    r = np.linalg.norm(probes - center, axis=1)
    np.testing.assert_allclose(r, radius + 2e-3, rtol=0.05)
    # Only the -z hemisphere was sampled.
    assert (probes[:, 2] < center[2] + 1e-4).all()


def test_load_mesh_round_trip(tmp_path: Path) -> None:
    m = _sphere()
    path = tmp_path / "sphere.stl"
    m.export(path)
    loaded = load_mesh(path)
    assert loaded.is_watertight
    assert len(loaded.faces) > 0
