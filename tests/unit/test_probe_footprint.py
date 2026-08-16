"""Probe selection on the pinna proper: protrusion-based detection tests."""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from headphone_sims.geometry.pinna import find_ear_canal_entrance, select_pinna_probes


def _synthetic_ear_scene() -> tuple[trimesh.Trimesh, np.ndarray]:
    """Flat 'head skin' plate at z~0 with an ellipsoidal 'pinna' bump toward -z.

    Scene orientation (driver side = -z). Canal at the bump center (0, 0, 0).
    """
    skin = trimesh.creation.box(extents=[0.10, 0.10, 0.004])
    skin.apply_translation([0.0, 0.0, 0.002])  # skin surface at z=0, solid behind
    bump = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    bump.apply_scale([0.015, 0.025, 0.012])  # 30x50 mm footprint, 12 mm protrusion
    mesh = trimesh.util.concatenate([skin, bump])
    assert isinstance(mesh, trimesh.Trimesh)
    canal = np.array([0.0, 0.0, 0.0])
    return mesh, canal


def test_probes_confined_to_bump() -> None:
    mesh, canal = _synthetic_ear_scene()
    probes = select_pinna_probes(mesh, canal, n_probes=200, offset=1e-3, concha_radius=5e-3)
    assert len(probes) > 50
    # All probes on/near the bump footprint (30x50 mm + dilation + offset).
    assert (np.abs(probes[:, 0]) < 0.015 + 4e-3).all()
    assert (np.abs(probes[:, 1]) < 0.025 + 4e-3).all()
    # None on the far flat skin.
    r = np.hypot(probes[:, 0] / 0.015, probes[:, 1] / 0.025)
    assert (r < 1.4).all()


def test_probes_cover_bottom_half() -> None:
    mesh, canal = _synthetic_ear_scene()
    probes = select_pinna_probes(mesh, canal, n_probes=200, offset=1e-3, concha_radius=5e-3)
    # The lower (negative-y) half must be covered too — the original footprint
    # bug left the lobule empty.
    assert (probes[:, 1] < -0.010).sum() >= 10
    assert (probes[:, 1] > 0.010).sum() >= 10


def test_back_facing_surfaces_excluded() -> None:
    mesh, canal = _synthetic_ear_scene()
    probes = select_pinna_probes(mesh, canal, n_probes=300, offset=1e-3, concha_radius=5e-3)
    # Driver-facing side only: nothing behind the skin plane (bump back half
    # and plate back are at z > 0).
    assert (probes[:, 2] < 3e-3).all()


def test_no_pinna_raises() -> None:
    flat = trimesh.creation.box(extents=[0.1, 0.1, 0.004])
    with pytest.raises(ValueError, match="protruding"):
        select_pinna_probes(flat, np.zeros(3), n_probes=50)


HUTUBS_PP1 = Path("data/hutubs/pp1_3DheadMesh.ply")


@pytest.mark.skipif(not HUTUBS_PP1.exists(), reason="HUTUBS pp1 mesh not downloaded")
def test_hutubs_canal_on_interaural_axis() -> None:
    mesh = trimesh.load(str(HUTUBS_PP1), force="mesh")
    assert isinstance(mesh, trimesh.Trimesh)
    canal = find_ear_canal_entrance(mesh, side="left", lateral_axis=1)
    # HUTUBS aligns the interaural (y) axis through the canal entrances.
    assert abs(canal[0]) < 2e-3
    assert abs(canal[2]) < 2e-3
    assert canal[1] < -0.05


def test_driver_occluded_probes_removed() -> None:
    import torch

    from headphone_sims.geometry.scene import _remove_driver_occluded_probes
    from headphone_sims.grid import Grid

    grid = Grid.create((20, 20, 20), dx=1e-3)
    occ = torch.zeros(grid.shape, dtype=torch.bool)
    occ[8:12, 8:12, 10] = True  # thin plate at z = 10.5 mm
    probes = np.array(
        [
            [10e-3, 10e-3, 8.0e-3],  # in front of the plate -> visible
            [10e-3, 10e-3, 14.0e-3],  # behind the plate -> occluded
            [2e-3, 2e-3, 14.0e-3],  # deep but off the plate column -> visible
        ]
    )
    out = _remove_driver_occluded_probes(probes, occ, grid)
    assert out.shape == (2, 3)
    assert (out[:, 2] != 14.0e-3).any() or (out[:, 0] == 2e-3).any()
    np.testing.assert_allclose(sorted(out[:, 2]), [8.0e-3, 14.0e-3])
    assert not ((out[:, 0] == 10e-3) & (out[:, 2] == 14e-3)).any()
