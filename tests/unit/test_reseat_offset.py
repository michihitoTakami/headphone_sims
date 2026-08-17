"""Re-seat vertical offset (PinnaSpec.offset_y): pinna+canal shift, driver stays."""

from pathlib import Path

import numpy as np
import trimesh

from headphone_sims.geometry.scene import DriverSpec, PinnaSpec, SceneConfig, build_scene


def _synthetic_head(path: Path) -> str:
    """HUTUBS-like left-ear stand-in: skin plate + ellipsoid pinna bump.

    Mesh convention: interaural axis = y, left ear at negative y, bump apex on
    the y axis (so the canal locator lands on it).
    """
    skin = trimesh.creation.box(extents=[0.12, 0.004, 0.12])
    skin.apply_translation([0.0, -0.083, 0.0])  # driver-facing surface at y=-0.085
    bump = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    bump.apply_scale([0.015, 0.012, 0.025])
    bump.apply_translation([0.0, -0.085, 0.0])  # protrudes to y=-0.097
    mesh = trimesh.util.concatenate([skin, bump])
    assert isinstance(mesh, trimesh.Trimesh)
    out = path / "head.ply"
    mesh.export(out)
    return str(out)


def _config(mesh_path: str, offset_y: float) -> SceneConfig:
    return SceneConfig(
        dx=1e-3,
        distance=12e-3,
        driver=DriverSpec(diameter=40e-3),
        pinna=PinnaSpec(kind="mesh", mesh_path=mesh_path, side="left", offset_y=offset_y),
        record_ms=0.1,
        n_probes=150,
    )


def test_offset_y_shifts_pinna_and_canal(tmp_path: Path) -> None:
    mesh_path = _synthetic_head(tmp_path)
    base = build_scene(_config(mesh_path, 0.0), device="cpu")
    shifted = build_scene(_config(mesh_path, 2e-3), device="cpu")

    # Canal moves by exactly the offset along y; depth placement is unchanged.
    assert np.isclose(shifted.canal_position[1] - base.canal_position[1], 2e-3)
    assert np.isclose(shifted.canal_position[2], base.canal_position[2])
    # The probe cloud rides along with the pinna...
    dy = shifted.probe_positions[:, 1].mean() - base.probe_positions[:, 1].mean()
    assert abs(dy - 2e-3) < 1e-3
    # ...while the driver does not move.
    assert shifted.driver_center == base.driver_center
