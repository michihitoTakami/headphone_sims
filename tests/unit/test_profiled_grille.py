"""Curved grille following the dome profile at constant gap."""

import dataclasses

import numpy as np
import torch

from headphone_sims.experiments.config import load_config
from headphone_sims.geometry.parametric import DomeProfile, profiled_plate
from headphone_sims.geometry.scene import build_scene
from headphone_sims.grid import Grid


def test_profiled_plate_keeps_constant_gap() -> None:
    grid = Grid.create((120, 120, 60), dx=1e-3)
    profile = DomeProfile(
        radius=30e-3, dome_fraction=0.5, dome_depth=7e-3, edge_height=4e-3, surround="cone"
    )
    center = (60e-3, 60e-3, 10e-3)
    gap, thickness = 2e-3, 1e-3
    occ, porosity = profiled_plate(grid, center, (0.0, 0.0, 1.0), profile, gap, thickness)
    assert porosity == 0.0  # no pattern -> solid shell
    idx = occ.nonzero(as_tuple=False).to(torch.float64)
    centers = (idx + 0.5) * grid.dx
    lat = torch.linalg.vector_norm(centers[:, :2] - torch.tensor(center[:2]), dim=1)
    axial = centers[:, 2] - center[2]
    h = profile.height(lat)
    rel = (axial - h).numpy()
    # Every shell cell sits within (gap, gap+thickness] of the local profile
    # height, +/- one cell of staircase.
    assert rel.min() > gap - 1.5 * grid.dx
    assert rel.max() <= gap + thickness + 1.5 * grid.dx


def test_sealed_profiled_solid_grille_is_airtight() -> None:
    cfg = load_config("configs/hutubs_70mm_z1r_dome.yaml").scene
    scene = dataclasses.replace(
        cfg,
        dx=1.0e-3,
        record_ms=0.3,
        sponge_thickness=8,
        lateral_margin=6e-3,
        axial_margin=8e-3,
        n_probes=20,
        pinna=dataclasses.replace(cfg.pinna, kind="none", mesh_path=None),
        filters=(
            dataclasses.replace(
                cfg.filters[0], kind="solid", follow_profile=True, gap=2e-3, thickness=2e-3
            ),
        ),
    )
    built = build_scene(scene, device="cpu")
    result = built.simulation.run()
    assert float(np.abs(result.p).max()) == 0.0
