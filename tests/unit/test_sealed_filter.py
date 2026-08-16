"""Sealed filter mounting: with a solid sealed plate no sound escapes the
driver cavity; unsealed, sound diffracts around the rim."""

import numpy as np

from headphone_sims.geometry.scene import (
    DriverSpec,
    FilterSpec,
    PinnaSpec,
    SceneConfig,
    build_scene,
)


def _config(sealed: bool) -> SceneConfig:
    return SceneConfig(
        dx=1.0e-3,
        distance=14e-3,
        record_ms=0.3,
        sponge_thickness=8,
        lateral_margin=6e-3,
        axial_margin=8e-3,
        n_probes=30,
        driver=DriverSpec(diameter=30e-3),
        filters=(FilterSpec(kind="solid", standoff=4e-3, thickness=2e-3, sealed=sealed),),
        pinna=PinnaSpec(kind="none"),
    )


def test_sealed_solid_plate_is_airtight() -> None:
    built = build_scene(_config(sealed=True), device="cpu")
    result = built.simulation.run()
    assert float(np.abs(result.p).max()) == 0.0


def test_unsealed_plate_leaks_around_rim() -> None:
    built = build_scene(_config(sealed=False), device="cpu")
    result = built.simulation.run()
    assert float(np.abs(result.p).max()) > 0.0
