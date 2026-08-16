import numpy as np
import pytest

from headphone_sims.geometry.scene import _filter_to_footprint


def test_footprint_keeps_only_inside_points() -> None:
    center = (0.05, 0.05, 0.04)
    rng = np.random.default_rng(0)
    probes = np.zeros((500, 3))
    probes[:, 0] = center[0] + rng.uniform(-0.05, 0.05, 500)
    probes[:, 1] = center[1] + rng.uniform(-0.05, 0.05, 500)
    probes[:, 2] = 0.04
    region = (0.042, 0.072)
    out = _filter_to_footprint(probes, center, region, n_max=400)
    d = ((out[:, 0] - center[0]) / (region[0] / 2)) ** 2 + (
        (out[:, 1] - center[1]) / (region[1] / 2)
    ) ** 2
    assert (d <= 1.0).all()
    assert 0 < len(out) <= 400


def test_footprint_caps_count() -> None:
    center = (0.0, 0.0, 0.0)
    probes = np.zeros((100, 3))  # all at the center -> all inside
    out = _filter_to_footprint(probes, center, (0.04, 0.07), n_max=10)
    assert len(out) == 10


def test_footprint_empty_raises() -> None:
    probes = np.full((5, 3), 1.0)
    with pytest.raises(ValueError, match="footprint"):
        _filter_to_footprint(probes, (0.0, 0.0, 0.0), (0.04, 0.07), n_max=10)
