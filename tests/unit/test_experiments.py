"""End-to-end smoke test of the experiment pipeline on a tiny scene."""

import json
from pathlib import Path

import yaml

from headphone_sims.experiments.config import load_config
from headphone_sims.experiments.runner import run_experiment

TINY_CONFIG = {
    "name": "tiny_smoke",
    "device": "cpu",
    "compare_without_filters": True,
    "scene": {
        "dx": 2.0e-3,
        "distance": 26.0e-3,
        "record_ms": 0.25,
        "snapshot_every": 30,
        "sponge_thickness": 8,
        "lateral_margin": 6.0e-3,
        "axial_margin": 8.0e-3,
        "n_probes": 40,
        "driver": {"diameter": 30.0e-3, "tilt_deg": 10.0},
        "filters": [{"kind": "hex", "hole_radius": 2.0e-3, "pitch": 6.0e-3, "standoff": 4.0e-3}],
        "pinna": {"kind": "parametric"},
    },
}


def test_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(TINY_CONFIG), encoding="utf-8")
    config = load_config(path)
    assert config.name == "tiny_smoke"
    assert config.scene.driver.tilt_deg == 10.0
    assert config.scene.filters[0].kind == "hex"


def test_run_experiment_produces_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(TINY_CONFIG), encoding="utf-8")
    config = load_config(path)
    run_dir = run_experiment(config, run_dir=tmp_path / "run")

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "signals.npz").exists()
    assert (run_dir / "signals_nofilter.npz").exists()
    payload = json.loads((run_dir / "metrics.json").read_text())
    assert "metrics" in payload
    assert "filter_comparison" in payload
    assert 0.0 <= payload["metrics"]["similarity_mean"] <= 1.0
    report = run_dir / "report" / "report.html"
    assert report.exists()
    assert "similarity" in report.read_text()
    assert (run_dir / "report" / "field.gif").exists()


def test_unknown_config_key_rejected(tmp_path: Path) -> None:
    bad = dict(TINY_CONFIG, typo_key=1)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="unknown"):
        load_config(path)
