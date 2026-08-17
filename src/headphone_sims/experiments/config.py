"""YAML <-> dataclass experiment configuration.

An experiment YAML mirrors the nested scene dataclasses::

    name: hex_filter_40mm
    device: null            # null = auto (cuda if available)
    compare_without_filters: true
    scene:
      dx: 0.5e-3
      distance: 15.0e-3
      driver: { diameter: 40.0e-3, tilt_deg: 0.0 }
      filters:
        - { kind: hex, hole_radius: 0.5e-3, pitch: 2.0e-3, standoff: 3.0e-3 }
      pinna: { kind: parametric }
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from headphone_sims.geometry.scene import DriverSpec, FilterSpec, PinnaSpec, SceneConfig


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    scene: SceneConfig = field(default_factory=SceneConfig)
    device: str | None = None
    compare_without_filters: bool = False
    output_dir: str = "runs"


def _build(cls: type[Any], data: dict[str, Any]) -> Any:
    """Recursively construct a (frozen) dataclass from plain dict/YAML data."""
    kwargs: dict[str, Any] = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    for key, value in data.items():
        f = fields[key]
        if key == "scene":
            kwargs[key] = _build(SceneConfig, value)
        elif key == "driver":
            kwargs[key] = _build(DriverSpec, value)
        elif key == "pinna":
            kwargs[key] = _build(PinnaSpec, value)
        elif key == "filters":
            kwargs[key] = tuple(_build(FilterSpec, v) for v in value)
        elif key == "rings":
            kwargs[key] = tuple((float(a), float(b)) for a, b in value)
        elif key == "amts_plug_cells" and value is not None:
            kwargs[key] = tuple((int(a), int(b)) for a, b in value)
        elif key in ("extra_rotation_deg", "canal_hint") and value is not None:
            kwargs[key] = tuple(float(v) for v in value)
        elif f.type in ("float", "float | None") and value is not None:
            kwargs[key] = float(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "name" not in data:
        raise ValueError(f"{path}: expected a mapping with at least a 'name' key")
    return _build(ExperimentConfig, data)  # type: ignore[no-any-return]


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return dataclasses.asdict(config)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config_to_dict(config), fh, sort_keys=False)
