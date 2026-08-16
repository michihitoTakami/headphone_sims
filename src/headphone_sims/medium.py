"""Acoustic medium properties (air) and simple loss models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Medium:
    """Homogeneous background medium (air at ~20 degC by default)."""

    sound_speed: float = 343.0  # m/s
    density: float = 1.204  # kg/m^3

    @property
    def bulk_modulus(self) -> float:
        return self.density * self.sound_speed**2


AIR = Medium()


def flow_resistivity_to_sigma(flow_resistivity: float, medium: Medium = AIR) -> float:
    """Convert flow resistivity (Pa*s/m^2) of a porous damping material to a
    first-order equivalent-fluid bulk absorption rate sigma (1/s).

    This is a deliberately simple model: momentum loss term dv/dt += -(phi/rho) * v
    with phi the flow resistivity. Valid as a qualitative damping knob, not as a
    quantitative porous-media model (no tortuosity/thermal effects).
    """
    return flow_resistivity / medium.density
