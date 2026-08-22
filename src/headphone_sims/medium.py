"""Acoustic medium properties (air) and simple loss models."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Medium:
    """Homogeneous background medium (air at ~20 degC by default)."""

    sound_speed: float = 343.0  # m/s
    density: float = 1.204  # kg/m^3
    dynamic_viscosity: float = 1.81e-5  # Pa*s (air at ~20 degC)

    @property
    def bulk_modulus(self) -> float:
        return self.density * self.sound_speed**2


AIR = Medium()


def maa_tube_flow_resistivity(radius: float, frequency: float, medium: Medium = AIR) -> float:
    """Viscous flow resistivity (Pa*s/m^2, i.e. resistance per unit length of
    tube) of air oscillating in a circular tube of ``radius`` at ``frequency``.

    Maa's interpolation of Crandall's exact narrow-tube solution:
    ``Phi = (8 eta / a^2) * sqrt(1 + s^2/32)`` with the shear wavenumber
    ``s = a sqrt(rho omega / eta)``. Limits: Poiseuille ``8 eta / a^2`` for
    s << 1; boundary-layer ``sqrt(2 eta rho omega) / a`` for s >> 1 (all
    headphone grille/AMTS bores here sit at s ~ 25-60). Reference: D.-Y. Maa,
    "Potential of microperforated panel absorber", JASA 104 (1998).

    Only the VISCOUS resistance is modeled — no thermal boundary-layer loss
    and no reactive (attached-mass) correction — so this is a lower bound on
    the true thermoviscous damping. The FDTD applies it as a
    frequency-independent momentum sink; evaluate at a band-center frequency
    (Phi grows only as sqrt(f) in the high-s regime, so a single mid-band
    value stays within ~20% across 5-10 kHz).
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    omega = 2.0 * math.pi * frequency
    s2 = radius**2 * medium.density * omega / medium.dynamic_viscosity
    return (8.0 * medium.dynamic_viscosity / radius**2) * math.sqrt(1.0 + s2 / 32.0)


def flow_resistivity_to_sigma(flow_resistivity: float, medium: Medium = AIR) -> float:
    """Convert flow resistivity (Pa*s/m^2) of a porous damping material to a
    first-order equivalent-fluid bulk absorption rate sigma (1/s).

    This is a deliberately simple model: momentum loss term dv/dt += -(phi/rho) * v
    with phi the flow resistivity. Valid as a qualitative damping knob, not as a
    quantitative porous-media model (no tortuosity/thermal effects).
    """
    return flow_resistivity / medium.density
