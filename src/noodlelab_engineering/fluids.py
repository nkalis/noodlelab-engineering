"""Fluid flow: the Reynolds number and pressure drop in straight pipes
(Darcy–Weisbach, with the Haaland friction factor in turbulent flow)."""

from __future__ import annotations

import math
from typing import NamedTuple

from noodlelab import Quantity, node

__all__ = ["pipe_pressure_drop", "reynolds_number"]

M = Quantity["m"]
Mm = Quantity["mm"]
Speed = Quantity["m/s"]
Density = Quantity["kg/m^3"]
Viscosity = Quantity["Pa*s"]
Flow = Quantity["L/s"]
KPa = Quantity["kPa"]

WATER_DENSITY = Quantity(998.0, "kg/m^3")  # 20 °C
WATER_VISCOSITY = Quantity(1.0e-3, "Pa*s")


class Reynolds(NamedTuple):
    reynolds: float
    regime: str


def _regime(re: float) -> str:
    return "laminar" if re < 2300 else "transitional" if re < 4000 else "turbulent"


@node(category="Engineering/Fluids", title="Reynolds Number", fold=True)
def reynolds_number(
    velocity: Speed,
    length: M = Quantity(0.05, "m"),  # noqa: B008 - immutable
    density: Density = WATER_DENSITY,
    viscosity: Viscosity = WATER_VISCOSITY,
) -> Reynolds:
    """Re = ρ v L / μ, with ``length`` the pipe diameter (or a body's
    characteristic length), and the pipe-flow regime: laminar below 2300,
    turbulent above 4000. The defaults are water at 20 °C."""
    re = float((density * velocity * length / viscosity).to("dimensionless").magnitude)
    return Reynolds(re, _regime(re))


class PressureDrop(NamedTuple):
    pressure_drop: KPa
    velocity: Speed
    reynolds: float
    friction_factor: float
    regime: str


@node(category="Engineering/Fluids", title="Pipe Pressure Drop", fold=True)
def pipe_pressure_drop(
    flow_rate: Flow = Quantity(2.0, "L/s"),  # noqa: B008 - immutable
    diameter: Mm = Quantity(50.0, "mm"),  # noqa: B008
    length: M = Quantity(100.0, "m"),  # noqa: B008
    roughness: Mm = Quantity(0.045, "mm"),  # noqa: B008
    density: Density = WATER_DENSITY,
    viscosity: Viscosity = WATER_VISCOSITY,
) -> PressureDrop:
    """The friction pressure drop along a straight, full, circular pipe,
    Δp = f (L / D) ρ v² / 2. The Darcy friction factor f is 64 / Re in
    laminar flow and from the Haaland equation otherwise. Roughness: about
    0.0015 mm drawn tubing and plastic, 0.045 mm commercial steel, 0.26 mm
    cast iron."""
    d = diameter.m_as("m")
    rho = density.m_as("kg/m^3")
    mu = viscosity.m_as("Pa*s")
    v = flow_rate.m_as("m^3/s") / (math.pi * d**2 / 4)
    re = rho * v * d / mu
    if re <= 0:
        raise ValueError("The flow rate must be positive")
    if re < 2300:
        f = 64.0 / re
    else:
        eps = roughness.m_as("m") / d
        f = (-1.8 * math.log10((eps / 3.7) ** 1.11 + 6.9 / re)) ** -2
    dp = f * length.m_as("m") / d * rho * v**2 / 2
    return PressureDrop(Quantity(dp / 1e3, "kPa"), Quantity(v, "m/s"), re, f, _regime(re))
