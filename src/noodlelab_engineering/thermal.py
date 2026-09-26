"""Heat: thermal expansion, one-dimensional conduction and convection
(steady state, constant properties). Temperature differences are in kelvin,
which equal degrees Celsius of difference."""

from __future__ import annotations

import math
from typing import NamedTuple

from noodlelab import Quantity, node

__all__ = ["convection", "pipe_conduction", "thermal_expansion", "wall_conduction"]

M = Quantity["m"]
Mm = Quantity["mm"]
M2 = Quantity["m^2"]
K = Quantity["K"]
W = Quantity["W"]
Expansion = Quantity["1/K"]
Conductivity = Quantity["W/(m*K)"]
HTC = Quantity["W/(m^2*K)"]
Resistance = Quantity["K/W"]


class HeatFlow(NamedTuple):
    heat_flow: W
    resistance: Resistance


@node(category="Engineering/Thermal", title="Thermal Expansion", fold=True, vectorized=True)
def thermal_expansion(
    length: M,
    expansion: Expansion = Quantity(12e-6, "1/K"),  # noqa: B008 - immutable
    temperature_change: K = Quantity(50.0, "K"),  # noqa: B008
) -> Mm:
    """The change in length of a free bar, ΔL = α L ΔT."""
    return (expansion * length * temperature_change).to("mm")


@node(category="Engineering/Thermal", title="Wall Conduction", fold=True)
def wall_conduction(
    conductivity: Conductivity,
    thickness: Mm = Quantity(100.0, "mm"),  # noqa: B008 - immutable
    area: M2 = Quantity(1.0, "m^2"),  # noqa: B008
    temperature_difference: K = Quantity(20.0, "K"),  # noqa: B008
) -> HeatFlow:
    """Heat flow through a plane wall, Q = k A ΔT / t, and its thermal
    resistance t / (k A) (add resistances in series for layered walls)."""
    r = (thickness / (conductivity * area)).to("K/W")
    return HeatFlow((temperature_difference / r).to("W"), r)


@node(category="Engineering/Thermal", title="Pipe Conduction", fold=True)
def pipe_conduction(
    conductivity: Conductivity,
    inner_radius: Mm = Quantity(25.0, "mm"),  # noqa: B008 - immutable
    outer_radius: Mm = Quantity(50.0, "mm"),  # noqa: B008
    length: M = Quantity(1.0, "m"),  # noqa: B008
    temperature_difference: K = Quantity(20.0, "K"),  # noqa: B008
) -> HeatFlow:
    """Radial heat flow through a pipe wall or insulation layer,
    Q = 2π k L ΔT / ln(r₂ / r₁)."""
    ratio = outer_radius.m_as("mm") / inner_radius.m_as("mm")
    if ratio <= 1:
        raise ValueError("The outer radius must be larger than the inner radius")
    r = (Quantity(math.log(ratio), "") / (2 * math.pi * conductivity * length)).to("K/W")
    return HeatFlow((temperature_difference / r).to("W"), r)


@node(category="Engineering/Thermal", title="Convection", fold=True)
def convection(
    coefficient: HTC = Quantity(10.0, "W/(m^2*K)"),  # noqa: B008 - immutable
    area: M2 = Quantity(1.0, "m^2"),  # noqa: B008
    temperature_difference: K = Quantity(20.0, "K"),  # noqa: B008
) -> HeatFlow:
    """Heat flow from a surface to a fluid, Q = h A ΔT (Newton's law of
    cooling). Typical h: 5–25 W/(m²·K) still air, 10–200 forced air,
    500–10 000 forced water."""
    r = (1 / (coefficient * area)).to("K/W")
    return HeatFlow((temperature_difference / r).to("W"), r)
