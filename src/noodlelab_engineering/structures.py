"""Beams, stresses, buckling and safety factors: the closed-form formulas of
elementary strength of materials (linear elastic, small deflections)."""

from __future__ import annotations

import math
from typing import Annotated, Literal, NamedTuple

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from noodlelab import Param, Quantity, node

__all__ = [
    "axial_stress",
    "beam",
    "bending_stress",
    "euler_buckling",
    "safety_factor",
    "von_mises",
]

M = Quantity["m"]
Mm = Quantity["mm"]
Mm2 = Quantity["mm^2"]
Mm3 = Quantity["mm^3"]
Mm4 = Quantity["mm^4"]
N = Quantity["N"]
KN = Quantity["kN"]
Nm = Quantity["N*m"]
MPa = Quantity["MPa"]
GPa = Quantity["GPa"]

Support = Literal["simply supported", "cantilever", "fixed-fixed"]
Load = Literal["point", "uniform"]


class Beam(NamedTuple):
    max_deflection: Mm
    max_moment: Nm
    max_shear: N
    x: NDArray[np.float64]
    deflection: NDArray[np.float64]
    plot: Figure
    summary: dict[str, float]


# (deflection, moment, shear) coefficients: δ = a W L³ / (E I), M = b W L, V = c W
_MAX = {
    ("simply supported", "point"): (1 / 48, 1 / 4, 1 / 2),
    ("simply supported", "uniform"): (5 / 384, 1 / 8, 1 / 2),
    ("cantilever", "point"): (1 / 3, 1.0, 1.0),
    ("cantilever", "uniform"): (1 / 8, 1 / 2, 1.0),
    ("fixed-fixed", "point"): (1 / 192, 1 / 8, 1 / 2),
    ("fixed-fixed", "uniform"): (1 / 384, 1 / 12, 1 / 2),
}


def _curve(support: str, load: str, x: np.ndarray, L: float, W: float, EI: float) -> np.ndarray:
    """Deflection (positive down) along the beam, x from the left support
    (the fixed end of a cantilever)."""
    w = W / L
    if support == "simply supported":
        if load == "uniform":
            return w * x * (L**3 - 2 * L * x**2 + x**3) / (24 * EI)
        xs = np.minimum(x, L - x)
        return W * xs * (3 * L**2 - 4 * xs**2) / (48 * EI)
    if support == "cantilever":
        if load == "uniform":
            return w * x**2 * (6 * L**2 - 4 * L * x + x**2) / (24 * EI)
        return W * x**2 * (3 * L - x) / (6 * EI)
    if load == "uniform":
        return w * x**2 * (L - x) ** 2 / (24 * EI)
    xs = np.minimum(x, L - x)
    return W * xs**2 * (3 * L - 4 * xs) / (48 * EI)


@node(category="Engineering/Structures", title="Beam")
def beam(
    length: M = Quantity(2.0, "m"),  # noqa: B008 - immutable
    load: KN = Quantity(5.0, "kN"),  # noqa: B008
    youngs_modulus: GPa = Quantity(200.0, "GPa"),  # noqa: B008
    second_moment: Mm4 = Quantity(4.0e6, "mm^4"),  # noqa: B008
    support: Support = "simply supported",
    load_type: Annotated[Load, Param(description="uniform: the load spread over the span")] = (
        "point"
    ),
    points: Annotated[int, Param(min=11, max=2001)] = 101,
) -> Beam:
    """Maximum deflection, bending moment and shear of a prismatic beam, and
    its deflected shape. ``load`` is the total load: a point load at midspan
    (at the free end of a cantilever), or spread uniformly over the span.
    Fixed-fixed moments are the largest, at the supports."""
    L = length.m_as("m")
    W = load.m_as("N")
    EI = youngs_modulus.m_as("Pa") * second_moment.m_as("m^4")
    a, b, c = _MAX[(support, load_type)]
    delta = a * W * L**3 / EI
    x = np.linspace(0.0, L, points)
    y = _curve(support, load_type, x, L, W, EI) * 1e3
    fig = Figure(figsize=(6.0, 3.0), layout="constrained")
    ax = fig.add_subplot()
    ax.plot(x, -y, lw=2)
    ax.axhline(0.0, color="0.6", lw=1)
    ax.set(title=f"{support}, {load_type} load", xlabel="x (m)", ylabel="deflection (mm)")
    ax.grid(alpha=0.3)
    return Beam(
        Quantity(delta * 1e3, "mm"),
        Quantity(b * W * L, "N*m"),
        Quantity(c * W, "N"),
        x,
        y,
        fig,
        {
            "Max deflection (mm)": delta * 1e3,
            "Max bending moment (N·m)": b * W * L,
            "Max shear force (N)": c * W,
            "Span / deflection": L / delta if delta else math.inf,
        },
    )


@beam.check
def _check_beam(length: M | None = None, second_moment: Mm4 | None = None):
    if length is not None and length.magnitude <= 0:
        return "The length must be positive"
    if second_moment is not None and second_moment.magnitude <= 0:
        return "The second moment of area must be positive"
    return None


@node(category="Engineering/Structures", title="Bending Stress", fold=True, vectorized=True)
def bending_stress(moment: Nm, section_modulus: Mm3) -> MPa:
    """The largest bending stress in a section, σ = M / Z."""
    return (moment / section_modulus).to("MPa")


@node(category="Engineering/Structures", title="Axial Stress", fold=True, vectorized=True)
def axial_stress(force: N, area: Mm2) -> MPa:
    """Direct stress under an axial force, σ = F / A (tension positive)."""
    return (force / area).to("MPa")


@node(category="Engineering/Structures", title="Von Mises Stress", fold=True, vectorized=True)
def von_mises(
    sigma_x: MPa,
    sigma_y: MPa = Quantity(0.0, "MPa"),  # noqa: B008 - immutable
    tau_xy: MPa = Quantity(0.0, "MPa"),  # noqa: B008
) -> MPa:
    """The equivalent (von Mises) stress of a plane stress state, to compare
    with the yield strength: √(σx² − σx σy + σy² + 3 τxy²)."""
    sx, sy, t = (q.m_as("MPa") for q in (sigma_x, sigma_y, tau_xy))
    return Quantity(np.sqrt(sx**2 - sx * sy + sy**2 + 3 * t**2), "MPa")


EndCondition = Literal["pinned-pinned", "fixed-free", "fixed-pinned", "fixed-fixed"]
_K = {"pinned-pinned": 1.0, "fixed-free": 2.0, "fixed-pinned": 0.699, "fixed-fixed": 0.5}


class Buckling(NamedTuple):
    critical_load: KN
    critical_stress: MPa
    slenderness: float


@node(category="Engineering/Structures", title="Euler Buckling", fold=True)
def euler_buckling(
    youngs_modulus: GPa,
    second_moment: Mm4,
    length: M,
    area: Mm2 | None = None,
    ends: EndCondition = "pinned-pinned",
) -> Buckling:
    """The elastic buckling load of a slender column, P = π² E I / (K L)²,
    with the effective length factor K for the end conditions. Give the
    ``area`` for the critical stress and slenderness ratio K L / r; stocky
    columns (slenderness below about 100 in steel) yield before they buckle."""
    kl = _K[ends] * length.m_as("m")
    E = youngs_modulus.m_as("Pa")
    i = second_moment.m_as("m^4")
    p = math.pi**2 * E * i / kl**2
    if area is None:
        return Buckling(Quantity(p / 1e3, "kN"), Quantity(math.nan, "MPa"), math.nan)
    a = area.m_as("m^2")
    return Buckling(Quantity(p / 1e3, "kN"), Quantity(p / a / 1e6, "MPa"), kl / math.sqrt(i / a))


class Safety(NamedTuple):
    factor: float
    margin: float
    passes: bool


@node(category="Engineering/Structures", title="Safety Factor", fold=True)
def safety_factor(
    capacity: Quantity,
    demand: Quantity,
    required: Annotated[float, Param(min=0.0, description="The factor the design needs")] = 1.5,
) -> Safety:
    """How far a design is from failing: the factor of safety capacity /
    demand (e.g. yield strength / stress), and the margin of safety
    capacity / (required × demand) − 1, which passes when it is ≥ 0."""
    if capacity.dimensionality != demand.dimensionality:
        raise ValueError(f"Cannot compare {capacity.units:~} with {demand.units:~}")
    fos = float((capacity / demand).to("dimensionless").magnitude)
    margin = fos / required - 1.0 if required else math.inf
    return Safety(fos, margin, margin >= 0.0)


@safety_factor.check
def _check_safety(capacity: Quantity | None = None, demand: Quantity | None = None):
    if capacity is None or demand is None:
        return None
    if capacity.dimensionality != demand.dimensionality:
        return f"Capacity ({capacity.units:~}) and demand ({demand.units:~}) differ in dimension"
    return None
