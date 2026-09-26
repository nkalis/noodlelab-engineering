"""Typical properties of common engineering materials.

The values are representative room-temperature figures for preliminary
design and trade studies, not design allowables: check the supplier's
datasheet or a standard (e.g. MMPDS, EN 1993) before relying on them.
Composites are quasi-isotropic laminates, and their "yield" is a typical
first-ply failure strength.
"""

from __future__ import annotations

from typing import Any, Literal, NamedTuple

import pandas as pd

from noodlelab import Quantity, node

__all__ = ["material", "material_table"]

# name: E (GPa), Poisson's ratio, density (kg/m³), yield (MPa), ultimate (MPa),
# thermal expansion (µm/(m·K)), thermal conductivity (W/(m·K))
MATERIALS: dict[str, tuple[float, float, float, float, float, float, float]] = {
    "Steel S275": (210.0, 0.30, 7850.0, 275.0, 430.0, 12.0, 50.0),
    "Steel S355": (210.0, 0.30, 7850.0, 355.0, 510.0, 12.0, 50.0),
    "Stainless steel 304": (193.0, 0.29, 8000.0, 215.0, 505.0, 17.3, 16.2),
    "Stainless steel 316": (193.0, 0.30, 8000.0, 205.0, 515.0, 16.0, 16.3),
    "Aluminium 6061-T6": (68.9, 0.33, 2700.0, 276.0, 310.0, 23.6, 167.0),
    "Aluminium 7075-T6": (71.7, 0.33, 2810.0, 503.0, 572.0, 23.4, 130.0),
    "Titanium Ti-6Al-4V": (113.8, 0.34, 4430.0, 880.0, 950.0, 8.6, 6.7),
    "Copper C11000": (115.0, 0.31, 8940.0, 69.0, 220.0, 17.0, 388.0),
    "Brass C36000": (97.0, 0.31, 8500.0, 124.0, 338.0, 20.5, 115.0),
    "Magnesium AZ31B": (45.0, 0.35, 1770.0, 200.0, 260.0, 26.0, 96.0),
    "CFRP (quasi-isotropic)": (60.0, 0.30, 1600.0, 500.0, 600.0, 2.0, 5.0),
    "GFRP (quasi-isotropic)": (20.0, 0.28, 1900.0, 200.0, 250.0, 15.0, 0.4),
    "PEEK": (3.6, 0.40, 1320.0, 100.0, 100.0, 47.0, 0.25),
    "Nylon 6/6": (3.0, 0.39, 1140.0, 70.0, 85.0, 80.0, 0.25),
    "Polycarbonate": (2.4, 0.37, 1200.0, 62.0, 70.0, 68.0, 0.20),
    "ABS": (2.3, 0.35, 1050.0, 40.0, 44.0, 90.0, 0.17),
}

MaterialName = Literal[
    "Steel S275",
    "Steel S355",
    "Stainless steel 304",
    "Stainless steel 316",
    "Aluminium 6061-T6",
    "Aluminium 7075-T6",
    "Titanium Ti-6Al-4V",
    "Copper C11000",
    "Brass C36000",
    "Magnesium AZ31B",
    "CFRP (quasi-isotropic)",
    "GFRP (quasi-isotropic)",
    "PEEK",
    "Nylon 6/6",
    "Polycarbonate",
    "ABS",
]

GPa = Quantity["GPa"]
MPa = Quantity["MPa"]
Density = Quantity["kg/m^3"]
Expansion = Quantity["1/K"]
Conductivity = Quantity["W/(m*K)"]


class Material(NamedTuple):
    youngs_modulus: GPa
    poisson_ratio: float
    density: Density
    yield_strength: MPa
    ultimate_strength: MPa
    thermal_expansion: Expansion
    thermal_conductivity: Conductivity
    properties: dict[str, Any]


@node(category="Engineering/Materials", title="Material", fold=True)
def material(name: MaterialName = "Aluminium 6061-T6") -> Material:
    """Typical properties of a common material: stiffness, density, strength,
    thermal expansion and conductivity. For preliminary design: check a
    datasheet before relying on the strengths."""
    e, nu, rho, sy, su, alpha, k = MATERIALS[name]
    return Material(
        Quantity(e, "GPa"),
        nu,
        Quantity(rho, "kg/m^3"),
        Quantity(sy, "MPa"),
        Quantity(su, "MPa"),
        Quantity(alpha * 1e-6, "1/K"),
        Quantity(k, "W/(m*K)"),
        {
            "Material": name,
            "Young's modulus (GPa)": e,
            "Poisson's ratio": nu,
            "Density (kg/m³)": rho,
            "Yield strength (MPa)": sy,
            "Ultimate strength (MPa)": su,
            "Thermal expansion (µm/(m·K))": alpha,
            "Thermal conductivity (W/(m·K))": k,
        },
    )


@node(category="Engineering/Materials", title="Material Table", fold=True)
def material_table() -> pd.DataFrame:
    """Every built-in material as a table, with specific stiffness and
    strength (per unit density), for filtering, plotting or a Decision
    Matrix."""
    rows = []
    for name, (e, nu, rho, sy, su, alpha, k) in MATERIALS.items():
        rows.append(
            {
                "material": name,
                "E_GPa": e,
                "poisson": nu,
                "density_kg_m3": rho,
                "yield_MPa": sy,
                "ultimate_MPa": su,
                "expansion_um_mK": alpha,
                "conductivity_W_mK": k,
                "specific_stiffness_MNm_kg": e * 1e3 / rho,
                "specific_strength_kNm_kg": sy * 1e3 / rho,
            }
        )
    return pd.DataFrame(rows)
