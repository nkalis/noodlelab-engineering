"""Section properties of common cross-sections.

``x`` is the horizontal axis through the centroid, so ``ixx`` resists bending
under a vertical load (the strong axis of an upright I-beam).
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

from noodlelab import Quantity, node

__all__ = ["section_properties"]

Mm = Quantity["mm"]
Mm2 = Quantity["mm^2"]
Mm3 = Quantity["mm^3"]
Mm4 = Quantity["mm^4"]

Shape = Literal["rectangle", "hollow rectangle", "circle", "tube", "I-beam"]


class Section(NamedTuple):
    area: Mm2
    ixx: Mm4
    iyy: Mm4
    section_modulus: Mm3
    radius_of_gyration: Mm
    summary: dict[str, float]


def _props(
    shape: str, b: float, h: float, t: float, tw: float
) -> tuple[float, float, float, float]:
    """Area, Ixx, Iyy and the extreme-fibre distance from the x axis, in mm."""
    if shape == "rectangle":
        return b * h, b * h**3 / 12, h * b**3 / 12, h / 2
    if shape == "hollow rectangle":
        bi, hi = b - 2 * t, h - 2 * t
        return (
            b * h - bi * hi,
            (b * h**3 - bi * hi**3) / 12,
            (h * b**3 - hi * bi**3) / 12,
            h / 2,
        )
    if shape == "circle":
        return math.pi * b**2 / 4, math.pi * b**4 / 64, math.pi * b**4 / 64, b / 2
    if shape == "tube":
        di = b - 2 * t
        i = math.pi * (b**4 - di**4) / 64
        return math.pi * (b**2 - di**2) / 4, i, i, b / 2
    # I-beam: flanges b × t, web tw thick between them
    hw = h - 2 * t
    return (
        2 * b * t + hw * tw,
        (b * h**3 - (b - tw) * hw**3) / 12,
        2 * t * b**3 / 12 + hw * tw**3 / 12,
        h / 2,
    )


@node(category="Engineering/Sections", title="Section Properties", fold=True)
def section_properties(
    shape: Shape = "rectangle",
    width: Mm = Quantity(50.0, "mm"),  # noqa: B008 - immutable
    height: Mm = Quantity(100.0, "mm"),  # noqa: B008
    wall: Mm = Quantity(5.0, "mm"),  # noqa: B008
    web: Mm = Quantity(5.0, "mm"),  # noqa: B008
) -> Section:
    """Area, second moments of area, elastic section modulus (about x) and
    least radius of gyration.

    * rectangle: ``width`` × ``height``
    * hollow rectangle: outside ``width`` × ``height``, ``wall`` thick
    * circle: diameter ``width``
    * tube: outside diameter ``width``, ``wall`` thick
    * I-beam: flanges ``width`` wide and ``wall`` thick, overall ``height``,
      ``web`` thick
    """
    b, h, t, tw = (q.m_as("mm") for q in (width, height, wall, web))
    area, ixx, iyy, c = _props(shape, b, h, t, tw)
    r = math.sqrt(min(ixx, iyy) / area)
    return Section(
        Quantity(area, "mm^2"),
        Quantity(ixx, "mm^4"),
        Quantity(iyy, "mm^4"),
        Quantity(ixx / c, "mm^3"),
        Quantity(r, "mm"),
        {
            "Area (mm²)": area,
            "Ixx (mm⁴)": ixx,
            "Iyy (mm⁴)": iyy,
            "Section modulus (mm³)": ixx / c,
            "Radius of gyration (mm)": r,
        },
    )


@section_properties.check
def _check_section(
    shape: str = "rectangle",
    width: Mm | None = None,
    height: Mm | None = None,
    wall: Mm | None = None,
    web: Mm | None = None,
):
    b = width.m_as("mm") if width is not None else 1.0
    h = height.m_as("mm") if height is not None else 1.0
    t = wall.m_as("mm") if wall is not None else 0.0
    tw = web.m_as("mm") if web is not None else 0.0
    if b <= 0 or h <= 0:
        return "Width and height must be positive"
    if shape == "hollow rectangle" and not 0 < t < min(b, h) / 2:
        return "The wall must be thinner than half the width and the height"
    if shape == "tube" and not 0 < t < b / 2:
        return "The wall must be thinner than half the diameter"
    if shape == "I-beam" and (not 0 < t < h / 2 or not 0 < tw <= b):
        return "The flanges must be thinner than half the height, the web no wider than them"
    return None
