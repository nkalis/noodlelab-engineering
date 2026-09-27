"""Decibels: conversions and gain/loss budgets (link budgets, noise budgets,
signal chains), where adding decibels multiplies ratios."""

from __future__ import annotations

import math
import re
from typing import Annotated, Literal, NamedTuple

import numpy as np
import pandas as pd

from noodlelab import Param, node

__all__ = ["db_budget", "from_db", "to_db"]

Kind = Annotated[
    Literal["power", "amplitude"],
    Param(description="power: 10 log10; amplitude (voltage, pressure): 20 log10"),
]


@node(category="Engineering/Decibels", title="To dB", fold=True, vectorized=True)
def to_db(ratio: float, kind: Kind = "power") -> float:
    """A ratio in decibels: 10 log10 of a power ratio, 20 log10 of an
    amplitude ratio. Use a ratio to 1 mW for dBm, to 1 W for dBW."""
    factor = 10.0 if kind == "power" else 20.0
    if isinstance(ratio, np.ndarray):  # a batch of Monte Carlo trials, as the scalar case
        if (ratio <= 0).any():
            raise ValueError("Only a positive ratio has a value in dB")
        return factor * np.log10(ratio)
    if ratio <= 0:
        raise ValueError("Only a positive ratio has a value in dB")
    return factor * math.log10(ratio)


@node(category="Engineering/Decibels", title="From dB", fold=True, vectorized=True)
def from_db(db: float, kind: Kind = "power") -> float:
    """The ratio a value in decibels stands for."""
    return 10 ** (db / (10.0 if kind == "power" else 20.0))


class Budget(NamedTuple):
    total: float
    table: pd.DataFrame


_NUMBER = r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?"
_LINE = re.compile(rf"^(?P<name>.*?)[\s:=]+(?P<value>{_NUMBER})\s*(?:dB\w*)?$")


def _items(text: str) -> list[tuple[str, float]]:
    out = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _LINE.match(line)
        if m is None:
            raise ValueError(f"Line {n}: expected a name and a value in dB, got '{raw.strip()}'")
        out.append((m["name"].strip() or f"Item {n}", float(m["value"])))
    return out


@node(category="Engineering/Decibels", title="dB Budget", fold=True)
def db_budget(
    items: Annotated[
        str,
        Param(multiline=True, description="One per line: name and value in dB (losses negative)"),
    ] = "Transmit power 30 dBm\nCable loss -2\nAntenna gain 12\nPath loss -120",
    extra: Annotated[float, Param(description="Added to the total, e.g. from another node")] = 0.0,
) -> Budget:
    """Add up gains and losses in dB, one per line (``#`` starts a comment).
    The table has the running total after each item, for a report."""
    rows = _items(items)
    if extra:
        rows.append(("Extra", extra))
    running, table = 0.0, []
    for name, value in rows:
        running += value
        table.append({"item": name, "dB": value, "running total dB": running})
    return Budget(running, pd.DataFrame(table, columns=["item", "dB", "running total dB"]))


@db_budget.check
def _check_budget(items: str = ""):
    try:
        _items(items)
    except ValueError as exc:
        return str(exc)
    return None
