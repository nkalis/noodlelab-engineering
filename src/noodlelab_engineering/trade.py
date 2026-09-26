"""Trade studies: score options against weighted criteria."""

from __future__ import annotations

from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from noodlelab import Param, node

__all__ = ["decision_matrix"]


class Decision(NamedTuple):
    ranking: pd.DataFrame
    best: str
    plot: Figure


def _criteria(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in text.replace("\n", ",").split(","):
        if not part.strip():
            continue
        name, _, weight = part.rpartition(":")
        if not name.strip():
            raise ValueError(f"Expected column:weight, got '{part.strip()}'")
        try:
            out[name.strip()] = float(weight)
        except ValueError:
            raise ValueError(f"The weight of '{name.strip()}' is not a number") from None
    if not out:
        raise ValueError("Enter at least one criterion, e.g. cost:-2, performance:3")
    return out


@node(category="Engineering/Trade", title="Decision Matrix")
def decision_matrix(
    table: pd.DataFrame,
    option: Annotated[str, Param(options_from="table.columns")] = "",
    criteria: Annotated[
        str,
        Param(
            multiline=True,
            description="column:weight, comma separated; a negative weight: lower is better",
        ),
    ] = "",
) -> Decision:
    """Rank options (one per row) by a weighted sum of criteria. Each
    criterion column is scaled from 0 (worst option) to 1 (best), so units
    do not matter, then weighted by the magnitude of its weight. The score
    is out of 100."""
    weights = _criteria(criteria)
    missing = [c for c in weights if c not in table.columns]
    if missing:
        raise KeyError(f"No column {', '.join(map(repr, missing))}")
    names = table[option].astype(str) if option else table.index.astype(str)
    total = sum(abs(w) for w in weights.values()) or 1.0
    out = pd.DataFrame({"option": names.to_numpy()})
    score = np.zeros(len(table))
    for col, w in weights.items():
        v = pd.to_numeric(table[col], errors="raise").to_numpy(np.float64)
        span = np.nanmax(v) - np.nanmin(v)
        s = (v - np.nanmin(v)) / span if span else np.ones_like(v)
        if w < 0:
            s = 1.0 - s
        out[f"{col} (score)"] = s
        score += abs(w) * s
    out["score"] = 100.0 * score / total
    out = out.sort_values("score", ascending=False, ignore_index=True)
    out.insert(1, "rank", np.arange(1, len(out) + 1))
    fig = Figure(figsize=(6.0, 0.4 * len(out) + 1.2), layout="constrained")
    ax = fig.add_subplot()
    ax.barh(out["option"][::-1], out["score"][::-1], color="#4c72b0")
    ax.set(xlabel="score (out of 100)", title="Decision matrix", xlim=(0, 100))
    ax.grid(alpha=0.3, axis="x")
    return Decision(out, str(out["option"].iloc[0]) if len(out) else "", fig)


@decision_matrix.check
def _check_decision(criteria: str = ""):
    try:
        _criteria(criteria)
    except ValueError as exc:
        return str(exc)
    return None
