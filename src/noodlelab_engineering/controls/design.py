"""Designing state feedback, u = −K x: by placing the closed-loop poles
where they should be, or as the linear-quadratic regulator that balances
how far the states stray against how hard the inputs work."""

from __future__ import annotations

from typing import Annotated, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from noodlelab import Param, node

from .build import OptionalMatrix, _roots, _si
from .types import System, continuous

__all__ = ["lqr", "pole_placement"]

DESIGN = "Engineering/Control Design"


class StateFeedback(NamedTuple):
    gain: NDArray[np.float64]
    closed_loop: System
    poles: NDArray[np.complex128]
    summary: dict[str, float]


def _closed(sys: Any, k: NDArray[np.float64]) -> Any:
    """The system with u = −K x + r: x' = (A − B K) x + B r."""
    import control

    return control.ss(sys.A - sys.B @ k, sys.B, sys.C - sys.D @ k, sys.D, sys.dt)


def _summary(k: NDArray[np.float64], poles: NDArray[np.complex128]) -> dict[str, float]:
    out = {f"K{i + 1}": float(v) for i, v in enumerate(np.ravel(k))}
    out["Slowest pole (real part)"] = float(np.max(poles.real)) if poles.size else float("nan")
    return out


@node(category=DESIGN, title="LQR")
def lqr(
    system: System,
    Q: Annotated[OptionalMatrix, Param(description="State weights; empty: the identity")] = None,
    R: Annotated[OptionalMatrix, Param(description="Input weights; empty: the identity")] = None,
) -> StateFeedback:
    """The linear-quadratic regulator: the state feedback u = −K x that
    minimises ∫ (xᵀQx + uᵀRu) dt. Larger Q entries hold those states closer;
    larger R entries spend less input. ``closed_loop`` is the system with the
    feedback in place, driven by a reference r added to the input."""
    import control

    sys = control.ss(system)
    n, m = sys.nstates, sys.ninputs
    q = np.eye(n) if Q is None else _si(Q, "Q")
    r = np.eye(m) if R is None else _si(R, "R")
    if q.shape != (n, n) or np.atleast_2d(r).shape != (m, m):
        raise ValueError(f"Q must be {n}×{n} (states) and R {m}×{m} (inputs)")
    design = control.lqr if continuous(sys) else control.dlqr
    k, _, poles = design(sys, q, np.atleast_2d(r))
    k = np.asarray(k, dtype=np.float64)
    poles = np.asarray(poles, dtype=np.complex128)
    return StateFeedback(k, _closed(sys, k), poles, _summary(k, poles))


@node(category=DESIGN, title="Pole Placement")
def pole_placement(
    system: System,
    poles: Annotated[
        str, Param(description="One per state, comma separated: -2, -3 ± 1j")
    ] = "-2, -3",
) -> StateFeedback:
    """The state feedback u = −K x that puts the closed-loop poles where you
    say: further left is faster, and a pair a ± bj has damping ratio
    −a/√(a² + b²). The system must be controllable."""
    import control

    sys = control.ss(system)
    wanted = _roots(poles)
    if len(wanted) != sys.nstates:
        raise ValueError(f"The system has {sys.nstates} states: give {sys.nstates} poles")
    if np.linalg.matrix_rank(control.ctrb(sys.A, sys.B)) < sys.nstates:
        raise ValueError("The system is not controllable: its poles cannot all be moved")
    k = np.asarray(control.place(sys.A, sys.B, wanted), dtype=np.float64)
    placed = np.asarray(np.linalg.eigvals(sys.A - sys.B @ k), dtype=np.complex128)
    return StateFeedback(k, _closed(sys, k), placed, _summary(k, placed))
