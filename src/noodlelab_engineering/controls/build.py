"""Making systems: transfer functions, zeros and poles, state space, PID
controllers, standard first- and second-order systems, and structures of
masses, springs and dampers."""

from __future__ import annotations

import re
from typing import Annotated, Any

import numpy as np
from numpy.typing import NDArray

from noodlelab import Param, Quantity, node
from noodlelab.core.units import is_quantity, ureg

from .types import Coefficients, System, poly, proper_problem

__all__ = [
    "first_order_system",
    "mass_spring_damper",
    "pid_controller",
    "second_order_system",
    "state_space",
    "transfer_function",
    "transfer_function_expression",
    "zero_pole_gain",
]

Seconds = Quantity["s"]
RadPerSecond = Quantity["rad/s"]
Matrix = Annotated[
    Quantity | NDArray[np.floating],
    Param(description="A matrix: a quantity holding a 2-D array, or plain numbers in SI units"),
]
OptionalMatrix = Annotated[
    Quantity | NDArray[np.floating] | None,
    Param(description="A matrix: a quantity holding a 2-D array, or plain numbers in SI units"),
]
ZERO_S = Quantity(0.0, "s")


@node(category="Engineering/Controls", title="Transfer Function", fold=True)
def transfer_function(
    numerator: Coefficients = "1",
    denominator: Coefficients = "1, 0.8, 1",
    sample_time: Annotated[Seconds, Param(description="0 s: continuous (s); else discrete (z)")] = (
        ZERO_S
    ),
) -> System:
    """A transfer function G(s) = num(s) / den(s), typed as polynomial
    coefficients in descending powers of s: ``1, 2, 1`` is s² + 2s + 1. With
    a sample time it is a discrete G(z) instead."""
    import control

    dt = sample_time.m_as("s")
    return control.tf(poly(numerator), poly(denominator), dt if dt > 0 else 0)


@transfer_function.check
def _check_transfer_function(
    numerator: str = "1", denominator: str = "1, 0.8, 1", sample_time: Any = None
) -> str | None:
    if sample_time is not None and sample_time.magnitude < 0:
        return "The sample time cannot be negative"
    return proper_problem(numerator, denominator)


def _roots(text: str) -> list[complex]:
    """``"-1, -2 ± 3j"`` as roots: a ± bj stands for the pair a + bj, a − bj."""
    out: list[complex] = []
    for token in (t.strip() for t in text.split(",")):
        if not token:
            continue
        token = token.replace(" ", "").replace("i", "j")
        pair = re.fullmatch(r"(.*?)(?:±|\+-|\+/-)(.*)", token)
        try:
            if pair:
                re_, im = float(pair.group(1) or 0), complex(pair.group(2))
                im = im.imag if im.imag else im.real  # "± 3j" or "± 3"
                out += [complex(re_, im), complex(re_, -im)]
            else:
                out.append(complex(token))
        except ValueError:
            raise ValueError(f"Not a number: '{token}'") from None
    return out


def _real_poly(roots: list[complex], name: str) -> NDArray[np.float64]:
    coeffs = np.poly(roots) if roots else np.array([1.0])
    if np.max(np.abs(np.imag(coeffs)), initial=0.0) > 1e-9 * np.max(np.abs(coeffs)):
        raise ValueError(f"Complex {name} come in pairs: give a ± bj, or both a + bj and a - bj")
    return np.real(coeffs)


@node(category="Engineering/Controls", title="Zero-Pole-Gain", fold=True)
def zero_pole_gain(
    zeros: Annotated[str, Param(description="Comma separated; empty: none")] = "",
    poles: Annotated[str, Param(description="Comma separated: -1, -2 ± 3j")] = "-1, -2",
    gain: float = 1.0,
) -> System:
    """A transfer function from its zeros, poles and gain:
    G(s) = k (s − z₁)(s − z₂)… / ((s − p₁)(s − p₂)…). ``a ± bj`` is a
    complex pair."""
    import control

    num = gain * _real_poly(_roots(zeros), "zeros")
    return control.tf(num, _real_poly(_roots(poles), "poles"))


@zero_pole_gain.check
def _check_zero_pole_gain(zeros: str = "", poles: str = "-1, -2") -> str | None:
    try:
        z, p = _roots(zeros), _roots(poles)
        _real_poly(z, "zeros")
        _real_poly(p, "poles")
    except ValueError as exc:
        return str(exc)
    if len(z) > len(p):
        return "More zeros than poles: the system is not proper"
    return None


def _si(value: Any, name: str, unit: str | None = None) -> NDArray[np.float64]:
    """A matrix input as numbers: in ``unit`` when given, else in SI base units."""
    from noodlelab.nodes.maths.linalg import _split

    arr, u = _split(value, name)
    if unit is not None and is_quantity(value) and not ureg().Quantity(1, u).dimensionless:
        try:
            return arr * ureg().Quantity(1, u).m_as(unit)
        except Exception:
            raise ValueError(f"{name} is in {u:~P}, which is not {unit}") from None
    factor = ureg().Quantity(1, u).to_base_units().magnitude
    return arr * factor


@node(category="Engineering/Controls", title="State Space")
def state_space(
    A: Matrix,
    B: Matrix,
    C: Matrix,
    D: OptionalMatrix = None,
    sample_time: Annotated[Seconds, Param(description="0 s: continuous; else discrete")] = ZERO_S,
) -> System:
    """A system in state space: x' = A x + B u, y = C x + D u. A is n×n, B
    n×m, C p×n and D p×m (zero when not linked). A in 1/time is taken in 1/s;
    the others in consistent SI units."""
    import control

    a = _si(A, "A", "1/s")
    b = _si(B, "B")
    c = _si(C, "C")
    for name, arr in (("A", a), ("B", b), ("C", c)):
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a matrix")
    n = a.shape[0]
    if a.shape != (n, n):
        raise ValueError(f"A must be square; it is {a.shape[0]}×{a.shape[1]}")
    if b.shape[0] != n:
        raise ValueError(f"B has {b.shape[0]} rows; it needs one per state ({n})")
    if c.shape[1] != n:
        raise ValueError(f"C has {c.shape[1]} columns; it needs one per state ({n})")
    d = np.zeros((c.shape[0], b.shape[1])) if D is None else _si(D, "D")
    if d.shape != (c.shape[0], b.shape[1]):
        raise ValueError(f"D must be {c.shape[0]}×{b.shape[1]} (outputs × inputs)")
    dt = sample_time.m_as("s")
    return control.ss(a, b, c, d, dt if dt > 0 else 0)


@node(category="Engineering/Controls", title="PID Controller", fold=True)
def pid_controller(
    kp: Annotated[float, Param(label="Kp", description="Proportional gain")] = 1.0,
    ki: Annotated[float, Param(label="Ki", description="Integral gain, per second")] = 0.0,
    kd: Annotated[float, Param(label="Kd", description="Derivative gain, in seconds")] = 0.0,
    derivative_filter: Annotated[
        Seconds, Param(description="Tf: the derivative acts through 1/(Tf s + 1)")
    ] = Quantity(0.01, "s"),  # noqa: B008 - immutable
) -> System:
    """A PID controller, C(s) = Kp + Ki/s + Kd s/(Tf s + 1). The derivative
    is filtered, as any real one is: without the filter it would be improper,
    amplifying noise without limit. Gains are in consistent SI units."""
    import control

    tf = derivative_filter.m_as("s")
    if kd and tf <= 0:
        raise ValueError("A derivative needs a filter time Tf > 0")
    if kd:
        num, den = [kp * tf + kd, kp + ki * tf, ki], [tf, 1.0, 0.0]
        if not ki:
            num, den = [kp * tf + kd, kp], [tf, 1.0]
    elif ki:
        num, den = [kp, ki], [1.0, 0.0]
    else:
        num, den = [kp], [1.0]
    if not any(num):
        raise ValueError("All three gains are zero")
    return control.tf(num, den)


@pid_controller.check
def _check_pid(kp: float = 1.0, ki: float = 0.0, kd: float = 0.0, derivative_filter: Any = None):
    if kd and derivative_filter is not None and derivative_filter.magnitude <= 0:
        return "A derivative needs a filter time Tf > 0"
    if not (kp or ki or kd):
        return "All three gains are zero"
    return None


@node(category="Engineering/Controls", title="Second-Order System", fold=True)
def second_order_system(
    natural_frequency: RadPerSecond = Quantity(1.0, "rad/s"),  # noqa: B008 - immutable
    damping_ratio: Annotated[float, Param(min=0.0, description="ζ: 1 is critical")] = 0.5,
    gain: float = 1.0,
) -> System:
    """The standard second-order system, K ωn² / (s² + 2ζωn s + ωn²): a mass
    on a spring and damper, an RLC circuit. Below ζ = 1 it overshoots."""
    import control

    wn = natural_frequency.m_as("rad/s")
    return control.tf([gain * wn**2], [1.0, 2 * damping_ratio * wn, wn**2])


@node(category="Engineering/Controls", title="First-Order System", fold=True)
def first_order_system(
    gain: float = 1.0,
    time_constant: Seconds = Quantity(1.0, "s"),  # noqa: B008 - immutable
    delay: Annotated[Seconds, Param(description="A dead time, as a Padé approximation")] = ZERO_S,
    pade_order: Annotated[int, Param(min=1, max=10)] = 3,
) -> System:
    """K / (τ s + 1): a lag such as a heater, a tank or an actuator, reaching
    63 % of its final value after one time constant. A delay is approximated
    by a Padé filter of ``pade_order``."""
    import control

    tau = time_constant.m_as("s")
    sys = control.tf([gain], [tau, 1.0]) if tau > 0 else control.tf([gain], [1.0])
    if (td := delay.m_as("s")) > 0:
        sys = sys * control.tf(*control.pade(td, pade_order))
    return sys


def _dofs(text: str, n: int, what: str) -> list[int]:
    out = []
    for token in (t.strip() for t in text.split(",")):
        if not token:
            continue
        if not token.isdigit() or not 1 <= int(token) <= n:
            raise ValueError(f"{what} '{token}': give a degree of freedom from 1 to {n}")
        out.append(int(token))
    if not out:
        raise ValueError(f"Give at least one {what}")
    return out


_MEASURE = re.compile(r"([xva])(\d+)")


@node(category="Engineering/Controls", title="Mass-Spring-Damper")
def mass_spring_damper(
    mass: Matrix,
    stiffness: Matrix,
    damping: OptionalMatrix = None,
    force_at: Annotated[
        str, Param(description="Where forces act, by degree of freedom: 1, or 1, 2")
    ] = "1",
    measure: Annotated[
        str, Param(description="The outputs: x1 (position), v1 (velocity), a1 (acceleration)")
    ] = "x1",
) -> System:
    """The state-space model of masses, springs and dampers,
    M x'' + C x' + K x = F, for Step Response, Feedback and the rest. The
    states are the positions x1… and velocities v1…, the inputs the forces
    F1… at ``force_at``, and the outputs as ``measure`` says. M in kg, K in
    N/m and C in N·s/m (other units are converted; plain numbers are SI)."""
    import control

    from noodlelab.nodes.maths.linalg import _split

    m = _si(mass, "mass", "kg")
    k = _si(stiffness, "stiffness", "N/m")
    n = k.shape[0]
    if m.ndim == 1:
        m = np.diag(m)
    c = np.zeros((n, n)) if damping is None else _si(damping, "damping", "N*s/m")
    for name, arr in (("mass", m), ("stiffness", k), ("damping", c)):
        if arr.shape != (n, n):
            raise ValueError(f"{name} must be {n}×{n}, like the stiffness matrix")
    _split(stiffness, "stiffness", square=True)
    minv = np.linalg.inv(m)
    zero, one = np.zeros((n, n)), np.eye(n)
    a = np.block([[zero, one], [-minv @ k, -minv @ c]])
    forces = _dofs(force_at, n, "force")
    select = np.zeros((n, len(forces)))
    for j, dof in enumerate(forces):
        select[dof - 1, j] = 1.0
    b = np.vstack([np.zeros((n, len(forces))), minv @ select])
    rows, drows, names = [], [], []
    for token in (t.strip() for t in measure.split(",")):
        if not token:
            continue
        found = _MEASURE.fullmatch(token)
        if not found or not 1 <= int(found.group(2)) <= n:
            raise ValueError(f"Output '{token}': x1…x{n}, v1…v{n} or a1…a{n}")
        kind, i = found.group(1), int(found.group(2)) - 1
        if kind == "x":
            rows.append(one[i] @ np.hstack([one, zero]))
            drows.append(np.zeros(len(forces)))
        elif kind == "v":
            rows.append(one[i] @ np.hstack([zero, one]))
            drows.append(np.zeros(len(forces)))
        else:  # the acceleration: the lower rows of A and B
            rows.append(a[n + i])
            drows.append(b[n + i])
        names.append(token)
    if not rows:
        raise ValueError("Give at least one output: x1")
    return control.ss(
        a,
        b,
        np.array(rows),
        np.array(drows),
        states=[f"x{i + 1}" for i in range(n)] + [f"v{i + 1}" for i in range(n)],
        inputs=[f"F{d}" for d in forces],
        outputs=names,
    )


@node(category="Engineering/Controls", title="Transfer Function (Expression)", fold=True)
def transfer_function_expression(
    text: Annotated[
        str, Param(description="A function of s: 10/(s*(s + 2)), K/(tau*s + 1)")
    ] = "10/(s*(s + 2))",
    values: Annotated[
        Any, Param(type=("noodlelab.nodes.symbolic.types.SymbolValues", type(None)))
    ] = None,
    variable: str = "s",
) -> System:
    """A transfer function typed as an expression in s, with other symbols
    taken from ``values`` (as plain numbers in SI units), such as
    ``K/(tau*s + 1)`` with K and tau from a Values node."""
    import control
    import sympy as sp

    from noodlelab.nodes.symbolic.parse import parse_expression

    expr = parse_expression(text)
    s = sp.Symbol(variable)
    subs = {}
    for name, v in (values or {}).items():
        subs[sp.Symbol(name)] = float(v.to_base_units().magnitude if is_quantity(v) else v)
    expr = sp.together(expr.subs(subs))
    left = sorted(str(x) for x in expr.free_symbols if x != s)
    if left:
        raise ValueError(f"No value for {', '.join(left)}: link Values")
    num, den = sp.fraction(expr)
    try:
        num_c = [float(c) for c in sp.Poly(num, s).all_coeffs()]
        den_c = [float(c) for c in sp.Poly(den, s).all_coeffs()]
    except (sp.PolynomialError, TypeError):
        raise ValueError(f"Not a ratio of polynomials in {variable}") from None
    if len(num_c) > len(den_c):
        raise ValueError("The numerator's order is higher than the denominator's (not proper)")
    return control.tf(num_c, den_c)
