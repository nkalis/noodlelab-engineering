"""The SYSTEM socket: a python-control linear time-invariant system, as a
transfer function or in state space, with its preview, meta and checkpoint codec.

python-control takes about a second to import, so nothing here imports it at
module level: the type is referred to by its qualified name, the way optional
dependencies are, and the helpers import it when a node runs.
"""

from __future__ import annotations

import io
import json
import math
from typing import IO, Annotated, Any

import numpy as np

from noodlelab import Param, Preview, register_codec, register_meta, register_preview, register_type

LTI = "control.lti.LTI"
SYSTEM_HELP = "A linear system: from Transfer Function, State Space, PID Controller..."
System = Annotated[Any, Param(type=LTI, description=SYSTEM_HELP)]
OptionalSystem = Annotated[Any, Param(type=(LTI, type(None)), description=SYSTEM_HELP)]
Coefficients = Annotated[
    str, Param(description="Coefficients in descending powers of s, e.g. 1, 2, 1")
]
Signal = Annotated[str, Param(description="Its name; empty: the first")]

register_type(LTI, "SYSTEM", "#4db6ac", "A linear time-invariant system (python-control)")


# --- helpers shared by the nodes -------------------------------------------------------------


def poly(text: str) -> list[float]:
    """``"1, 0.8, 1"`` as coefficients, highest power first."""
    try:
        out = [float(c) for c in text.replace(";", ",").replace(" ", ",").split(",") if c]
    except ValueError:
        raise ValueError(f"Not a list of numbers: '{text}'") from None
    if not out or not any(out):
        raise ValueError("Enter at least one non-zero coefficient")
    return out


def proper_problem(numerator: str, denominator: str) -> str | None:
    try:
        num, den = poly(numerator), poly(denominator)
    except ValueError as exc:
        return str(exc)
    if len(np.trim_zeros(num, "f")) > len(np.trim_zeros(den, "f")):
        return "The numerator's order is higher than the denominator's (not proper)"
    return None


def continuous(sys: Any) -> bool:
    return not sys.isdtime(strict=True)


def stable(sys: Any) -> bool:
    """All poles in the left half plane, or inside the unit circle when discrete."""
    poles = np.asarray(sys.poles())
    if not poles.size:
        return True
    if continuous(sys):
        return bool(np.all(poles.real < 0))
    return bool(np.all(np.abs(poles) < 1))


def unstable_poles(sys: Any) -> int:
    """Poles in the right half plane (outside the unit circle), not counting
    those on the boundary within round-off, such as an integrator's."""
    poles = np.asarray(sys.poles())
    if not poles.size:
        return 0
    margin = 1e-9 * np.maximum(1.0, np.abs(poles))
    if continuous(sys):
        return int(np.sum(poles.real > margin))
    return int(np.sum(np.abs(poles) > 1 + margin))


def status(sys: Any) -> str:
    """Stable, marginally stable (poles on the boundary, such as an
    integrator's, and none beyond it) or unstable."""
    if stable(sys):
        return "stable"
    return "marginally stable" if unstable_poles(sys) == 0 else "unstable"


def select(sys: Any, input: str = "", output: str = "") -> Any:
    """The single-input, single-output part of ``sys`` from ``input`` to
    ``output``, by name (empty: the first)."""
    if sys.ninputs == 1 and sys.noutputs == 1:
        return sys
    i = _index(sys.input_labels, input, "input")
    o = _index(sys.output_labels, output, "output")
    return sys[o, i]


def _index(labels: list[str], name: str, kind: str) -> int:
    if not name.strip():
        return 0
    if name.strip() not in labels:
        raise ValueError(f"No {kind} '{name}'; the system has {', '.join(labels)}")
    return labels.index(name.strip())


def clean(sys: Any) -> Any:
    """A transfer function without the round-off coefficients (1e-16 s) that
    conversions leave behind; anything else unchanged."""
    import control

    if not isinstance(sys, control.TransferFunction):
        return sys
    num, den = control.tfdata(sys)

    def tidy(c: Any) -> Any:
        c = np.array(c, dtype=np.float64)
        c[np.abs(c) < 1e-12 * max(np.max(np.abs(c)), 1e-300)] = 0.0
        return c

    num = [[tidy(c) for c in row] for row in num]
    den = [[tidy(c) for c in row] for row in den]
    return control.tf(num, den, sys.dt)


def auto_duration(sys: Any, seconds: float) -> float:
    """``seconds``, or when 0 about four time constants of the slowest pole."""
    if seconds > 0:
        return seconds
    poles = np.asarray(sys.poles())
    if not continuous(sys):
        dt = float(sys.dt) if sys.dt is not True else 1.0
        radius = max((abs(p) for p in poles if 0 < abs(p) < 1), default=0.5)
        return dt * min(max(8.0 / -math.log(radius), 20.0), 10_000.0)
    slowest = min((-p.real for p in poles if p.real < 0), default=1.0)
    return 8.0 / slowest if stable(sys) else 10.0


def times(sys: Any, duration: float, points: int) -> Any:
    """Evenly spaced times: ``points`` of them, or every sample of a discrete system."""
    if continuous(sys):
        return np.linspace(0.0, duration, points)
    dt = float(sys.dt) if sys.dt is not True else 1.0
    return np.arange(0.0, duration + dt / 2, dt)


def dt_of(sys: Any) -> float:
    return 0.0 if continuous(sys) else (1.0 if sys.dt is True else float(sys.dt))


# --- preview and meta --------------------------------------------------------------------------


def _poly_typst(coeffs: Any, var: str) -> str:
    coeffs = np.trim_zeros(np.asarray(coeffs, dtype=np.float64), "f")
    n = len(coeffs) - 1
    terms: list[str] = []
    for i, c in enumerate(coeffs):
        power = n - i
        if c == 0:
            continue
        size = abs(c)
        number = f"{size:.4g}"
        if "e" in number:
            mantissa, exponent = number.split("e")
            number = f"{mantissa} times 10^({int(exponent)})"
        x = "" if power == 0 else (var if power == 1 else f"{var}^{power}")
        body = number if not x else (x if size == 1 else f"{number} {x}")
        sign = "-" if c < 0 else "+"
        terms.append(f"{'-' if sign == '-' else ''}{body}" if not terms else f" {sign} {body}")
    return "".join(terms) or "0"


def _describe(sys: Any) -> tuple[str, str]:
    """A one-line summary and the full text of a system."""
    import control

    kind = "transfer function" if isinstance(sys, control.TransferFunction) else "state space"
    parts = [kind]
    if isinstance(sys, control.StateSpace):
        parts.append(f"{sys.nstates} states")
    if sys.ninputs > 1 or sys.noutputs > 1:
        parts.append(f"{sys.ninputs} in, {sys.noutputs} out")
    if not continuous(sys):
        parts.append(f"dt = {dt_of(sys):g} s")
    parts.append(status(sys))
    return ", ".join(parts), str(sys)


@register_preview(LTI)
def _preview_system(value: Any, ctx: Any) -> Preview:
    """A single-input, single-output transfer function typeset as G(s); other
    systems as text, with their matrices."""
    import control

    summary, text = _describe(value)
    if not (isinstance(value, control.TransferFunction) and value.ninputs == value.noutputs == 1):
        return Preview(kind="text", summary=summary, text=text)
    from noodlelab.reports.math import math_preview

    var = "s" if continuous(value) else "z"
    num, den = control.tfdata(value)
    math_text = f"G({var}) = ({_poly_typst(num[0][0], var)})/({_poly_typst(den[0][0], var)})"
    preview = math_preview(math_text, summary=summary, text=text)
    return preview if preview.kind == "math" else Preview(kind="text", summary=summary, text=text)


@register_meta(LTI)
def _meta_system(value: Any) -> dict[str, Any]:
    return {
        "inputs": list(value.input_labels),
        "outputs": list(value.output_labels),
        "states": int(getattr(value, "nstates", 0)),
    }


# --- checkpoints: numbers only, never pickle -----------------------------------------------------


def _dt_json(sys: Any) -> Any:
    return sys.dt if sys.dt in (None, True) else float(sys.dt)


def _save_system(value: Any, fh: IO[bytes]) -> dict[str, Any]:
    import control

    labels = {"inputs": list(value.input_labels), "outputs": list(value.output_labels)}
    if isinstance(value, control.TransferFunction):
        num, den = control.tfdata(value)
        payload = {
            "num": [[[float(x) for x in c] for c in row] for row in num],
            "den": [[[float(x) for x in c] for c in row] for row in den],
        }
        fh.write(json.dumps(payload).encode())
        info = {"kind": "tf", "dt": _dt_json(value), **labels}
    elif isinstance(value, control.StateSpace):
        buf = io.BytesIO()
        np.savez(buf, A=value.A, B=value.B, C=value.C, D=value.D)
        fh.write(buf.getvalue())
        info = {"kind": "ss", "dt": _dt_json(value), "states": list(value.state_labels), **labels}
    else:
        raise TypeError(f"a {type(value).__name__} has no safe format")
    return {"system": json.dumps(info)}


def _load_system(fh: IO[bytes], info: dict[str, Any]) -> Any:
    import control

    meta = json.loads(info["system"])
    names = {"inputs": meta["inputs"], "outputs": meta["outputs"]}
    if meta["kind"] == "tf":
        data = json.loads(fh.read().decode())
        num = [[np.array(c) for c in row] for row in data["num"]]
        den = [[np.array(c) for c in row] for row in data["den"]]
        return control.tf(num, den, meta["dt"], **names)
    m = np.load(io.BytesIO(fh.read()), allow_pickle=False)
    return control.ss(m["A"], m["B"], m["C"], m["D"], meta["dt"], states=meta["states"], **names)


register_codec(LTI, "python-control", save=_save_system, load=_load_system, suffix=".bin")
