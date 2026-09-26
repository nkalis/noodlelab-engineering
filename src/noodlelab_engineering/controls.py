"""Control systems with python-control: step and frequency responses of a
continuous-time transfer function G(s) = num(s) / den(s), typed as polynomial coefficients
in descending powers of s ("1, 2, 1" is s² + 2s + 1)."""

from __future__ import annotations

import math
from typing import Annotated, NamedTuple

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from noodlelab import Param, node

__all__ = ["bode_plot", "step_response"]

Coefficients = Annotated[
    str, Param(description="Coefficients in descending powers of s, e.g. 1, 2, 1")
]


def _poly(text: str) -> list[float]:
    try:
        out = [float(c) for c in text.replace(";", ",").replace(" ", ",").split(",") if c]
    except ValueError:
        raise ValueError(f"Not a list of numbers: '{text}'") from None
    if not out or not any(out):
        raise ValueError("Enter at least one non-zero coefficient")
    return out


def _problem(numerator: str, denominator: str) -> str | None:
    try:
        num, den = _poly(numerator), _poly(denominator)
    except ValueError as exc:
        return str(exc)
    if len(np.trim_zeros(num, "f")) > len(np.trim_zeros(den, "f")):
        return "The numerator's order is higher than the denominator's (not proper)"
    return None


class Step(NamedTuple):
    t: NDArray[np.float64]
    y: NDArray[np.float64]
    plot: Figure
    final_value: float
    overshoot: float
    rise_time: float
    settling_time: float
    stable: bool
    summary: dict[str, float]


def _tf(numerator: str, denominator: str):
    import control

    return control.tf(_poly(numerator), _poly(denominator))


@node(category="Engineering/Controls", title="Step Response")
def step_response(
    numerator: Coefficients = "1",
    denominator: Coefficients = "1, 0.8, 1",
    duration: Annotated[float, Param(min=0.0, description="Seconds; 0: automatic")] = 0.0,
    points: Annotated[int, Param(min=50, max=20_000)] = 1000,
) -> Step:
    """The response to a unit step, with python-control: overshoot (%),
    10–90 % rise time, 2 % settling time and final value. An unstable system
    (a pole in the right half plane) has no final value, overshoot or
    settling time (NaN)."""
    import control

    sys = _tf(numerator, denominator)
    poles = control.poles(sys)
    stable = bool(np.all(poles.real < 0)) if len(poles) else True
    if duration <= 0:
        slowest = min((-p.real for p in poles if p.real < 0), default=1.0)
        duration = 8.0 / slowest if stable else 10.0
    response = control.step_response(sys, T=np.linspace(0.0, duration, points))
    t = np.asarray(response.time, dtype=np.float64)
    y = np.asarray(np.squeeze(response.outputs), dtype=np.float64)
    final = overshoot = rise = settling = math.nan
    if stable:
        info = control.step_info(sys, SettlingTimeThreshold=0.02, RiseTimeLimits=(0.1, 0.9))
        final = float(info["SteadyStateValue"])
        overshoot = float(info["Overshoot"])
        rise = float(info["RiseTime"])
        settling = float(info["SettlingTime"])
    fig = Figure(figsize=(6.0, 3.2), layout="constrained")
    ax = fig.add_subplot()
    ax.plot(t, y, lw=2)
    if math.isfinite(final):
        ax.axhline(final, color="0.5", ls="--", lw=1)
    ax.set(title="Step response", xlabel="t (s)", ylabel="y")
    ax.grid(alpha=0.3)
    return Step(
        t,
        y,
        fig,
        final,
        overshoot,
        rise,
        settling,
        stable,
        {
            "Final value": final,
            "Overshoot (%)": overshoot,
            "Rise time 10–90 % (s)": rise,
            "Settling time 2 % (s)": settling,
        },
    )


@step_response.check
def _check_step(numerator: str = "1", denominator: str = "1, 0.8, 1"):
    return _problem(numerator, denominator)


class Bode(NamedTuple):
    frequency: NDArray[np.float64]
    magnitude_db: NDArray[np.float64]
    phase_deg: NDArray[np.float64]
    plot: Figure
    gain_margin_db: float
    phase_margin_deg: float
    summary: dict[str, float]


@node(category="Engineering/Controls", title="Bode Plot")
def bode_plot(
    numerator: Coefficients = "1",
    denominator: Coefficients = "1, 3, 3, 1",
    min_frequency: Annotated[float, Param(min=0.0, description="rad/s")] = 0.01,
    max_frequency: Annotated[float, Param(min=0.0, description="rad/s")] = 100.0,
    points: Annotated[int, Param(min=50, max=20_000)] = 500,
) -> Bode:
    """Magnitude (dB) and phase (degrees) against frequency (rad/s), with the
    open-loop gain and phase margins from python-control (NaN where the
    curve does not cross 0 dB or −180°)."""
    import control

    sys = _tf(numerator, denominator)
    w = np.logspace(np.log10(min_frequency), np.log10(max_frequency), points)
    fr = control.frequency_response(sys, w)
    mag = 20 * np.log10(np.asarray(np.squeeze(fr.magnitude), dtype=np.float64))
    phase = np.degrees(np.unwrap(np.asarray(np.squeeze(fr.phase), dtype=np.float64)))
    gm_ratio, pm, _, _ = control.margin(sys)
    gm = 20 * math.log10(gm_ratio) if math.isfinite(gm_ratio) and gm_ratio > 0 else math.nan
    pm = float(pm) if math.isfinite(pm) else math.nan
    fig = Figure(figsize=(6.0, 4.4), layout="constrained")
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.semilogx(w, mag, lw=2)
    ax1.axhline(0.0, color="0.6", lw=1)
    ax1.set(ylabel="magnitude (dB)", title="Bode plot")
    ax2.semilogx(w, phase, lw=2)
    ax2.axhline(-180.0, color="0.6", lw=1)
    ax2.set(xlabel="ω (rad/s)", ylabel="phase (°)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3, which="both")
    return Bode(
        w,
        mag,
        phase,
        fig,
        gm,
        pm,
        {"Gain margin (dB)": gm, "Phase margin (°)": pm},
    )


@bode_plot.check
def _check_bode(
    numerator: str = "1",
    denominator: str = "1, 3, 3, 1",
    min_frequency: float = 0.01,
    max_frequency: float = 100.0,
):
    if min_frequency <= 0 or max_frequency <= min_frequency:
        return "The frequency range must be positive and increasing"
    return _problem(numerator, denominator)
