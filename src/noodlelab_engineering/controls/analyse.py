"""Analysing systems: responses in time (step, impulse, initial condition, any
input), in frequency (Bode, Nyquist, stability margins), and their poles and
zeros (pole-zero map, root locus), controllability and observability.

python-control's own plots use pyplot, which is not safe in the worker
threads nodes run in, so every figure here is drawn from python-control's
numbers onto a plain Matplotlib ``Figure``.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, NamedTuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray

from noodlelab import Param, Quantity, node
from noodlelab.plugin.units import is_quantity

from .types import (
    Signal,
    System,
    auto_duration,
    clean,
    continuous,
    dt_of,
    select,
    stable,
    times,
    unstable_poles,
)

__all__ = [
    "bode_plot",
    "controllability",
    "dc_gain",
    "impulse_response",
    "initial_response",
    "nyquist_plot",
    "pole_zero_map",
    "root_locus",
    "simulate",
    "stability_margins",
    "step_response",
]

Seconds = Quantity["s"]
RadPerSecond = Quantity["rad/s"]
ZERO_S = Quantity(0.0, "s")
AUTO_W = Quantity(0.0, "rad/s")
Duration = Annotated[Seconds, Param(description="0 s: automatic, from the slowest pole")]
Points = Annotated[int, Param(min=50, max=100_000)]
CATEGORY = "Engineering/Control Analysis"


def _figure(height: float = 3.2) -> tuple[Figure, Any]:
    fig = Figure(figsize=(6.0, height), layout="constrained")
    return fig, fig.add_subplot()


def _seconds(t: NDArray[np.float64]) -> Quantity:
    return Quantity(np.asarray(t, dtype=np.float64), "s")


def _nan_if(value: Any) -> float:
    value = float(value)
    return value if math.isfinite(value) else math.nan


# --- time responses ----------------------------------------------------------------------------


class Step(NamedTuple):
    t: Seconds
    y: NDArray[np.float64]
    plot: Figure
    final_value: float
    overshoot: float
    rise_time: Seconds
    settling_time: Seconds
    peak_time: Seconds
    stable: bool
    summary: dict[str, float]


@node(category=CATEGORY, title="Step Response", version=2)
def step_response(
    system: System,
    duration: Duration = ZERO_S,
    points: Points = 1000,
    input: Signal = "",
    output: Signal = "",
    settling: Annotated[
        float, Param(min=0.1, max=20.0, description="The settling band, % of the final value")
    ] = 2.0,
) -> Step:
    """The response to a unit step: overshoot (%), 10–90 % rise time,
    settling time (within ``settling`` % of the final value for good), time
    of the peak and final value. An unstable system has no final value,
    overshoot or settling time (NaN). The system comes from Transfer
    Function, Feedback and the other builders (its coefficients used to be
    typed here)."""
    import control

    sys = select(system, input, output)
    t = times(sys, auto_duration(sys, duration.m_as("s")), points)
    response = control.step_response(sys, T=t)
    y = np.asarray(np.squeeze(response.outputs), dtype=np.float64)
    is_stable = stable(sys)
    final = overshoot = rise = settle = peak = math.nan
    if is_stable:
        info = control.step_info(
            sys, T=t, SettlingTimeThreshold=settling / 100, RiseTimeLimits=(0.1, 0.9)
        )
        final = _nan_if(info["SteadyStateValue"])
        overshoot = _nan_if(info["Overshoot"])
        rise = _nan_if(info["RiseTime"])
        settle = _nan_if(info["SettlingTime"])
        peak = _nan_if(info["PeakTime"])
    fig, ax = _figure()
    ax.plot(t, y, lw=2)
    if math.isfinite(final):
        ax.axhline(final, color="0.5", ls="--", lw=1)
        for edge in (1 - settling / 100, 1 + settling / 100):
            ax.axhline(final * edge, color="0.75", ls=":", lw=1)
    if math.isfinite(settle):
        ax.axvline(settle, color="#c44e52", ls=":", lw=1, label=f"settled, {settle:.3g} s")
        ax.legend(fontsize="small")
    ax.set(title="Step response", xlabel="t (s)", ylabel="y")
    ax.grid(alpha=0.3)
    return Step(
        _seconds(t),
        y,
        fig,
        final,
        overshoot,
        Quantity(rise, "s"),
        Quantity(settle, "s"),
        Quantity(peak, "s"),
        is_stable,
        {
            "Final value": final,
            "Overshoot (%)": overshoot,
            "Rise time 10–90 % (s)": rise,
            f"Settling time {settling:g} % (s)": settle,
            "Peak time (s)": peak,
        },
    )


class Response(NamedTuple):
    t: Seconds
    y: NDArray[np.float64]
    plot: Figure
    peak: float
    final_value: float
    summary: dict[str, float]


def _response(t: Any, y: Any, title: str, u: Any = None) -> Response:
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fig, ax = _figure()
    if u is not None:
        ax.plot(t, u, color="0.6", lw=1, label="input u")
    ax.plot(t, y, lw=2, label="output y")
    if u is not None:
        ax.legend(fontsize="small")
    ax.set(title=title, xlabel="t (s)", ylabel="y")
    ax.grid(alpha=0.3)
    peak = float(y[int(np.argmax(np.abs(y)))]) if y.size else math.nan
    final = float(y[-1]) if y.size else math.nan
    return Response(_seconds(t), y, fig, peak, final, {"Peak": peak, "Final value": final})


@node(category=CATEGORY, title="Impulse Response")
def impulse_response(
    system: System,
    duration: Duration = ZERO_S,
    points: Points = 1000,
    input: Signal = "",
    output: Signal = "",
) -> Response:
    """The response to a unit impulse (a hammer blow): the system's own
    motion, which rings at its natural frequencies. ``peak`` is the value
    largest in size, with its sign."""
    import control

    sys = select(system, input, output)
    t = times(sys, auto_duration(sys, duration.m_as("s")), points)
    r = control.impulse_response(sys, T=t)
    return _response(r.time, np.squeeze(r.outputs), "Impulse response")


def _floats(text: str) -> list[float]:
    try:
        return [float(v) for v in text.replace(";", ",").replace(" ", ",").split(",") if v]
    except ValueError:
        raise ValueError(f"Not a list of numbers: '{text}'") from None


@node(category=CATEGORY, title="Initial Response")
def initial_response(
    system: System,
    initial_state: Annotated[
        str, Param(description="One value per state, comma separated: 0.01, 0, 0, 0")
    ] = "1, 0",
    duration: Duration = ZERO_S,
    points: Points = 1000,
    output: Signal = "",
) -> Response:
    """The free motion from an initial state with no input: a structure
    released from a deflected shape. The states are those of the state-space
    model (Mass-Spring-Damper: positions, then velocities, in SI units)."""
    import control

    sys = control.ss(system)
    x0 = _floats(initial_state)
    if len(x0) != sys.nstates:
        raise ValueError(f"The system has {sys.nstates} states: give {sys.nstates} values")
    o = sys.output_labels.index(output.strip()) if output.strip() else 0
    t = times(sys, auto_duration(sys, duration.m_as("s")), points)
    r = control.initial_response(sys, T=t, X0=x0)
    y = np.atleast_2d(np.asarray(r.outputs))[o]
    return _response(r.time, y, "Initial condition response")


@node(category=CATEGORY, title="Simulate", sample=False)
def simulate(
    system: System,
    t: Annotated[Quantity | NDArray[np.floating], Param(description="Times, evenly spaced")],
    u: NDArray[np.floating],
    input: Signal = "",
    output: Signal = "",
) -> Response:
    """The response to any input signal u(t), from rest: a measured road
    profile, a set-point schedule, a sine sweep. ``t`` in seconds (a plain
    array counts as seconds), one value of u per time."""
    import control

    sys = select(system, input, output)
    tv = np.asarray(t.m_as("s") if is_quantity(t) else t, dtype=np.float64)
    uv = np.asarray(u, dtype=np.float64)
    if tv.shape != uv.shape or tv.ndim != 1:
        raise ValueError(f"t and u must be two lists of the same length ({tv.size} and {uv.size})")
    r = control.forced_response(sys, T=tv, U=uv)
    return _response(r.time, np.squeeze(r.outputs), "Response", u=uv)


# --- frequency responses -------------------------------------------------------------------------


def _frequencies(sys: Any, lo: float, hi: float, points: int) -> NDArray[np.float64]:
    """From ``lo`` to ``hi`` rad/s; either 0: a decade beyond the poles and zeros."""
    corners = [abs(x) for x in np.concatenate([sys.poles(), sys.zeros()]) if abs(x) > 1e-12]
    if not continuous(sys):
        dt = dt_of(sys)
        corners = [abs(np.log(x)) / dt for x in corners if abs(np.log(x)) > 1e-12]
    lo = lo if lo > 0 else (0.1 * min(corners) if corners else 0.01)
    hi = hi if hi > 0 else (10 * max(corners) if corners else 100.0)
    if not continuous(sys):
        hi = min(hi, math.pi / dt_of(sys))
    if hi <= lo:
        hi = lo * 1000
    return np.logspace(np.log10(lo), np.log10(hi), points)


class Bode(NamedTuple):
    frequency: NDArray[np.float64]
    magnitude_db: NDArray[np.float64]
    phase_deg: NDArray[np.float64]
    plot: Figure
    gain_margin_db: float
    phase_margin_deg: float
    summary: dict[str, float]


@node(category=CATEGORY, title="Bode Plot", version=2)
def bode_plot(
    system: System,
    min_frequency: Annotated[RadPerSecond, Param(description="0: automatic")] = AUTO_W,
    max_frequency: Annotated[RadPerSecond, Param(description="0: automatic")] = AUTO_W,
    points: Points = 500,
    input: Signal = "",
    output: Signal = "",
) -> Bode:
    """Magnitude (dB) and phase (degrees) against frequency (rad/s), with the
    gain and phase margins read as for an open loop (NaN where the curve does
    not cross 0 dB or −180°). Link the open loop L(s), the controller and
    plant in series, to see how far the closed loop is from instability."""
    import control

    sys = select(system, input, output)
    w = _frequencies(sys, min_frequency.m_as("rad/s"), max_frequency.m_as("rad/s"), points)
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
    return Bode(w, mag, phase, fig, gm, pm, {"Gain margin (dB)": gm, "Phase margin (°)": pm})


class Margins(NamedTuple):
    gain_margin_db: float
    phase_margin_deg: float
    gain_crossover: RadPerSecond
    phase_crossover: RadPerSecond
    stability_margin: float
    closed_loop_stable: bool
    summary: dict[str, float]


@node(category=CATEGORY, title="Stability Margins")
def stability_margins(system: System, input: Signal = "", output: Signal = "") -> Margins:
    """How far an open loop L(s) is from instability once the loop is closed
    with unity negative feedback. The gain margin (dB) is how much the gain
    can rise, at the phase crossover (phase −180°); the phase margin (°) how
    much phase lag can be added, at the gain crossover (|L| = 1). Infinite
    when the curve never crosses. ``stability_margin`` is the closest the
    Nyquist curve comes to −1 (1 is far, 0 is unstable)."""
    import control

    sys = select(system, input, output)
    gm, pm, sm, wpc, wgc, _ = control.stability_margins(sys)
    gm_db = 20 * math.log10(gm) if gm > 0 else math.nan
    closed = stable(control.feedback(sys, 1))
    summary = {
        "Gain margin (dB)": float(gm_db),
        "Phase margin (°)": float(pm),
        "Gain crossover (rad/s)": float(wgc),
        "Phase crossover (rad/s)": float(wpc),
        "Stability margin": float(sm),
    }
    return Margins(
        float(gm_db),
        float(pm),
        Quantity(float(wgc), "rad/s"),
        Quantity(float(wpc), "rad/s"),
        float(sm),
        closed,
        summary,
    )


class Nyquist(NamedTuple):
    plot: Figure
    encirclements: int
    open_loop_unstable_poles: int
    closed_loop_stable: bool
    summary: dict[str, float]


@node(category=CATEGORY, title="Nyquist Plot")
def nyquist_plot(system: System, input: Signal = "", output: Signal = "") -> Nyquist:
    """The open loop L(jω) drawn in the complex plane for all frequencies.
    The closed loop (unity negative feedback) is stable when the curve
    encircles −1 anticlockwise once for each unstable open-loop pole: Z = N + P
    unstable closed-loop poles, with N the clockwise encirclements."""
    import control

    # as a transfer function, an integrator's pole is exactly 0 rather than
    # ±1e-15, which would move the contour to the wrong side of it
    sys = clean(control.tf(select(system, input, output)))
    r = control.nyquist_response(sys)
    unstable = unstable_poles(sys)
    n = int(r.count)
    curve = np.asarray(r.response)
    fig, ax = _figure(4.4)
    ax.plot(curve.real, curve.imag, lw=2, label="ω > 0")
    ax.plot(curve.real, -curve.imag, lw=1, ls="--", color="0.5", label="ω < 0")
    ax.plot([-1], [0], "r+", ms=12, mew=2)
    ax.set(title="Nyquist plot", xlabel="real", ylabel="imaginary")
    ax.axhline(0, color="0.8", lw=1)
    ax.axvline(0, color="0.8", lw=1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")
    ok = n + unstable == 0
    return Nyquist(
        fig,
        n,
        unstable,
        ok,
        {"Encirclements of −1 (clockwise)": float(n), "Unstable open-loop poles": unstable},
    )


# --- poles and zeros ------------------------------------------------------------------------------


def _s_plane(sys: Any, roots: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Discrete roots as their continuous equivalents, s = ln(z)/dt."""
    if continuous(sys) or not roots.size:
        return roots
    with np.errstate(divide="ignore"):
        return np.log(roots.astype(np.complex128)) / dt_of(sys)


class PoleZero(NamedTuple):
    poles: NDArray[np.complex128]
    zeros: NDArray[np.complex128]
    table: pd.DataFrame
    plot: Figure
    stable: bool
    min_damping: float
    dc_gain: float
    summary: dict[str, float]


@node(category=CATEGORY, title="Pole-Zero Map")
def pole_zero_map(system: System, input: Signal = "", output: Signal = "") -> PoleZero:
    """The poles (×) and zeros (○) in the complex plane, and a table with
    each pole's natural frequency ωn, damping ratio ζ and time constant τ.
    Poles in the right half plane (outside the unit circle, when discrete)
    make the system unstable; lightly damped ones make it ring."""
    import control

    sys = select(system, input, output)
    poles = np.asarray(sys.poles(), dtype=np.complex128)
    zeros = np.asarray(sys.zeros(), dtype=np.complex128)
    s = _s_plane(sys, poles)
    wn = np.abs(s)
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta = np.where(wn > 0, -s.real / wn, 1.0)
        tau = np.where(s.real != 0, -1.0 / s.real, np.inf)
    table = pd.DataFrame(
        {
            "pole": [f"{p.real:.4g}{p.imag:+.4g}j" for p in poles],
            "natural frequency (rad/s)": wn,
            "damping ratio": zeta,
            "time constant (s)": tau,
        }
    )
    fig, ax = _figure(4.0)
    if not continuous(sys):
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color="0.7", lw=1)
    ax.plot(poles.real, poles.imag, "x", ms=9, mew=2, label="poles")
    if zeros.size:
        ax.plot(zeros.real, zeros.imag, "o", ms=8, mfc="none", mew=2, label="zeros")
    ax.axhline(0, color="0.8", lw=1)
    ax.axvline(0, color="0.8", lw=1)
    ax.set(title="Pole-zero map", xlabel="real", ylabel="imaginary")
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")
    gain = float(np.real(control.dcgain(sys)))
    least = float(np.min(zeta)) if zeta.size else math.nan
    is_stable = stable(sys)
    return PoleZero(
        poles,
        zeros,
        table,
        fig,
        is_stable,
        least,
        gain,
        {"Poles": float(poles.size), "Smallest damping ratio": least, "DC gain": gain},
    )


@node(category=CATEGORY, title="DC Gain")
def dc_gain(system: System, input: Signal = "", output: Signal = "") -> float:
    """The steady-state gain: the output for a constant unit input, once
    everything has settled, G(0) (or G(1) when discrete). Infinite with an
    integrator."""
    import control

    return float(np.real(control.dcgain(select(system, input, output))))


class RootLocus(NamedTuple):
    plot: Figure
    poles_at_gain: NDArray[np.complex128]
    stable_at_gain: bool
    critical_gain: float
    summary: dict[str, float]


def _closed_loop_stable(num: Any, den: Any, k: float, discrete: bool) -> bool:
    roots = np.roots(np.polyadd(den, k * np.asarray(num)))
    return bool(np.all(np.abs(roots) < 1) if discrete else np.all(roots.real < 0))


@node(category=CATEGORY, title="Root Locus")
def root_locus(
    system: System,
    gain: Annotated[float, Param(min=0.0, description="Mark the poles at this gain K")] = 1.0,
    max_gain: Annotated[float, Param(min=0.0, description="0: automatic")] = 0.0,
    input: Signal = "",
    output: Signal = "",
) -> RootLocus:
    """Where the closed-loop poles go as the gain K of the loop K L(s) rises
    from 0: they start at the open-loop poles (×) and end at its zeros (○) or
    run off to infinity. ``critical_gain`` is the smallest gain at which the
    closed loop turns unstable (infinite if it never does, NaN if it is never
    stable)."""
    import control

    sys = clean(control.tf(select(system, input, output)))
    num, den = (np.asarray(c[0][0], dtype=np.float64) for c in control.tfdata(sys))
    if max_gain > 0:
        gains = np.concatenate([[0.0], np.geomspace(max_gain * 1e-4, max_gain, 400)])
        rl = control.root_locus_map(sys, gains=gains)
    else:
        rl = control.root_locus_map(sys)
        gains = np.asarray(rl.gains, dtype=np.float64)
    loci = np.asarray(rl.loci)
    at = np.asarray(control.root_locus_map(sys, gains=[gain]).loci[0], dtype=np.complex128)
    discrete = not continuous(sys)
    scan = np.concatenate([[0.0], np.geomspace(1e-6, max(gains.max(), gain, 1.0) * 100, 600)])
    flags = [_closed_loop_stable(num, den, k, discrete) for k in scan]
    critical = math.inf
    if not any(flags):
        critical = math.nan
    else:
        first = flags.index(True)
        for i in range(first, len(scan) - 1):
            if flags[i] and not flags[i + 1]:
                lo, hi = scan[i], scan[i + 1]
                for _ in range(60):  # bisect to the gain where a pole crosses
                    mid = (lo + hi) / 2
                    ok = _closed_loop_stable(num, den, mid, discrete)
                    lo, hi = (mid, hi) if ok else (lo, mid)
                critical = (lo + hi) / 2
                break
    fig, ax = _figure(4.4)
    for j in range(loci.shape[1]):
        ax.plot(loci[:, j].real, loci[:, j].imag, lw=1.5)
    poles, zeros = sys.poles(), sys.zeros()
    ax.plot(poles.real, poles.imag, "kx", ms=9, mew=2)
    if len(zeros):
        ax.plot(zeros.real, zeros.imag, "ko", ms=8, mfc="none", mew=2)
    ax.plot(at.real, at.imag, "s", color="#c44e52", ms=7, label=f"K = {gain:g}")
    ax.axhline(0, color="0.8", lw=1)
    ax.axvline(0, color="0.8", lw=1)
    ax.set(title="Root locus", xlabel="real", ylabel="imaginary")
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")
    ok = _closed_loop_stable(num, den, gain, discrete)
    return RootLocus(fig, at, ok, critical, {"Critical gain": critical, "Stable at K": float(ok)})


# --- structure -------------------------------------------------------------------------------


class Controllability(NamedTuple):
    controllable: bool
    observable: bool
    controllable_rank: int
    observable_rank: int
    states: int
    controllability_matrix: NDArray[np.float64]
    observability_matrix: NDArray[np.float64]
    summary: dict[str, float]


@node(category=CATEGORY, title="Controllability & Observability")
def controllability(system: System) -> Controllability:
    """Whether the inputs can steer every state (controllable) and the
    outputs reveal every state (observable): the ranks of the controllability
    matrix [B AB A²B …] and the observability matrix [C; CA; CA²; …] against
    the number of states. Pole placement and LQR need a controllable system."""
    import control

    sys = control.ss(system)
    wc = np.asarray(control.ctrb(sys.A, sys.B), dtype=np.float64)
    wo = np.asarray(control.obsv(sys.A, sys.C), dtype=np.float64)
    rc, ro, n = int(np.linalg.matrix_rank(wc)), int(np.linalg.matrix_rank(wo)), sys.nstates
    return Controllability(
        rc == n,
        ro == n,
        rc,
        ro,
        n,
        wc,
        wo,
        {"States": float(n), "Controllable rank": float(rc), "Observable rank": float(ro)},
    )
