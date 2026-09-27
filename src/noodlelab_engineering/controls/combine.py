"""Connecting systems: in series, in parallel and in a feedback loop, and
changing their form: transfer function or state space, continuous or
discrete, and the smallest equivalent."""

from __future__ import annotations

from typing import Annotated, Literal

from noodlelab import Param, Quantity, node

from .types import OptionalSystem, System, clean

Seconds = Quantity["s"]

__all__ = ["convert_system", "discretize", "feedback", "minimal_realisation", "parallel", "series"]

SLYCOT = "without the optional slycot package, only for single-input, single-output systems"


@node(category="Engineering/Controls", title="Series")
def series(a: System, b: System, c: OptionalSystem = None) -> System:
    """Systems one after another: the signal goes through a, then b (then c).
    A controller, an actuator and a plant in series make the open loop L(s)."""
    import control

    out = control.series(a, b)
    return out if c is None else control.series(out, c)


@node(category="Engineering/Controls", title="Parallel")
def parallel(a: System, b: System, sign: Literal["add", "subtract"] = "add") -> System:
    """Two systems side by side on the same input, their outputs added (or b
    subtracted from a)."""
    return a + b if sign == "add" else a - b


@node(category="Engineering/Controls", title="Feedback")
def feedback(
    system: System,
    feedback: Annotated[
        OptionalSystem, Param(description="H(s) in the feedback path; empty: 1 (unity)")
    ] = None,
    sign: Literal["negative", "positive"] = "negative",
) -> System:
    """Close the loop around ``system``: G / (1 + G H) with negative feedback,
    the usual kind, where the output is compared with the set point. With
    unity feedback (no H) and the open loop L = C G linked, this is the
    closed loop T = L / (1 + L)."""
    import control

    h = 1 if feedback is None else feedback
    return clean(control.feedback(system, h, sign=-1 if sign == "negative" else 1))


@node(category="Engineering/Controls", title="Convert System")
def convert_system(
    system: System, to: Literal["transfer function", "state space"] = "state space"
) -> System:
    """The same system as a transfer function or in state space. Without the
    optional slycot package, only single-input, single-output systems convert
    to a transfer function."""
    import control

    try:
        return clean(control.tf(system)) if to == "transfer function" else control.ss(system)
    except Exception as exc:  # python-control needs slycot for MIMO conversions
        raise ValueError(f"Cannot convert this system ({exc}): this works {SLYCOT}") from None


@node(category="Engineering/Controls", title="Discretize")
def discretize(
    system: System,
    sample_time: Seconds = Quantity(0.01, "s"),  # noqa: B008 - immutable
    method: Annotated[
        Literal["zoh", "foh", "tustin", "matched", "euler", "backward_diff"],
        Param(description="zoh: a sample held until the next; tustin: the bilinear transform"),
    ] = "zoh",
) -> System:
    """A continuous system as a discrete one sampled every ``sample_time``,
    for a digital controller: G(s) becomes G(z)."""
    import control

    if not system.isctime(strict=True):
        raise ValueError("The system is discrete already")
    return control.sample_system(system, sample_time.m_as("s"), method=method)


@discretize.check
def _check_discretize(sample_time: Quantity | None = None) -> str | None:
    if sample_time is not None and sample_time.magnitude <= 0:
        return "The sample time must be positive"
    return None


@node(category="Engineering/Controls", title="Minimal Realisation")
def minimal_realisation(
    system: System,
    tolerance: Annotated[float, Param(min=0.0, description="How close a pole and zero cancel")] = (
        1e-6
    ),
) -> System:
    """The system without its cancelling poles and zeros: (s + 1)/((s + 1)(s + 2))
    is 1/(s + 2). Without the optional slycot package, a state-space system
    must have one input and one output."""
    import control

    if isinstance(system, control.TransferFunction):
        return control.minreal(system, tolerance, verbose=False)
    if system.ninputs == 1 and system.noutputs == 1:
        reduced = control.minreal(clean(control.tf(system)), tolerance, verbose=False)
        return control.ss(reduced)
    raise ValueError(f"A state-space system is reduced {SLYCOT}")
