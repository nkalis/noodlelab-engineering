"""Control systems with python-control, in the manner of MATLAB's Control
System Toolbox: build a system, connect it, analyse it, design for it.

Systems flow between nodes on SYSTEM sockets: a linear time-invariant system,
as a transfer function G(s) or in state space (x' = A x + B u, y = C x + D u),
continuous or discrete. Systems are unitless, so their numbers are in
consistent SI units; the nodes that take quantities (time constants, natural
frequencies, mass and stiffness matrices) convert them first.

* :mod:`.types`: the SYSTEM socket, its preview and checkpoint codec
* :mod:`.build`: transfer functions, zeros and poles, state space, PID,
  first- and second-order systems, masses on springs and dampers
* :mod:`.combine`: series, parallel, feedback, conversion, discretisation
* :mod:`.analyse`: step, impulse, initial and forced responses; Bode,
  Nyquist and stability margins; pole-zero map, root locus, DC gain;
  controllability and observability
* :mod:`.design`: pole placement and LQR state feedback
"""

from __future__ import annotations

from . import analyse, build, combine, design
from .analyse import *  # noqa: F403
from .build import *  # noqa: F403
from .combine import *  # noqa: F403
from .design import *  # noqa: F403

__all__ = [*build.__all__, *combine.__all__, *analyse.__all__, *design.__all__]
