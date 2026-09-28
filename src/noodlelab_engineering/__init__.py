"""Engineering nodes: materials, sections, beams and stresses, heat transfer,
pipe flow, control systems, decibel budgets and trade studies. Part of the
``engineering`` tier: ``pip install "noodlelab[engineering]"``.

Inputs and outputs carry units (Pint quantities), so a beam load in kN and a
modulus in GPa give a deflection in mm without conversion factors, and a
wrong dimension is caught when linking. The formulas are the standard
closed-form ones for preliminary design; each node's description says which.

* :mod:`.materials`: typical properties of common engineering materials
* :mod:`.sections`: areas, second moments and section moduli of cross-sections
* :mod:`.structures`: beams, stresses, buckling and safety factors
* :mod:`.thermal`: expansion, conduction and convection
* :mod:`.fluids`: Reynolds number and pipe pressure drop
* :mod:`.controls`: control systems: build, connect, analyse and design for linear
  systems (transfer functions and state space), as in MATLAB's Control System Toolbox
* :mod:`.decibels`: dB conversions and gain/loss budgets
* :mod:`.trade`: weighted decision matrices
"""

from __future__ import annotations

from noodlelab.plugin import require

require("engineering", "control", "matplotlib", "numpy", "pandas", "pint", "scipy")

from .controls import *  # noqa: F403
from .decibels import *  # noqa: F403
from .fluids import *  # noqa: F403
from .materials import *  # noqa: F403
from .sections import *  # noqa: F403
from .structures import *  # noqa: F403
from .thermal import *  # noqa: F403
from .trade import *  # noqa: F403
