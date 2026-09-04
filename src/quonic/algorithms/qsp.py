"""Quantum Signal Processing (QSP) — minimal demo.

Boundary conditions:
- Single-qubit rotation sequence
- Demonstrates signal processing concept
- NOT a full QSP implementation

Example::

    from quonic.algorithms import qsp
    result = qsp()
"""

from __future__ import annotations

import math

from ..result import Result
from ..simulator import StatevectorSimulator


def qsp(angle: float = math.pi / 4) -> Result:
    """Minimal QSP demo: single-qubit signal processing."""
    sim = StatevectorSimulator(1)
    # QSP sequence: Rz(φ) · exp(i·angle·Z) · Rz(φ)
    phi = math.pi / 6
    sim.apply("rz", (0,), (phi,))
    sim.apply("rz", (0,), (angle,))
    sim.apply("rz", (0,), (phi,))
    return Result.from_value(sim.expectation("Z"))
