"""Quantum Topological Data Analysis (QTDA) — minimal demo.

Boundary conditions:
- Estimates Betti numbers of a point cloud
- Minimal: 2-point cloud, 0th Betti number
- Uses quantum circuits for topological features

Example::

    from quonic.algorithms import qtda
    result = qtda()
"""

from __future__ import annotations

from ..result import Result
from ..simulator import StatevectorSimulator


def qtda() -> Result:
    """Minimal QTDA demo: estimate 0th Betti number of 2 points."""
    import numpy as np

    # 2-point cloud: distance = 0.5
    distance = 0.5
    # Encode distance as rotation angle
    angle = distance * np.pi

    sim = StatevectorSimulator(1)
    sim.apply("ry", (0,), (angle,))

    # Measure: connected components count
    prob_connected = (1 + sim.expectation("Z")) / 2
    betti_0 = 1 if prob_connected > 0.5 else 2

    return Result.from_value(float(betti_0), prob_connected=prob_connected)
