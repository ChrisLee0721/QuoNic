"""Quantum Graph Neural Network (QGNN) — minimal demo.

Boundary conditions:
- 3-node graph with quantum message passing
- Parameterized circuit on graph structure
- NOT a production GNN — demonstrates the concept

Example::

    from quonic.algorithms import qgnn
    result = qgnn()
"""

from __future__ import annotations

from ..result import Result
from ..simulator import StatevectorSimulator


def qgnn() -> Result:
    """Minimal QGNN demo on 3-node graph."""
    edges = [(0, 1), (1, 2)]
    params = [0.5, 0.3, 0.7]

    sim = StatevectorSimulator(3)
    # Initial node features
    for i in range(3):
        sim.apply("ry", (i,), (params[i],))
    # Message passing: CX along edges
    for i, j in edges:
        sim.apply("cx", (i, j))
    # Readout
    return Result.from_value(sim.expectation("ZZZ"))
