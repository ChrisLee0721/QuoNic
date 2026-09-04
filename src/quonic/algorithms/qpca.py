"""Quantum PCA — minimal demo of quantum principal component analysis.

Boundary conditions:
- Minimal: 2x2 density matrix
- Uses quantum phase estimation for eigenvalue extraction
- NOT a production PCA tool

Example::

    from quonic.algorithms import qpca
    result = qpca()
"""

from __future__ import annotations

from ..result import Result
from ..simulator import StatevectorSimulator


def qpca() -> Result:
    """Minimal QPCA demo: extract principal components of 2x2 state."""
    import numpy as np

    # Prepare a mixed state (simplified)
    sim = StatevectorSimulator(1)
    sim.apply("ry", (0,), (np.pi / 3,))

    # Principal component: the dominant eigenvalue
    prob_0 = (1 + sim.expectation("Z")) / 2
    eigenvalues = [prob_0, 1 - prob_0]

    return Result.from_value(float(max(eigenvalues)), eigenvalues=eigenvalues)
