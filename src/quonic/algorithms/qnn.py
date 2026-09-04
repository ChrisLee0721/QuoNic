"""Quantum Neural Network (QNN) — generic variational quantum circuit framework.

Boundary conditions:
- General parameterized circuit with configurable depth
- Can be used for classification, regression, or generative tasks
- Wraps VQC with more flexible architecture

Example::

    from quonic.algorithms import qnn
    result = qnn()
"""

from __future__ import annotations

from ..result import Result
from ..simulator import StatevectorSimulator


def qnn(
    n_qubits: int = 2,
    depth: int = 2,
) -> Result:
    """Minimal QNN demo with configurable depth."""
    import numpy as np

    params = np.random.randn(depth * n_qubits * 2) * 0.1
    sim = StatevectorSimulator(n_qubits)

    idx = 0
    for d in range(depth):
        # Rotation layer
        for q in range(n_qubits):
            sim.apply("ry", (q,), (float(params[idx]),))
            idx += 1
            sim.apply("rz", (q,), (float(params[idx]),))
            idx += 1
        # Entangling layer
        for q in range(n_qubits - 1):
            sim.apply("cx", (q, q + 1))

    # Measure all qubits
    pauli = "Z" + "I" * (n_qubits - 1)  # measure Z on first qubit
    expectations = [sim.expectation(pauli)]
    return Result.from_value(float(expectations[0]))
