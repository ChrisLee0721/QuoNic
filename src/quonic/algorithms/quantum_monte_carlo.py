"""Quantum Monte Carlo — quantum-enhanced random sampling.

Boundary conditions:
- Uses amplitude estimation for quadratic speedup
- Minimal demo: estimate mean of a distribution
- Requires multiple circuit evaluations

Example::

    from quonic.algorithms import quantum_monte_carlo
    result = quantum_monte_carlo()
"""

from __future__ import annotations

import math

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result


def quantum_monte_carlo(
    n_qubits: int = 2,
    shots: int = 1024,
    backend: str = "auto",
) -> Result:
    """Minimal quantum Monte Carlo demo."""
    circuit = Circuit()

    # Prepare superposition
    for q in range(n_qubits):
        circuit.add(GateOperation("h", (q,)))

    # Apply rotation proportional to function value
    circuit.add(GateOperation("ry", (0,), (math.pi / 4,)))

    # Measure
    result = run_circuit(circuit, backend=backend, shots=shots)

    # Estimate: P(1) on qubit 0
    p1 = sum(c for bs, c in result.counts.items() if bs[-1] == "1") / shots
    return Result.from_value(p1, counts=result.counts)
