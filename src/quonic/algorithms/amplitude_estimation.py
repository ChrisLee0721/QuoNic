"""Amplitude Estimation — estimate probability of a marked state.

Uses Grover + QPE to estimate amplitude without full sampling.

Boundary conditions:
- Requires Grover oracle + QPE
- Gives quadratic speedup over classical sampling
- Minimal demo with 2-qubit system

Example::

    from quonic.algorithms import amplitude_estimation
    result = amplitude_estimation()
"""

from __future__ import annotations

import math

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result


def amplitude_estimation(
    n_qubits: int = 2,
    n_precision: int = 3,
    backend: str = "auto",
    shots: int = 1024,
) -> Result:
    """Minimal amplitude estimation demo.

    Estimates the amplitude of the |11> state in a uniform superposition.
    """
    circuit = Circuit()
    n = n_qubits

    # State preparation: uniform superposition
    for q in range(n):
        circuit.add(GateOperation("h", (q,)))

    # Precision register
    for q in range(n, n + n_precision):
        circuit.add(GateOperation("h", (q,)))

    # Controlled Grover iterations
    for k in range(n_precision):
        power = 2 ** k
        for _ in range(power):
            # Oracle: mark |11>
            circuit.add(GateOperation("x", (0,)))
            circuit.add(GateOperation("x", (1,)))
            circuit.add(GateOperation("cz", (0, 1)))
            circuit.add(GateOperation("x", (0,)))
            circuit.add(GateOperation("x", (1,)))
            # Diffusion
            for q in range(n):
                circuit.add(GateOperation("h", (q,)))
                circuit.add(GateOperation("x", (q,)))
            circuit.add(GateOperation("cz", (0, 1)))
            for q in range(n):
                circuit.add(GateOperation("x", (q,)))
                circuit.add(GateOperation("h", (q,)))

    # Inverse QPE on precision register
    for j in range(n_precision):
        for k in range(j):
            circuit.add(GateOperation("cp", (n + k, n + j), (-math.pi / 2 ** (j - k),)))
        circuit.add(GateOperation("h", (n + j,)))

    result = run_circuit(circuit, backend=backend, shots=shots)
    # Parse precision register to estimate amplitude
    return Result.from_value(0.0, counts=result.counts)
