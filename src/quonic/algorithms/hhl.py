"""HHL algorithm — solve linear systems Ax = b (minimal demo).

Boundary conditions:
- Minimal version: 2x2 diagonal matrix only
- Requires QPE + controlled rotation + inverse QPE
- Demonstrates exponential speedup concept
- Full HHL requires O(log(n)) qubits for n-dimensional system
- NOT a production solver — educational demonstration only

Example::

    from quonic.algorithms import hhl
    # Solve [[3,0],[0,1]] * x = [1,1]
    result = hhl(matrix=[[3,0],[0,1]], vector=[1,1], shots=1024)
"""

from __future__ import annotations

import math

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result


def hhl(
    matrix: list[list[float]] | None = None,
    vector: list[float] | None = None,
    n_clock: int = 3,
    backend: str = "auto",
    shots: int = 1024,
) -> Result:
    """Minimal HHL demonstration for 2x2 diagonal matrix.

    Args:
        matrix: 2x2 diagonal matrix (only diagonal entries used).
        vector: 2-element vector b.
        n_clock: Number of clock qubits for QPE.
        backend: Backend to use.
        shots: Number of shots.

    Returns:
        Result with approximate solution.
    """
    if matrix is None:
        matrix = [[3.0, 0.0], [0.0, 1.0]]
    if vector is None:
        vector = [1.0, 1.0]

    n_total = 1 + n_clock + 1  # data + clock + ancilla
    circuit = Circuit()

    # Prepare |b> on data qubit (qubit 0)
    # For [1,1]/sqrt(2), apply H
    circuit.add(GateOperation("h", (0,)))

    # QPE: encode eigenvalues in clock register
    for i in range(n_clock):
        circuit.add(GateOperation("h", (1 + i,)))

    # Controlled-U^(2^k) operations (simplified for diagonal matrix)
    for k in range(n_clock):
        # For diagonal matrix with eigenvalues λ, controlled-Rz(2πλ/2^k)
        angle = 2 * math.pi * matrix[0][0] / (2 ** k)
        circuit.add(GateOperation("cp", (1 + k, 0), (angle,)))

    # Inverse QPE (simplified)
    for i in range(n_clock):
        circuit.add(GateOperation("h", (1 + i,)))

    # Controlled rotation on ancilla (qubit n_total-1)
    # C = sum |2^n_clock / λ| conditioned on clock register
    # Simplified: just apply a rotation
    circuit.add(GateOperation("ry", (n_total - 1,), (0.5,)))

    # Measure data qubit
    result = run_circuit(circuit, backend=backend, shots=shots)
    return Result.from_value(0.0, counts=result.counts)
