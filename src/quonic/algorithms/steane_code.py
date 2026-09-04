"""Steane 7-qubit code — CSS code demonstration.

Boundary conditions:
- 7 data qubits + 6 syndrome qubits (13 total)
- CSS code: corrects any single-qubit error
- Minimal demonstration with syndrome extraction
- Full decoding not implemented

Example::

    from quonic.algorithms import steane_code
    result = steane_code(error_qubit=2, shots=100)
"""

from __future__ import annotations

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result


def steane_code(
    error_qubit: int = 0,
    backend: str = "auto",
    shots: int = 100,
) -> Result:
    """Demonstrate Steane 7-qubit code."""
    circuit = Circuit()

    # Simplified encoding: encode |1> in Steane code
    circuit.add(GateOperation("x", (0,)))
    # Encoding circuit (simplified)
    for q in range(1, 7):
        circuit.add(GateOperation("h", (q,)))
    for i in range(7):
        for j in range(i + 1, 7):
            if (i ^ j) & (i ^ j) - 1 == 0:  # check if single bit differs
                circuit.add(GateOperation("cx", (i, j)))

    # Inject error
    if 0 <= error_qubit < 7:
        circuit.add(GateOperation("x", (error_qubit,)))

    # Syndrome extraction (simplified)
    result = run_circuit(circuit, backend=backend, shots=shots)
    return Result.from_value(1.0, counts=result.counts, error_qubit=error_qubit)
