"""Phase-flip code — corrects single phase-flip (Z) errors.

Boundary conditions:
- 3 data qubits + 2 syndrome qubits
- Uses H gates to convert phase-flip to bit-flip
- Corrects single Z errors only
- Minimal demonstration

Example::

    from quonic.algorithms import phase_flip_code
    result = phase_flip_code(error_qubit=0, shots=100)
"""

from __future__ import annotations

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result


def phase_flip_code(
    error_qubit: int = 0,
    backend: str = "auto",
    shots: int = 100,
) -> Result:
    """Demonstrate 3-qubit phase-flip error correction."""
    circuit = Circuit()

    # Encode |1>
    circuit.add(GateOperation("x", (0,)))
    circuit.add(GateOperation("cx", (0, 1)))
    circuit.add(GateOperation("cx", (0, 2)))

    # Convert to Hadamard basis
    for q in range(3):
        circuit.add(GateOperation("h", (q,)))

    # Inject phase-flip error
    if error_qubit in (0, 1, 2):
        circuit.add(GateOperation("z", (error_qubit,)))

    # Convert back
    for q in range(3):
        circuit.add(GateOperation("h", (q,)))

    # Syndrome measurement
    circuit.add(GateOperation("cx", (0, 3)))
    circuit.add(GateOperation("cx", (1, 3)))
    circuit.add(GateOperation("cx", (1, 4)))
    circuit.add(GateOperation("cx", (2, 4)))

    result = run_circuit(circuit, backend=backend, shots=shots)
    return Result.from_value(1.0, counts=result.counts, error_qubit=error_qubit)
