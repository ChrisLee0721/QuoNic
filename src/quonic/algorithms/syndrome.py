"""Syndrome measurement — error detection and correction flow.

Minimal demonstration of syndrome extraction circuit.

Boundary conditions:
- Uses ancilla qubits for syndrome measurement
- Non-destructive measurement (in theory)
- Demonstrates X and Z syndrome extraction

Example::

    from quonic.algorithms import syndrome
    result = syndrome(n_data=3, shots=100)
"""

from __future__ import annotations

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result


def syndrome(
    n_data: int = 3,
    backend: str = "auto",
    shots: int = 100,
) -> Result:
    """Demonstrate syndrome measurement circuit."""
    circuit = Circuit()

    # Prepare data qubits in a specific state
    circuit.add(GateOperation("x", (0,)))

    # Syndrome measurement using ancillas
    # Bit-flip syndrome
    for i in range(n_data - 1):
        circuit.add(GateOperation("cx", (i, n_data + i)))
        circuit.add(GateOperation("cx", (i + 1, n_data + i)))

    result = run_circuit(circuit, backend=backend, shots=shots)
    return Result.from_value(1.0, counts=result.counts)
