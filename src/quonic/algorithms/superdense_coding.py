"""Superdense Coding — send 2 classical bits using 1 qubit.

Boundary conditions:
- Requires 2 qubits (Bell pair)
- Alice encodes 2 bits by applying one of 4 gates (I, X, Z, XZ)
- Bob decodes by Bell measurement
- Deterministic: always succeeds

Example::

    from quonic.algorithms import superdense_coding
    result = superdense_coding(message="10", shots=100)
    print(result["decoded"])  # "10"
"""

from __future__ import annotations

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result


def superdense_coding(
    message: str = "00",
    backend: str = "auto",
    shots: int = 100,
) -> Result:
    """Run superdense coding protocol.

    Args:
        message: 2-bit string to send ("00", "01", "10", "11").
        backend: Backend to use.
        shots: Number of shots.

    Returns:
        Result with decoded message.
    """
    circuit = Circuit()

    # Create Bell pair
    circuit.add(GateOperation("h", (0,)))
    circuit.add(GateOperation("cx", (0, 1)))

    # Alice encodes message
    if message == "01":
        circuit.add(GateOperation("x", (0,)))
    elif message == "10":
        circuit.add(GateOperation("z", (0,)))
    elif message == "11":
        circuit.add(GateOperation("x", (0,)))
        circuit.add(GateOperation("z", (0,)))
    # "00" → do nothing (identity)

    # Bob decodes
    circuit.add(GateOperation("cx", (0, 1)))
    circuit.add(GateOperation("h", (0,)))

    result = run_circuit(circuit, backend=backend, shots=shots)
    decoded = max(result.counts, key=result.counts.get)
    return Result.from_value(float(int(decoded, 2)), decoded=decoded, counts=result.counts)
