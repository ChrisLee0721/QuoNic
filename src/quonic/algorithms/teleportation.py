"""Quantum Teleportation — transfer a quantum state using entanglement.

Standard 3-qubit protocol: Alice has qubit 0 (state to teleport), shares
Bell pair with Bob (qubits 1,2). Alice measures, Bob applies corrections.

Boundary conditions:
- Requires 3 qubits
- Deterministic: always succeeds (in noise-free simulation)
- Demonstrates entanglement as resource
- The teleported state is on qubit 2 at the end

Example::

    from quonic.algorithms import teleportation
    result = teleportation(shots=1024)
    print(result.counts)  # should show |00> and |11> if input was |+>
"""

from __future__ import annotations

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result


def teleportation(
    theta: float = 0.0,
    backend: str = "auto",
    shots: int = 1024,
) -> Result:
    """Run quantum teleportation protocol.

    Args:
        theta: Rotation angle for the state to teleport (Ry(theta)|0>).
        backend: Backend to use.
        shots: Number of shots.

    Returns:
        Result with measurement counts.
    """
    circuit = Circuit()

    # Prepare state to teleport on qubit 0
    if theta != 0.0:
        circuit.add(GateOperation("ry", (0,), (theta,)))

    # Create Bell pair between qubits 1 and 2
    circuit.add(GateOperation("h", (1,)))
    circuit.add(GateOperation("cx", (1, 2)))

    # Alice's operations
    circuit.add(GateOperation("cx", (0, 1)))
    circuit.add(GateOperation("h", (0,)))

    # Alice measures qubits 0 and 1 (auto-measured by backend)
    # Bob applies corrections based on Alice's results
    # In simulation, we apply all corrections and let the backend handle it
    circuit.add(GateOperation("cx", (1, 2)))
    circuit.add(GateOperation("cz", (0, 2)))

    return run_circuit(circuit, backend=backend, shots=shots)
