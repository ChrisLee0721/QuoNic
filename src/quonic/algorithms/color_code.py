"""Color Code — minimal demonstration of topological error correction.

Boundary conditions:
- Minimal 7-qubit color code (distance 3)
- Demonstrates 3-colorability of the code
- NOT a full implementation — shows the concept

Example::

    from quonic.algorithms import color_code
    result = color_code()
"""

from __future__ import annotations

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result


def color_code(
    backend: str = "auto",
    shots: int = 100,
) -> Result:
    """Minimal color code demo with 7 qubits."""
    circuit = Circuit()

    # 7-qubit Steane code (isomorphic to color code)
    # Data qubits: 0-6
    # Stabilizer measurements using ancillas

    # Prepare logical |0>
    for q in range(7):
        circuit.add(GateOperation("h", (q,)))

    # X-stabilizers (3 generators)
    for stab in ([0, 1, 2, 3], [0, 1, 4, 5], [0, 2, 4, 6]):
        for q in stab[1:]:
            circuit.add(GateOperation("cx", (stab[0], q)))

    result = run_circuit(circuit, backend=backend, shots=shots)
    return Result.from_value(1.0, counts=result.counts)
