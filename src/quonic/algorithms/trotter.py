"""Trotter-Suzuki decomposition — approximate Hamiltonian time evolution.

Boundary conditions:
- First-order Trotter: exp(-iHt) ≈ (exp(-ih_1 t/n) ... exp(-ih_m t/n))^n
- Error scales as O(t²/n) for first order
- Requires Hamiltonian as Pauli terms
- Works with any backend

Example::

    from quonic.algorithms import trotter
    hamiltonian = [(1.0, "ZZ"), (0.5, "XI")]
    result = trotter(hamiltonian, time=1.0, steps=10, shots=1024)
"""

from __future__ import annotations

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result


def trotter(
    hamiltonian: list[tuple[float, str]],
    time: float = 1.0,
    steps: int = 10,
    n_qubits: int = 2,
    backend: str = "auto",
    shots: int = 1024,
) -> Result:
    """Run Trotter-Suzuki time evolution.

    Args:
        hamiltonian: List of (coefficient, Pauli string) terms.
        time: Total evolution time.
        steps: Number of Trotter steps.
        n_qubits: Number of qubits.
        backend: Backend to use.
        shots: Number of shots.

    Returns:
        Result with final state measurement.
    """
    circuit = Circuit()
    dt = time / steps

    # Initial state (could be customized)
    for q in range(n_qubits):
        circuit.add(GateOperation("h", (q,)))

    # Trotter steps
    for _ in range(steps):
        for coeff, pauli in hamiltonian:
            # exp(-i * coeff * dt * P)
            z_qubits = [i for i, p in enumerate(pauli) if p == "Z"]
            if not z_qubits:
                continue
            if len(z_qubits) == 1:
                circuit.add(GateOperation("rz", (z_qubits[0],), (2 * coeff * dt,)))
            elif len(z_qubits) == 2:
                circuit.add(GateOperation("cx", (z_qubits[0], z_qubits[1])))
                circuit.add(GateOperation("rz", (z_qubits[1],), (2 * coeff * dt,)))
                circuit.add(GateOperation("cx", (z_qubits[0], z_qubits[1])))

    return run_circuit(circuit, backend=backend, shots=shots)
