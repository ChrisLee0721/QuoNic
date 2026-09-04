"""SWAP Test — compare the overlap of two quantum states.

The SWAP test estimates |⟨ψ|φ⟩|² by using an ancilla qubit and controlled-SWAP
gates. Measurement of the ancilla in |0⟩ gives probability (1+|⟨ψ|φ⟩|²)/2.

Boundary conditions:
- Requires 2n+1 qubits (1 ancilla + 2 registers of n qubits)
- Estimates |⟨ψ|φ⟩|², not the sign of the overlap
- Statistical: need many shots for accurate estimation
- Input states must be prepared by the caller via prepare functions
- Complexity: O(n) CSWAP gates + 1 Hadamard
- Ancilla qubit is at index 2n (leftmost in bitstring)

Example::

    from quonic.algorithms import swap_test

    # Compare |0⟩ and |0⟩ (identical → overlap = 1)
    def prepare_zero(circuit, start, n):
        pass  # |0...0⟩ is the default

    # Compare |0⟩ and |1⟩ (orthogonal → overlap = 0)
    def prepare_one(circuit, start, n):
        from quonic.ir import GateOperation
        circuit.add(GateOperation("x", (start,)))

    result = swap_test(1, prepare_zero, prepare_one, shots=10000)
    print(result["overlap"])  # ≈ 0.0
"""

from __future__ import annotations

from typing import Callable

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result

StatePrepFn = Callable[[Circuit, int, int], None]


def swap_test(
    n_qubits: int,
    prepare_a: StatePrepFn,
    prepare_b: StatePrepFn,
    backend: str = "auto",
    shots: int = 10000,
) -> Result:
    """Run the SWAP test to estimate |⟨ψ|φ⟩|².

    Args:
        n_qubits: Number of qubits per register.
        prepare_a: Function that prepares state |ψ⟩ starting at qubit index 0.
            Signature: prepare_a(circuit, start_qubit, n_qubits).
        prepare_b: Function that prepares state |φ⟩ starting at qubit index n.
            Signature: prepare_b(circuit, start_qubit, n_qubits).
        backend: Backend to use.
        shots: Number of measurement shots.

    Returns:
        Result with "overlap" (estimated |⟨ψ|φ⟩|²) in metadata.
    """
    circuit = Circuit()
    n = n_qubits
    ancilla = 2 * n  # ancilla qubit index

    # Prepare states
    prepare_a(circuit, 0, n)
    prepare_b(circuit, n, n)

    # SWAP test circuit
    circuit.add(GateOperation("h", (ancilla,)))
    for i in range(n):
        # CSWAP: controlled-SWAP on (ancilla, i, n+i)
        # Uses native cswap gate (backends translate via translators/cswap.py)
        circuit.add(GateOperation("cswap", (ancilla, i, n + i)))
    circuit.add(GateOperation("h", (ancilla,)))

    # Don't add explicit measure gates — let the backend auto-measure all qubits.
    # Bitstring: ancilla (leftmost) + data qubits (rightmost).
    # Extract leftmost character = ancilla measurement.

    # Native backend supports cswap via statevector; external backends may not.
    use_backend = backend if backend != "auto" else "native"
    result = run_circuit(circuit, backend=use_backend, shots=shots)
    # Extract ancilla bit (leftmost character of bitstring)
    p0 = sum(c for bs, c in result.counts.items() if bs[0] == "0") / shots
    # P(ancilla=0) = (1 + |⟨ψ|φ⟩|²) / 2
    # → |⟨ψ|φ⟩|² = 2 * P(0) - 1
    overlap = max(0.0, 2.0 * p0 - 1.0)
    return Result.from_value(overlap, overlap=overlap, p_ancilla_0=p0, counts=result.counts)
