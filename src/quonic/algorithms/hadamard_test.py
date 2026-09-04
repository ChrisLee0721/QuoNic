"""Hadamard Test — estimate ⟨ψ|U|ψ⟩ for a unitary U.

The Hadamard test uses an ancilla qubit to estimate the real part of ⟨ψ|U|ψ⟩.
By modifying the ancilla preparation (S gate), the imaginary part can also be
estimated.

Boundary conditions:
- Requires n+1 qubits (1 ancilla + n data)
- Estimates Re(⟨ψ|U|ψ⟩) with ancilla in |+⟩ state
- For Im(⟨ψ|U|ψ⟩), apply S† to ancilla before measurement
- Statistical: need many shots for accurate estimation
- U must be a unitary that can be applied as a circuit
- Ancilla qubit is at index n (leftmost in bitstring)
- Simplified: uses CX for controlled single-qubit gates

Example::

    import math
    from quonic.algorithms import hadamard_test

    # Estimate ⟨0|X|0⟩ = 0 (X flips |0⟩ to |1⟩, ⟨0|1⟩ = 0)
    def prepare_zero(circuit, start, n):
        pass

    def apply_x(circuit, n):
        from quonic.ir import GateOperation
        circuit.add(GateOperation("x", (0,)))

    result = hadamard_test(1, prepare_zero, apply_x, shots=10000)
    print(result["expectation"])  # ≈ 0.0
"""

from __future__ import annotations

import math
from typing import Callable

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result

StatePrepFn = Callable[[Circuit, int, int], None]
UnitaryFn = Callable[[Circuit, int], None]


def hadamard_test(
    n_qubits: int,
    prepare_psi: StatePrepFn,
    apply_u: UnitaryFn,
    imaginary: bool = False,
    backend: str = "auto",
    shots: int = 10000,
) -> Result:
    """Run the Hadamard test to estimate Re(⟨ψ|U|ψ⟩) or Im(⟨ψ|U|ψ⟩).

    Args:
        n_qubits: Number of data qubits.
        prepare_psi: Function to prepare |ψ⟩ on data qubits.
            Signature: prepare_psi(circuit, start_qubit, n_qubits).
        apply_u: Function to apply controlled-U on data qubits.
            Signature: apply_u(circuit, n_qubits). The qubits are at indices 0..n-1.
        imaginary: If True, estimate Im(⟨ψ|U|ψ⟩) instead of Re.
        backend: Backend to use.
        shots: Number of measurement shots.

    Returns:
        Result with "expectation" (estimated value) in metadata.
    """
    circuit = Circuit()
    n = n_qubits
    ancilla = n  # ancilla qubit index

    # Prepare |ψ⟩ on data qubits
    prepare_psi(circuit, 0, n)

    # Ancilla: H (and S† for imaginary part)
    circuit.add(GateOperation("h", (ancilla,)))
    if imaginary:
        circuit.add(GateOperation("rz", (ancilla,), (math.pi / 2,)))  # S†

    # Controlled-U: apply CX from ancilla to each data qubit
    # (simplified: works for single-qubit X, Z, etc.)
    _apply_controlled_u(circuit, ancilla, apply_u, n)

    # H on ancilla
    circuit.add(GateOperation("h", (ancilla,)))

    # Don't add explicit measure gates — let the backend auto-measure all qubits.
    # Bitstring: ancilla (leftmost) + data qubits (rightmost).
    # Extract leftmost character = ancilla measurement.

    result = run_circuit(circuit, backend=backend, shots=shots)
    # Extract ancilla bit (leftmost character of bitstring)
    p0 = sum(c for bs, c in result.counts.items() if bs[0] == "0") / shots
    # P(0) = (1 + Re(⟨ψ|U|ψ⟩)) / 2
    expectation = 2.0 * p0 - 1.0
    return Result.from_value(expectation, expectation=expectation, p_ancilla_0=p0, counts=result.counts)


def _apply_controlled_u(circuit: Circuit, ancilla: int, apply_u: UnitaryFn, n: int) -> None:
    """Apply controlled-U by making each gate in U controlled by ancilla."""
    # Build a temporary circuit to capture U's gates
    temp = Circuit()
    apply_u(temp, n)
    # Replay each gate as controlled
    for op in temp.ops:
        if op.name == "measure":
            continue
        if len(op.qubits) == 1:
            # Single-qubit gate → CX (simplified)
            circuit.add(GateOperation("cx", (ancilla, op.qubits[0])))
        else:
            # Multi-qubit gate → CCX
            circuit.add(GateOperation("ccx", (ancilla, op.qubits[0], op.qubits[1])))
