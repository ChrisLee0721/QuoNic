"""Quantum Fourier transform (QFT) and its inverse.

The qubits argument is a list of qubit indices; the first element is treated as the least-significant bit.
Uses the no-swap convention (consistent with QPE).
"""

from __future__ import annotations

import math

from .ir import Circuit, GateOperation


def _add_cp(circuit: Circuit, c: int, t: int, phi: float) -> None:
    # controlled phase CP(phi)=diag(1,1,1,e^{i phi}); the backend supports the native cp gate directly
    circuit.add(GateOperation("cp", (c, t), (phi,)))


def add_qft(circuit: Circuit, qubits: tuple[int, ...]) -> None:
    """Forward QFT (no swap, qubits[0] is the least-significant bit)."""
    n = len(qubits)
    for j in range(n - 1, -1, -1):
        circuit.add(GateOperation("h", (qubits[j],)))
        for k in range(j - 1, -1, -1):
            _add_cp(circuit, qubits[k], qubits[j], math.pi / 2 ** (j - k))


def add_iqft(circuit: Circuit, qubits: tuple[int, ...]) -> None:
    """Inverse QFT (no swap), the inverse of add_qft."""
    n = len(qubits)
    for j in range(n):
        for k in range(j):
            _add_cp(circuit, qubits[k], qubits[j], -math.pi / 2 ** (j - k))
        circuit.add(GateOperation("h", (qubits[j],)))
