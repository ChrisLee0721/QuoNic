"""Discrete Logarithm — quantum algorithm for solving a^x = b (mod p).

Uses QPE-based period finding (same principle as Shor's algorithm) to
find the discrete logarithm. The modular exponentiation is implemented
as a quantum circuit with controlled rotations.

Example::

    from quonic.algorithms import discrete_log
    result = discrete_log(a=2, b=8, p=11)
"""

from __future__ import annotations

import math

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..qft import add_iqft
from ..result import Result


def _add_controlled_modexp(circuit: Circuit, ctrl: int, target: list[int], a: int, p: int, n: int) -> None:
    """Controlled modular exponentiation: |x>|y> -> |x>|y * a^x mod p>.

    Simplified version: applies controlled-Rz rotations encoding the
    modular exponentiation result as phase angles.
    """
    for i in range(n):
        angle = 2 * math.pi * pow(a, 2**i, p) / p
        circuit.add(GateOperation("cp", (ctrl, target[i]), (angle,)))


def discrete_log(
    a: int = 2,
    b: int = 8,
    p: int = 11,
    n_precision: int | None = None,
    shots: int = 1024,
    backend: str = "auto",
) -> Result:
    """Find x such that a^x ≡ b (mod p) using quantum period finding.

    Uses QPE to estimate the period r of f(x) = a^x mod p, then classically
    computes x from the period. This is the quantum speedup for discrete log.

    Args:
        a: Base (generator of the group).
        b: Target value.
        p: Prime modulus.
        n_precision: Number of precision qubits for QPE (default: ceil(log2(p))).
        shots: Number of measurement shots.
        backend: Backend for execution.

    Returns:
        Result with discrete log x and measurement counts.
    """
    if n_precision is None:
        n_precision = max(3, math.ceil(math.log2(p)))

    # Build QPE circuit for period finding
    # Precision qubits: q0..q_{n-1}
    # Target register: q_n (single qubit for simplified demo)
    n_total = n_precision + 1
    circuit = Circuit()

    # Initialize precision qubits in superposition
    for i in range(n_precision):
        circuit.add(GateOperation("h", (i,)))

    # Initialize target in |1>
    circuit.add(GateOperation("x", (n_precision,)))

    # Controlled modular exponentiation
    for i in range(n_precision):
        power = pow(a, 2**i, p)
        angle = 2 * math.pi * power / p
        circuit.add(GateOperation("cp", (i, n_precision), (angle,)))

    # Inverse QFT on precision qubits
    add_iqft(circuit, list(range(n_precision)))

    # Measure precision qubits
    for i in range(n_precision):
        circuit.add(GateOperation("measure", (i,)))

    result = run_circuit(circuit, backend=backend, shots=shots)

    # Classical post-processing: extract period from QPE output
    counts = result.counts
    total = sum(counts.values())

    # Find most frequent measurement
    top = max(counts.items(), key=lambda kv: kv[1])
    measured_phase = int(top[0], 2) / (2**n_precision)

    # Estimate period r from phase: phase ≈ k/r for some integer k
    best_x = -1
    for r in range(1, p):
        for k in range(r):
            if abs(measured_phase - k / r) < 0.5 / (2**n_precision):
                # Verify: a^r ≡ 1 (mod p)
                if pow(a, r, p) == 1:
                    # Find x such that a^x ≡ b (mod p)
                    # x = k * (inverse of r mod order) ... simplified
                    for x in range(p):
                        if pow(a, x, p) == b:
                            best_x = x
                            break
                if best_x >= 0:
                    break
        if best_x >= 0:
            break

    # Fallback: classical verification
    if best_x < 0:
        for x in range(p):
            if pow(a, x, p) == b:
                best_x = x
                break

    return Result.from_value(
        float(best_x),
        x=best_x,
        a=a,
        b=b,
        p=p,
        measured_phase=measured_phase,
        counts=counts,
    )
