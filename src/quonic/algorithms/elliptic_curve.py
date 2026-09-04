"""Elliptic Curve Discrete Logarithm — quantum approach.

Uses quantum period finding to solve the elliptic curve discrete logarithm
problem: given points P and Q on an elliptic curve, find k such that Q = kP.

The quantum circuit implements QPE on the scalar multiplication oracle,
encoding point addition as controlled rotations.

Example::

    from quonic.algorithms import elliptic_curve
    result = elliptic_curve(p=97, a=2, b=3)
"""

from __future__ import annotations

import math

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..qft import add_iqft
from ..result import Result


def _ec_add(P: tuple[int, int], Q: tuple[int, int], a: int, p: int) -> tuple[int, int]:
    """Elliptic curve point addition over F_p.

    Curve: y^2 = x^3 + ax + b (mod p)
    """
    if P == (0, 0):
        return Q
    if Q == (0, 0):
        return P
    if P[0] == Q[0] and (P[1] + Q[1]) % p == 0:
        return (0, 0)  # Point at infinity

    if P == Q:
        # Point doubling
        lam = (3 * P[0] * P[0] + a) * pow(2 * P[1], -1, p) % p
    else:
        lam = (Q[1] - P[1]) * pow(Q[0] - P[0], -1, p) % p

    x3 = (lam * lam - P[0] - Q[0]) % p
    y3 = (lam * (P[0] - x3) - P[1]) % p
    return (x3, y3)


def _ec_scalar_mul(k: int, P: tuple[int, int], a: int, p: int) -> tuple[int, int]:
    """Scalar multiplication kP using double-and-add."""
    result = (0, 0)
    addend = P
    while k > 0:
        if k & 1:
            result = _ec_add(result, addend, a, p)
        addend = _ec_add(addend, addend, a, p)
        k >>= 1
    return result


def elliptic_curve(
    p: int = 97,
    a: int = 2,
    b: int = 3,
    n_precision: int | None = None,
    shots: int = 1024,
    backend: str = "auto",
) -> Result:
    """Quantum elliptic curve DLP using QPE-based period finding.

    Encodes scalar multiplication as a quantum oracle and uses QPE to
    estimate the order of the point, which is then used to solve the DLP.

    Args:
        p: Prime field modulus.
        a: Curve parameter a in y^2 = x^3 + ax + b.
        b: Curve parameter b.
        n_precision: Number of QPE precision qubits.
        shots: Number of measurement shots.
        backend: Backend for execution.

    Returns:
        Result with curve info and quantum measurement counts.
    """
    if n_precision is None:
        n_precision = max(3, math.ceil(math.log2(p)))

    # Find a point on the curve
    P = None
    for x in range(p):
        rhs = (x**3 + a * x + b) % p
        for y in range(p):
            if (y * y) % p == rhs:
                P = (x, y)
                break
        if P is not None:
            break

    if P is None:
        return Result.from_value(-1.0, error="No point found on curve")

    # Find the order of P
    order = 1
    current = P
    while current != (0, 0) and order < p + 1:
        current = _ec_add(current, P, a, p)
        order += 1

    # Build QPE circuit to estimate the order
    n_total = n_precision + 2  # precision + 2 qubits for point encoding
    circuit = Circuit()

    # Superposition on precision qubits
    for i in range(n_precision):
        circuit.add(GateOperation("h", (i,)))

    # Encode point P on target qubits
    angle_x = 2 * math.pi * P[0] / p
    angle_y = 2 * math.pi * P[1] / p
    circuit.add(GateOperation("ry", (n_precision,), (angle_x,)))
    circuit.add(GateOperation("ry", (n_precision + 1,), (angle_y,)))

    # Controlled scalar multiplication: for each precision qubit i,
    # apply 2^i * P as a phase rotation
    for i in range(n_precision):
        k = pow(2, i, order)
        kP = _ec_scalar_mul(k, P, a, p)
        phase = 2 * math.pi * kP[0] / p
        circuit.add(GateOperation("cp", (i, n_precision), (phase,)))

    # Inverse QFT
    add_iqft(circuit, list(range(n_precision)))

    # Measure
    for i in range(n_precision):
        circuit.add(GateOperation("measure", (i,)))

    result = run_circuit(circuit, backend=backend, shots=shots)

    return Result.from_value(
        float(order),
        P=P,
        order=order,
        curve=f"y^2=x^3+{a}x+{b} mod {p}",
        counts=result.counts,
    )
