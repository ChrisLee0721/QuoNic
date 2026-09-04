"""Bernstein-Vazirani algorithm — find a hidden bitstring s.

Given an oracle that computes f(x) = s·x (mod 2) for a secret bitstring s,
the Bernstein-Vazirani algorithm finds s in a single query.

Boundary conditions:
- Requires n+1 qubits (n input + 1 output)
- Oracle implements f(x) = s·x (mod 2) via CNOT gates
- Classical: n queries (one per bit); quantum: 1 query
- Output measurement directly gives s
- Noise-free assumption: measurement noise may flip bits

Example::

    from quonic.algorithms import bernstein_vazirani

    # Oracle for secret s = "101"
    def oracle_101(circuit, n):
        from quonic.ir import GateOperation
        circuit.add(GateOperation("cx", (0, n)))  # s[0] = 1
        # s[1] = 0, skip
        circuit.add(GateOperation("cx", (2, n)))  # s[2] = 1

    result = bernstein_vazirani(3, oracle_101, shots=100)
    print(result["secret"])  # "101"
"""

from __future__ import annotations

from typing import Callable

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result

OracleFn = Callable[[Circuit, int], None]


def bernstein_vazirani(
    n_qubits: int,
    oracle: OracleFn,
    backend: str = "auto",
    shots: int = 100,
) -> Result:
    """Run the Bernstein-Vazirani algorithm.

    Args:
        n_qubits: Number of input qubits (= length of secret s).
        oracle: Function that applies the oracle to a Circuit. The oracle
            implements f(x) = s·x (mod 2) using CNOT gates.
        backend: Backend to use.
        shots: Number of measurement shots.

    Returns:
        Result with "secret" string in metadata.
    """
    circuit = Circuit()
    n = n_qubits

    # Prepare: H on input qubits, X then H on output qubit
    for q in range(n):
        circuit.add(GateOperation("h", (q,)))
    circuit.add(GateOperation("x", (n,)))
    circuit.add(GateOperation("h", (n,)))

    # Oracle
    oracle(circuit, n)

    # H on input qubits
    for q in range(n):
        circuit.add(GateOperation("h", (q,)))

    # Don't add explicit measure gates — let the backend auto-measure all qubits.
    # Bitstring: output qubit (leftmost) + input qubits (rightmost).
    # Extract rightmost n characters = secret s.

    result = run_circuit(circuit, backend=backend, shots=shots)
    # Extract input qubit bits (rightmost n characters)
    secret = max(result.counts, key=result.counts.get)[-n:]
    return Result.from_value(float(int(secret, 2)), secret=secret, counts=result.counts)
