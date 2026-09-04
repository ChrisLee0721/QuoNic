"""Deutsch-Jozsa algorithm — determine if a function is constant or balanced.

The Deutsch-Jozsa algorithm demonstrates quantum parallelism: it determines
whether a Boolean function f:{0,1}^n → {0,1} is constant (same output for all
inputs) or balanced (outputs 0 for exactly half the inputs) in a single query.

Boundary conditions:
- Requires n+1 qubits (n input + 1 output)
- Oracle must be provided as a function that applies the oracle to a Circuit
- Classical: worst case 2^(n-1)+1 queries; quantum: 1 query
- Result: all-zero input qubits → constant; otherwise → balanced
- Noise-free assumption: with noise, the all-zero outcome may not be exact

Example::

    from quonic.algorithms import deutsch_jozsa

    # Constant oracle (f(x) = 0 for all x)
    def constant_oracle(circuit, n):
        pass  # do nothing

    # Balanced oracle (f(x) = x_0)
    def balanced_oracle(circuit, n):
        from quonic.ir import GateOperation
        circuit.add(GateOperation("cx", (0, n)))

    result = deutsch_jozsa(2, balanced_oracle, shots=100)
    print(result["is_balanced"])  # True
"""

from __future__ import annotations

from typing import Callable

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result

OracleFn = Callable[[Circuit, int], None]


def deutsch_jozsa(
    n_qubits: int,
    oracle: OracleFn,
    backend: str = "auto",
    shots: int = 100,
) -> Result:
    """Run the Deutsch-Jozsa algorithm.

    Args:
        n_qubits: Number of input qubits.
        oracle: Function that applies the oracle to a Circuit. The oracle acts
            on n input qubits (indices 0..n-1) and 1 output qubit (index n).
        backend: Backend to use.
        shots: Number of measurement shots.

    Returns:
        Result with "is_balanced" in metadata.
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
    # The bitstring includes output qubit (leftmost) + input qubits (rightmost).
    # We check if the rightmost n characters (input qubits) are all zero.

    result = run_circuit(circuit, backend=backend, shots=shots)
    # Extract input qubit bits (rightmost n characters of bitstring)
    all_zero_input = "0" * n
    is_balanced = all(
        bs[-n:] != all_zero_input for bs in result.counts
    )
    return Result.from_value(
        float(is_balanced),
        is_balanced=is_balanced,
        counts=result.counts,
    )
