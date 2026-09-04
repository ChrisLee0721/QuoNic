"""Simon's algorithm — find the period (hidden bitstring) of a 2-to-1 function.

Given a function f:{0,1}^n → {0,1}^n that satisfies f(x) = f(x⊕s) for a
hidden bitstring s, Simon's algorithm finds s in O(n) queries.

Boundary conditions:
- Requires 2n qubits (n input + n output)
- Oracle must implement a 2-to-1 function with period s
- Classical: O(2^(n/2)) queries (birthday paradox); quantum: O(n) queries
- Post-processing requires solving a system of linear equations mod 2
- Noise-free assumption: measurement errors corrupt the linear system

Example::

    from quonic.algorithms import simon

    # Oracle for secret s = "11" (2 qubits)
    # f(00)=01, f(01)=10, f(10)=10, f(11)=01
    def oracle_11(circuit, n):
        from quonic.ir import GateOperation
        # Copy input to output: cx(0, n), cx(1, n+1)
        circuit.add(GateOperation("cx", (0, n)))
        circuit.add(GateOperation("cx", (1, n + 1)))
        # XOR with s: cx(0, n+1), cx(1, n) for s="11"
        circuit.add(GateOperation("cx", (0, n + 1)))
        circuit.add(GateOperation("cx", (1, n)))

    result = simon(2, oracle_11, shots=200)
    print(result["secret"])  # "11"
"""

from __future__ import annotations

from typing import Callable

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result

OracleFn = Callable[[Circuit, int], None]


def _solve_mod2(equations: list[list[int]], n: int) -> str:
    """Solve a system of linear equations mod 2 via Gaussian elimination."""
    # Build augmented matrix
    mat = [eq[:] for eq in equations]
    # Forward elimination
    row = 0
    for col in range(n):
        # Find pivot
        pivot = None
        for r in range(row, len(mat)):
            if mat[r][col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        mat[row], mat[pivot] = mat[pivot], mat[row]
        for r in range(len(mat)):
            if r != row and mat[r][col] == 1:
                mat[r] = [(mat[r][c] ^ mat[row][c]) for c in range(n + 1)]
        row += 1
    # Check for unique non-trivial solution
    # The null space should have dimension 1 (the secret s)
    pivot_cols = []
    for r in range(min(row, len(mat))):
        for c in range(n):
            if mat[r][c] == 1:
                pivot_cols.append(c)
                break
    free_cols = [c for c in range(n) if c not in pivot_cols]
    if not free_cols:
        return "0" * n
    # Set the free variable to 1, solve for the rest
    s = [0] * n
    s[free_cols[0]] = 1
    for r in range(min(row, len(mat))):
        pivot_col = None
        for c in range(n):
            if mat[r][c] == 1:
                pivot_col = c
                break
        if pivot_col is not None:
            s[pivot_col] = sum(mat[r][c] * s[c] for c in range(n)) % 2
    return "".join(str(s[i]) for i in range(n))


def simon(
    n_qubits: int,
    oracle: OracleFn,
    backend: str = "auto",
    shots: int = 200,
) -> Result:
    """Run Simon's algorithm.

    Args:
        n_qubits: Number of input qubits.
        oracle: Function that applies the 2-to-1 oracle to a Circuit.
            The oracle acts on n input qubits (0..n-1) and n output qubits (n..2n-1).
        backend: Backend to use.
        shots: Number of measurement shots (need ~n independent equations).

    Returns:
        Result with "secret" string in metadata.
    """
    n = n_qubits
    equations: list[list[int]] = []

    for _ in range(shots):
        circuit = Circuit()
        # H on input qubits
        for q in range(n):
            circuit.add(GateOperation("h", (q,)))
        # Oracle
        oracle(circuit, n)
        # H on input qubits
        for q in range(n):
            circuit.add(GateOperation("h", (q,)))
        # Measure input qubits only
        for q in range(n):
            circuit.add(GateOperation("measure", (q,)))

        result = run_circuit(circuit, backend=backend, shots=1)
        outcome = next(iter(result.counts))
        y = [int(c) for c in outcome]
        # Only keep non-zero equations
        if any(b == 1 for b in y):
            eq = y + [0]  # augmented: y·s = 0
            if eq not in equations:
                equations.append(eq)
        if len(equations) >= n:
            break

    secret = _solve_mod2(equations, n) if equations else "0" * n
    return Result.from_value(float(int(secret, 2)), secret=secret, equations=equations)
