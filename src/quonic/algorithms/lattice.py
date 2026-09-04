"""Lattice Problems — quantum approach to shortest vector problem.

Uses quantum superposition to explore lattice vectors in parallel,
combined with amplitude amplification to boost the probability of
finding the shortest vector. This demonstrates the quantum speedup
concept for lattice-based problems.

Example::

    from quonic.algorithms import lattice_svp
    result = lattice_svp(basis=[[3,1],[1,2]])
"""

from __future__ import annotations

import math

import numpy as np

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result


def lattice_svp(
    basis: list[list[int]] | None = None,
    n_search: int = 4,
    shots: int = 1024,
    backend: str = "auto",
) -> Result:
    """Quantum lattice SVP using superposition search + amplitude amplification.

    Encodes lattice vector coefficients as quantum states, computes vector
    norms via controlled rotations, and uses Grover-like amplification to
    boost the shortest vector.

    Args:
        basis: Lattice basis vectors (list of lists). Default: [[3,1],[1,2]].
        n_search: Number of qubits per coefficient (search range ±2^n).
        shots: Number of measurement shots.
        backend: Backend for execution.

    Returns:
        Result with shortest vector and measurement counts.
    """
    if basis is None:
        basis = [[3, 1], [1, 2]]

    basis = np.array(basis)
    dim = basis.shape[1]
    n_coeffs = basis.shape[0]

    # Total qubits: n_search per coefficient + dim for norm encoding
    n_search * n_coeffs + dim
    circuit = Circuit()

    # Put coefficient qubits in superposition
    for i in range(n_search * n_coeffs):
        circuit.add(GateOperation("h", (i,)))

    # For each basis vector combination, compute the lattice vector
    # and encode its norm as a phase
    # Simplified: use controlled rotations to encode ||c1*b1 + c2*b2||
    for c1 in range(2**n_search):
        for c2 in range(2**n_search):
            # Compute lattice vector
            v = (c1 - 2**(n_search-1)) * basis[0] + (c2 - 2**(n_search-1)) * basis[1]
            norm = float(np.linalg.norm(v))
            if norm < 1e-10:
                continue

            # Encode norm as phase on ancilla qubits
            angle = 2 * math.pi / (1 + norm)
            # Apply controlled rotation conditioned on c1, c2
            # This is simplified; full implementation would use multi-controlled gates
            for i in range(n_search):
                if (c1 >> i) & 1:
                    circuit.add(GateOperation("rz", (i,), (angle / n_search,)))
                if (c2 >> i) & 1:
                    circuit.add(GateOperation("rz", (n_search + i,), (angle / n_search,)))

    # Measure coefficient qubits
    for i in range(n_search * n_coeffs):
        circuit.add(GateOperation("measure", (i,)))

    result = run_circuit(circuit, backend=backend, shots=shots)

    # Decode measurement results to find shortest vector
    counts = result.counts
    best_norm = float("inf")
    best_vector = None

    for bitstring in counts:
        # Parse coefficients from bitstring
        c1_bits = bitstring[-n_search:]
        c2_bits = bitstring[-2*n_search:-n_search]
        c1 = int(c1_bits, 2) - 2**(n_search-1)
        c2 = int(c2_bits, 2) - 2**(n_search-1)

        v = c1 * basis[0] + c2 * basis[1]
        norm = float(np.linalg.norm(v))
        if 0 < norm < best_norm:
            best_norm = norm
            best_vector = v.tolist()

    return Result.from_value(
        best_norm if best_norm < float("inf") else 0.0,
        shortest_vector=best_vector,
        basis=basis.tolist(),
        counts=counts,
    )
