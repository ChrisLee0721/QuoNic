"""QAOA for Maximum Independent Set (MIS).

Finds the largest set of vertices in a graph with no two adjacent.

Boundary conditions:
- n vertices require n qubits
- Constraint: no two adjacent vertices both selected
- Uses penalty method to enforce constraints
- Optimal for sparse graphs where MIS is large

Example::

    from quonic.algorithms import qaoa_mis

    # Path graph: 0-1-2
    edges = [(0,1), (1,2)]
    result = qaoa_mis(edges, 3, p=1)
    print(result["mis_size"])  # 2 (vertices 0 and 2)
"""

from __future__ import annotations

from ..result import Result
from .qaoa_generic import qaoa


def _mis_hamiltonian(
    edges: list[tuple[int, int]],
    n_vertices: int,
    penalty: float = 2.0,
) -> list[tuple[float, str]]:
    """Build MIS cost Hamiltonian.

    Maximize sum of x_i subject to x_i + x_j ≤ 1 for all edges (i,j).
    Equivalent to: minimize -sum(x_i) + penalty * sum(x_i * x_j for edges).
    """
    n = n_vertices
    terms: list[tuple[float, str]] = []

    # Objective: -sum(x_i) = -sum((I - Z_i)/2) = constant + sum(Z_i)/2
    for i in range(n):
        pauli = ["I"] * n
        pauli[i] = "Z"
        terms.append((0.5, "".join(pauli)))

    # Constraint: penalty * x_i * x_j for each edge
    for i, j in edges:
        # x_i * x_j = (I - Z_i)(I - Z_j)/4 = (I - Z_i - Z_j + Z_iZ_j)/4
        pauli_zi = ["I"] * n
        pauli_zj = ["I"] * n
        pauli_zizj = ["I"] * n
        pauli_zi[i] = "Z"
        pauli_zj[j] = "Z"
        pauli_zizj[i] = "Z"
        pauli_zizj[j] = "Z"
        terms.append((penalty / 4, "".join(pauli_zizj)))
        terms.append((-penalty / 4, "".join(pauli_zi)))
        terms.append((-penalty / 4, "".join(pauli_zj)))

    return terms


def qaoa_mis(
    edges: list[tuple[int, int]],
    n_vertices: int,
    p: int = 1,
    penalty: float = 2.0,
    **kwargs,
) -> Result:
    """Solve MIS using QAOA.

    Args:
        edges: Edge list [(i,j), ...].
        n_vertices: Number of vertices.
        p: QAOA layers.
        penalty: Constraint violation penalty.
        **kwargs: Passed to qaoa().

    Returns:
        Result with MIS size in metadata.
    """
    terms = _mis_hamiltonian(edges, n_vertices, penalty)
    result = qaoa(terms, n_vertices, p=p, **kwargs)
    # MIS size ≈ -energy (approximately)
    mis_size = max(0, -result.value)
    return Result.from_value(mis_size, energy=result.value, params=result.metadata.get("params"))
