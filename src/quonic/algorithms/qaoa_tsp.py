"""QAOA for Traveling Salesman Problem (TSP).

Encodes TSP as an Ising Hamiltonian and solves with QAOA.

Boundary conditions:
- n cities require n² qubits (binary encoding of permutation matrix)
- Constraint penalties must be large enough to enforce valid tours
- Optimal for small instances (n≤4) due to exponential qubit count
- Returns approximate shortest tour length

Example::

    from quonic.algorithms import qaoa_tsp

    # 3-city triangle with distances
    distances = {(0,1): 1, (1,2): 1, (0,2): 2}
    result = qaoa_tsp(distances, 3, p=1)
    print(result["tour_length"])
"""

from __future__ import annotations

from ..result import Result
from .qaoa_generic import qaoa


def _tsp_hamiltonian(
    distances: dict[tuple[int, int], float],
    n_cities: int,
    penalty: float = 10.0,
) -> tuple[list[tuple[float, str]], int]:
    """Build TSP cost Hamiltonian.

    Qubit encoding: q[i*n + t] = 1 if city i visited at time t.
    Cost = sum of distances for consecutive visits + penalty for invalid tours.
    """
    n = n_cities
    n_qubits = n * n
    terms: list[tuple[float, str]] = []

    # Distance cost: sum_{i,j,t} d_{ij} * x_{i,t} * x_{j,t+1}
    for (i, j), d in distances.items():
        for t in range(n):
            t_next = (t + 1) % n
            # x_{i,t} * x_{j,t+1} = (I - Z_{i,t})/2 * (I - Z_{j,t+1})/2
            # = (I - Z_{i,t} - Z_{j,t+1} + Z_{i,t}Z_{j,t+1}) / 4
            qi = i * n + t
            qj = j * n + t_next
            pauli_zi = ["I"] * n_qubits
            pauli_zj = ["I"] * n_qubits
            pauli_zizj = ["I"] * n_qubits
            pauli_zi[qi] = "Z"
            pauli_zj[qj] = "Z"
            pauli_zizj[qi] = "Z"
            pauli_zizj[qj] = "Z"
            terms.append((d / 4, "".join(pauli_zizj)))
            terms.append((-d / 4, "".join(pauli_zi)))
            terms.append((-d / 4, "".join(pauli_zj)))
            terms.append((d / 4, "I" * n_qubits))

    # Constraint: each city visited exactly once
    for i in range(n):
        for t1 in range(n):
            for t2 in range(t1 + 1, n):
                qi1 = i * n + t1
                qi2 = i * n + t2
                pauli = ["I"] * n_qubits
                pauli[qi1] = "Z"
                pauli[qi2] = "Z"
                terms.append((penalty / 4, "".join(pauli)))

    # Constraint: each time slot has exactly one city
    for t in range(n):
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                qi1 = i1 * n + t
                qi2 = i2 * n + t
                pauli = ["I"] * n_qubits
                pauli[qi1] = "Z"
                pauli[qi2] = "Z"
                terms.append((penalty / 4, "".join(pauli)))

    return terms, n_qubits


def qaoa_tsp(
    distances: dict[tuple[int, int], float],
    n_cities: int,
    p: int = 1,
    penalty: float = 10.0,
    **kwargs,
) -> Result:
    """Solve TSP using QAOA.

    Args:
        distances: Dict mapping (i,j) pairs to distances.
        n_cities: Number of cities.
        p: QAOA layers.
        penalty: Constraint violation penalty.
        **kwargs: Passed to qaoa().

    Returns:
        Result with approximate tour length.
    """
    terms, n_qubits = _tsp_hamiltonian(distances, n_cities, penalty)
    result = qaoa(terms, n_qubits, p=p, **kwargs)
    # Parse best solution to extract tour
    return Result.from_value(
        result.value,
        energy=result.value,
        params=result.metadata.get("params"),
    )
