"""QAOA for Knapsack Problem.

Given items with weights and values, maximize total value subject to weight constraint.

Boundary conditions:
- n items require n qubits (binary: include or exclude)
- Weight constraint enforced via penalty term
- Best for small instances (n≤10)

Example::

    from quonic.algorithms import qaoa_knapsack

    weights = [2, 3, 4]
    values = [3, 4, 5]
    capacity = 5
    result = qaoa_knapsack(weights, values, capacity, p=1)
    print(result["max_value"])
"""

from __future__ import annotations

from ..result import Result
from .qaoa_generic import qaoa


def _knapsack_hamiltonian(
    weights: list[float],
    values: list[float],
    capacity: float,
    penalty: float = 10.0,
) -> list:
    """Build Knapsack cost Hamiltonian.

    Maximize sum(v_i * x_i) subject to sum(w_i * x_i) ≤ C.
    Using penalty method: minimize -sum(v_i * x_i) + penalty * max(0, sum(w_i*x_i) - C)^2.
    """
    n = len(weights)
    terms = []

    # Objective: -sum(v_i * x_i)
    for i in range(n):
        pauli = ["I"] * n
        pauli[i] = "Z"
        terms.append((values[i] / 2, "".join(pauli)))

    # Constraint: penalty * (sum(w_i*x_i) - C)^2
    # Expand: penalty * [sum(w_i^2 * x_i) + 2*sum(w_i*w_j * x_i*x_j) - 2*C*sum(w_i*x_i) + C^2]
    # Linear terms
    for i in range(n):
        pauli = ["I"] * n
        pauli[i] = "Z"
        coeff = penalty * (weights[i] ** 2 / 4 - weights[i] * capacity / 2)
        terms.append((coeff, "".join(pauli)))

    # Quadratic terms
    for i in range(n):
        for j in range(i + 1, n):
            pauli = ["I"] * n
            pauli[i] = "Z"
            pauli[j] = "Z"
            terms.append((penalty * weights[i] * weights[j] / 4, "".join(pauli)))

    return terms


def qaoa_knapsack(
    weights: list[float],
    values: list[float],
    capacity: float,
    p: int = 1,
    penalty: float = 10.0,
    **kwargs,
) -> Result:
    """Solve Knapsack using QAOA.

    Args:
        weights: Item weights.
        values: Item values.
        capacity: Weight capacity.
        p: QAOA layers.
        penalty: Constraint violation penalty.
        **kwargs: Passed to qaoa().

    Returns:
        Result with approximate max value.
    """
    terms = _knapsack_hamiltonian(weights, values, capacity, penalty)
    n = len(weights)
    result = qaoa(terms, n, p=p, **kwargs)
    return Result.from_value(
        -result.value,  # negate because we minimized
        energy=result.value,
        params=result.metadata.get("params"),
    )
