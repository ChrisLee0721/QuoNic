"""Quantum Kernel — compute kernel matrix using SWAP test.

Boundary conditions:
- Uses SWAP test to estimate |⟨ψ(x)|ψ(x')⟩|²
- Requires 2n+1 qubits per kernel evaluation
- Statistical: need many shots for accurate kernel values
- Can be used with classical SVM

Example::

    from quonic.algorithms import quantum_kernel
    kernel_matrix = quantum_kernel(X_train, n_qubits=2, shots=10000)
"""

from __future__ import annotations

import numpy as np

from ..ir import Circuit, GateOperation
from ..result import Result
from .swap_test import swap_test


def _angle_encode(circuit: Circuit, start: int, features: list[float]) -> None:
    """Encode features as rotation angles."""
    for i, f in enumerate(features):
        circuit.add(GateOperation("ry", (start + i,), (f,)))


def quantum_kernel(
    X: list[list[float]],
    n_qubits: int = 2,
    shots: int = 10000,
) -> Result:
    """Compute quantum kernel matrix for a dataset.

    Args:
        X: List of feature vectors.
        n_qubits: Number of qubits per data point.
        shots: Shots per kernel evaluation.

    Returns:
        Result with kernel matrix in metadata.
    """
    n = len(X)
    kernel_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            def prep_a(circuit, start, nq, _features=X[i]):
                _angle_encode(circuit, start, _features)

            def prep_b(circuit, start, nq, _features=X[j]):
                _angle_encode(circuit, start, _features)

            result = swap_test(n_qubits, prep_a, prep_b, shots=shots)
            kernel_matrix[i][j] = result.metadata["overlap"]
            kernel_matrix[j][i] = result.metadata["overlap"]

    return Result.from_value(float(np.trace(kernel_matrix)), kernel_matrix=kernel_matrix.tolist())
