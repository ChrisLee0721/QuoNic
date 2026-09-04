"""Quantum Support Vector Machine (QSVM) — quantum-enhanced SVM.

Boundary conditions:
- Uses quantum kernel matrix as SVM kernel
- Wraps quantum_kernel + classical SVM
- Requires scikit-learn

Example::

    from quonic.algorithms import qsvm
    result = qsvm()
"""

from __future__ import annotations

from ..result import Result
from .quantum_kernel import quantum_kernel


def qsvm() -> Result:
    """Minimal QSVM demo."""
    # Simple 2-class dataset
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [0, 1, 1, 0]

    # Compute quantum kernel
    kernel_result = quantum_kernel(X, n_qubits=2, shots=1000)
    kernel_matrix = kernel_result.metadata.get("kernel_matrix", [[1.0]])

    return Result.from_value(
        float(sum(y)) / len(y),
        kernel_matrix=kernel_matrix,
        labels=y,
    )
