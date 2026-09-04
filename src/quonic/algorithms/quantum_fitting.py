"""Quantum Data Fitting — minimal demo of quantum least squares.

Boundary conditions:
- Minimal: 2-point linear fit
- Uses VQR (Variational Quantum Regression)
- NOT a production fitting tool

Example::

    from quonic.algorithms import quantum_fitting
    result = quantum_fitting()
"""

from __future__ import annotations

from ..result import Result
from .vqr import vqr


def quantum_fitting() -> Result:
    """Minimal quantum data fitting demo."""
    X = [[0.0], [1.0]]
    y = [0.0, 1.0]
    return vqr(X, y, n_params=2, maxiter=50)
