"""Quantum Matrix Inversion — minimal demo (HHL special case).

Boundary conditions:
- 2x2 diagonal matrix only
- Uses QPE + rotation
- NOT a production solver

Example::

    from quonic.algorithms import quantum_matrix_inversion
    result = quantum_matrix_inversion()
"""

from __future__ import annotations

from ..result import Result
from .hhl import hhl


def quantum_matrix_inversion() -> Result:
    """Minimal quantum matrix inversion demo."""
    # Solve [[2,0],[0,1]] * x = [1,1]
    return hhl(matrix=[[2.0, 0.0], [0.0, 1.0]], vector=[1.0, 1.0])
