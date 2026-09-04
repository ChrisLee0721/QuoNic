"""Quantum Eigenvalue Solver — minimal demo using QPE.

Boundary conditions:
- Uses QPE to estimate eigenvalues of a unitary
- Minimal: 1-qubit unitary with known eigenvalues
- Requires QPE infrastructure

Example::

    from quonic.algorithms import quantum_eigenvalue
    result = quantum_eigenvalue()
"""

from __future__ import annotations

import math

from ..result import Result
from .qpe import qpe


def quantum_eigenvalue() -> Result:
    """Minimal quantum eigenvalue solver demo."""
    # Estimate eigenphase of Rz(π/2) on |1>
    # Rz(π/2)|1> = e^{iπ/4}|1>, so eigenphase = π/4
    return qpe(math.pi / 2, n_precision=4, shots=1024)
