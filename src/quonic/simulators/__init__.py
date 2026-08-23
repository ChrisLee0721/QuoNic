"""In-house simulation engines (naive version): no backend dependency, numpy only.

Four engines correspond to four simulation methods, for use by the native
backend and scheduler fallback:

- StatevectorEngine   —— 2^n complex amplitudes, exact (including non-Clifford gates)
- StabilizerEngine    —— Clifford tableau, polynomial scaling (basic Clifford gate set only)
- MPSEngine           —— matrix product state, breaks the 2^n memory wall for low-entanglement circuits
- DensityMatrixEngine —— density matrix, supports depolarizing noise
"""

from ._density import DensityMatrixEngine
from ._mps import MPSEngine
from ._stabilizer import StabilizerEngine
from ._statevector import StatevectorEngine

__all__ = [
    "DensityMatrixEngine",
    "MPSEngine",
    "StabilizerEngine",
    "StatevectorEngine",
]
