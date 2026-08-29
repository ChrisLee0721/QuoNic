"""Quantum search components for GCIQA.

This subpackage contains the quantum Grover search oracle and backend
implementations. These are separated from the classical core so that
GCIQA can be used as a pure classical algorithm without quantum dependencies.

Usage::

    from gciqa.late import GroverOracle, grover_search
"""

from .oracle import GroverOracle, estimate_oracle_qubits
from .search import SearchResult, grover_search

__all__ = [
    "GroverOracle",
    "SearchResult",
    "estimate_oracle_qubits",
    "grover_search",
]
