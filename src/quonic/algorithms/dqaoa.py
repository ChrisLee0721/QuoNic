"""Distributed QAOA — partitioned QAOA for large problems.

Boundary conditions:
- Splits qubits into 2 partitions
- Runs QAOA on each partition independently
- Minimal demonstration of distributed quantum computing concept

Example::

    from quonic.algorithms import dqaoa
    result = dqaoa()
"""

from __future__ import annotations

from ..result import Result
from .qaoa_generic import qaoa


def dqaoa() -> Result:
    """Minimal distributed QAOA demo with 2 partitions."""
    # Partition 1: qubits 0,1 with local cost Z0Z1
    terms1 = [(1.0, "ZI"), (1.0, "IZ")]
    r1 = qaoa(terms1, 2, p=1, maxiter=50)

    # Partition 2: qubits 2,3 with local cost Z2Z3
    terms2 = [(1.0, "ZI"), (1.0, "IZ")]
    r2 = qaoa(terms2, 2, p=1, maxiter=50)

    total_energy = r1.value + r2.value
    return Result.from_value(total_energy, partition1=r1.value, partition2=r2.value)
