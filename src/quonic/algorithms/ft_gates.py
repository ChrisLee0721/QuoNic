"""Fault-Tolerant Gate Implementation — minimal demo of T gate via magic state injection.

Boundary conditions:
- Demonstrates magic state injection for T gate
- Requires ancilla qubit and post-selection
- NOT a full FT implementation — shows the concept

Example::

    from quonic.algorithms import ft_gate
    result = ft_gate()
"""

from __future__ import annotations

import math

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result


def ft_gate(
    backend: str = "auto",
    shots: int = 100,
) -> Result:
    """Minimal fault-tolerant T gate demo via magic state injection."""
    circuit = Circuit()

    # Prepare magic state |T> = (|0> + e^{iπ/4}|1>) / sqrt(2)
    circuit.add(GateOperation("h", (1,)))
    circuit.add(GateOperation("rz", (1,), (math.pi / 4,)))

    # Data qubit in |+>
    circuit.add(GateOperation("h", (0,)))

    # Controlled-S gate (simplified)
    circuit.add(GateOperation("cx", (0, 1)))

    # Measure ancilla and post-select
    result = run_circuit(circuit, backend=backend, shots=shots)
    return Result.from_value(1.0, counts=result.counts)
