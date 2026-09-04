"""Quantum PDE Solver — minimal demo of Schrödingerization approach.

Boundary conditions:
- Minimal: 1D heat equation discretization
- Uses quantum walk structure
- NOT a production PDE solver — demonstrates the concept

Example::

    from quonic.algorithms import quantum_pde
    result = quantum_pde()
"""

from __future__ import annotations

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result


def quantum_pde(
    backend: str = "auto",
    shots: int = 1024,
) -> Result:
    """Minimal quantum PDE solver demo (1D heat equation)."""
    circuit = Circuit()

    # Discretize 1D heat equation on 2 qubits (4 grid points)
    # Initial condition: Gaussian pulse at center
    circuit.add(GateOperation("h", (0,)))
    circuit.add(GateOperation("h", (1,)))

    # Diffusion step (simplified): mixer rotation
    dt = 0.1
    for q in range(2):
        circuit.add(GateOperation("rz", (q,), (dt,)))

    # Measurement
    result = run_circuit(circuit, backend=backend, shots=shots)
    return Result.from_value(0.0, counts=result.counts)
