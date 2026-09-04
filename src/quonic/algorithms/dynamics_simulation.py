"""Dynamics Simulation — simulate time-dependent quantum evolution.

Boundary conditions:
- Time-dependent Hamiltonian H(t)
- Uses piecewise-constant approximation
- Minimal: linearly ramped transverse field

Example::

    from quonic.algorithms import dynamics_simulation
    result = dynamics_simulation()
"""

from __future__ import annotations

import math

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result


def dynamics_simulation(
    n_steps: int = 10,
    backend: str = "auto",
    shots: int = 1024,
) -> Result:
    """Minimal dynamics simulation demo."""
    circuit = Circuit()

    # Initial state: |+>
    circuit.add(GateOperation("h", (0,)))

    # Time-dependent evolution: H(t) = (1-t)·X + t·Z
    for i in range(n_steps):
        t = i / n_steps
        # X component (decreasing)
        angle_x = (1 - t) * math.pi / n_steps
        circuit.add(GateOperation("rx", (0,), (angle_x,)))
        # Z component (increasing)
        angle_z = t * math.pi / n_steps
        circuit.add(GateOperation("rz", (0,), (angle_z,)))

    return run_circuit(circuit, backend=backend, shots=shots)
