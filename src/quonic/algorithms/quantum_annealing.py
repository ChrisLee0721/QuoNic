"""Quantum Annealing — minimal simulation of quantum annealing process.

Boundary conditions:
- Simulates transverse field Ising model annealing
- NOT actual quantum annealing hardware (D-Wave)
- Uses Trotter steps to approximate adiabatic evolution

Example::

    from quonic.algorithms import quantum_annealing
    result = quantum_annealing()
"""

from __future__ import annotations

import math

from ..result import Result
from ..simulator import StatevectorSimulator


def quantum_annealing(
    n_qubits: int = 2,
    steps: int = 20,
) -> Result:
    """Minimal quantum annealing demo."""
    sim = StatevectorSimulator(n_qubits)

    for step in range(steps):
        s = step / steps  # annealing parameter 0 → 1
        # Problem Hamiltonian: Z_i Z_j (increasing)
        sim.apply("rz", (0,), (s * math.pi / 2,))
        sim.apply("rz", (1,), (s * math.pi / 2,))
        sim.apply("cx", (0, 1))
        sim.apply("rz", (1,), (s * math.pi,))
        sim.apply("cx", (0, 1))

        # Mixer Hamiltonian: X_i (decreasing)
        mix_angle = (1 - s) * math.pi / steps
        sim.apply("rx", (0,), (mix_angle,))
        sim.apply("rx", (1,), (mix_angle,))

    return Result.from_value(sim.expectation("ZZ"))
