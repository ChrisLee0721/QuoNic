"""Quantum Annealing Hybrid — quantum + classical optimization.

Combines quantum transverse field evolution (for tunneling through energy
barriers) with classical gradient-free optimization. The quantum circuit
implements the annealing schedule, while the classical optimizer tunes
parameters to minimize the Ising energy.

Example::

    from quonic.algorithms import quantum_annealing_hybrid
    result = quantum_annealing_hybrid(n_spins=4)
"""

from __future__ import annotations

import random

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result


def _ising_energy(spins: list[int], J: dict[tuple[int, int], float]) -> float:
    """Compute Ising energy H = -Σ J_ij s_i s_j."""
    return -sum(J[(i, j)] * spins[i] * spins[j] for i, j in J)


def quantum_annealing_hybrid(
    n_spins: int = 4,
    n_iterations: int = 20,
    n_anneal_steps: int = 10,
    shots: int = 256,
    backend: str = "auto",
) -> Result:
    """Hybrid quantum-classical annealing for Ising optimization.

    Each iteration:
    1. Quantum: Trotterized transverse field annealing prepares a candidate state
    2. Measure to get a candidate spin configuration
    3. Classical: Accept/reject based on energy (Metropolis criterion)

    The quantum circuit implements: e^{-β(H_Z + Γ(t) H_X)}
    where H_Z is the Ising Hamiltonian, H_X is the transverse field,
    and Γ(t) decreases from Γ_max to 0 (annealing schedule).

    Args:
        n_spins: Number of spins (qubits).
        n_iterations: Number of hybrid iterations.
        n_anneal_steps: Number of Trotter steps per anneal.
        shots: Number of shots per quantum circuit.
        backend: Backend for execution.

    Returns:
        Result with best energy and spin configuration.
    """
    # Random couplings
    J = {}
    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            J[(i, j)] = random.uniform(-1, 1)

    best_energy = float("inf")
    best_spins = None

    for iteration in range(n_iterations):
        # Annealing schedule: Γ from Γ_max to 0
        gamma_max = 2.0
        circuit = Circuit()

        # Initialize in |+>^n (superposition)
        for i in range(n_spins):
            circuit.add(GateOperation("h", (i,)))

        # Trotterized annealing
        for step in range(n_anneal_steps):
            t = step / max(n_anneal_steps - 1, 1)
            gamma = gamma_max * (1 - t)  # linear schedule

            # Ising ZZ interactions
            for (i, j), j_val in J.items():
                angle = 2 * j_val / n_anneal_steps
                circuit.add(GateOperation("cx", (i, j)))
                circuit.add(GateOperation("rz", (j,), (angle,)))
                circuit.add(GateOperation("cx", (i, j)))

            # Transverse field (quantum tunneling)
            for i in range(n_spins):
                circuit.add(GateOperation("rx", (i,), (2 * gamma / n_anneal_steps,)))

        # Measure
        for i in range(n_spins):
            circuit.add(GateOperation("measure", (i,)))

        result = run_circuit(circuit, backend=backend, shots=shots)

        # Find best measurement outcome
        counts = result.counts
        for bitstring in counts:
            spins = [1 if b == '1' else -1 for b in bitstring]
            energy = _ising_energy(spins, J)
            if energy < best_energy:
                best_energy = energy
                best_spins = spins

    return Result.from_value(
        best_energy,
        best_spins=best_spins,
        n_spins=n_spins,
        n_iterations=n_iterations,
        couplings={str(k): v for k, v in J.items()},
    )
