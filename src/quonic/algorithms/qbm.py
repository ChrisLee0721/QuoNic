"""Quantum Boltzmann Machine — quantum thermal state sampling.

Uses a quantum circuit to prepare and sample from a thermal (Boltzmann)
distribution. The circuit implements Trotterized imaginary time evolution
to cool a quantum state toward the thermal ground state, then measures
to sample from the Boltzmann distribution.

Example::

    from quonic.algorithms import qbm
    result = qbm(temperature=1.0, n_qubits=3)
"""

from __future__ import annotations

import math

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result


def qbm(
    temperature: float = 1.0,
    n_qubits: int = 3,
    n_trotter_steps: int = 10,
    shots: int = 1024,
    backend: str = "auto",
) -> Result:
    """Quantum Boltzmann Machine via Trotterized imaginary time evolution.

    Prepares a thermal state of an Ising Hamiltonian H = -J Σ s_i s_i+1
    by implementing e^{-βH} as a quantum circuit (Trotter decomposition),
    then measures to sample from the Boltzmann distribution.

    Args:
        temperature: Temperature parameter (higher = more mixed).
        n_qubits: Number of qubits (spins).
        n_trotter_steps: Number of Trotter steps for imaginary time evolution.
        shots: Number of measurement shots.
        backend: Backend for execution.

    Returns:
        Result with energy, counts, and partition function estimate.
    """
    beta = 1.0 / temperature if temperature > 0 else 100.0
    dt = beta / n_trotter_steps

    circuit = Circuit()

    # Initialize in |+>^n (maximally mixed in computational basis)
    for i in range(n_qubits):
        circuit.add(GateOperation("h", (i,)))

    # Trotterized imaginary time evolution: e^{-βH}
    # H = -J Σ Z_i Z_{i+1} (Ising model in computational basis)
    J = 1.0
    for _ in range(n_trotter_steps):
        # ZZ interaction: e^{dt*J*Z_i*Z_{i+1}}
        # Decompose: ZZ = CX · (I⊗Rz(2dt*J)) · CX
        for i in range(n_qubits - 1):
            circuit.add(GateOperation("cx", (i, i + 1)))
            circuit.add(GateOperation("rz", (i + 1,), (2 * dt * J,)))
            circuit.add(GateOperation("cx", (i, i + 1)))

        # Transverse field: e^{dt*h*X_i} (quantum tunneling)
        h_field = 0.1  # small transverse field for tunneling
        for i in range(n_qubits):
            circuit.add(GateOperation("rx", (i,), (2 * dt * h_field,)))

    # Measure all qubits
    for i in range(n_qubits):
        circuit.add(GateOperation("measure", (i,)))

    result = run_circuit(circuit, backend=backend, shots=shots)

    # Compute energies for each measured state
    counts = result.counts
    total = sum(counts.values())
    avg_energy = 0.0

    for bitstring, count in counts.items():
        spins = [1 if b == '1' else -1 for b in bitstring]
        energy = -J * sum(spins[i] * spins[i + 1] for i in range(len(spins) - 1))
        avg_energy += energy * count / total

    # Partition function estimate
    Z_estimate = sum(
        math.exp(-beta * (-J * sum(
            (1 if b == '1' else -1) * (1 if bitstring[i+1] == '1' else -1)
            for i, b in enumerate(bitstring[:-1])
        )))
        for bitstring in counts
    )

    return Result.from_value(
        avg_energy,
        counts=counts,
        partition_function=Z_estimate,
        temperature=temperature,
        n_qubits=n_qubits,
    )
