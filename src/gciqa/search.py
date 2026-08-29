"""Grover search for GCIQA.

Implements the quantum search component that finds high-probability
conformations satisfying geometric constraints.

Example::

    from gciqa import grover_search, GroverOracle

    results = grover_search(
        oracle=oracle,
        n_qubits=20,
        n_iterations=50,
        n_shots=1000,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    """Result of a Grover search run.

    Attributes:
        measurements: Dict of measured bitstrings to counts.
        n_shots: Total number of measurements.
        n_unique: Number of unique states measured.
        top_states: List of (bitstring, count) sorted by frequency.
    """

    measurements: dict[str, int]
    n_shots: int
    n_unique: int
    top_states: list[tuple[str, int]]

    def decode_coordinates(
        self,
        n_atoms: int,
        bits_per_coord: int = 10,
        coord_range: tuple[float, float] = (-50.0, 50.0),
    ) -> list[dict[str, tuple[float, float, float]]]:
        """Decode measured bitstrings to atomic coordinates.

        Args:
            n_atoms: Number of atoms.
            bits_per_coord: Bits per coordinate.
            coord_range: Physical coordinate range (Angstrom).

        Returns:
            List of conformations, each a dict of atom -> (x, y, z).
        """
        conformations = []
        bits_per_atom = 3 * bits_per_coord
        total_bits = n_atoms * bits_per_atom
        lo, hi = coord_range
        scale = (hi - lo) / (2**bits_per_coord - 1)

        for bitstring, count in self.top_states:
            if len(bitstring) < total_bits:
                continue
            # Reverse to get qubit order (LSB first)
            bits = bitstring[::-1]
            coords = {}
            for i in range(n_atoms):
                start = i * bits_per_atom
                # Reverse each coordinate's bits for int() (MSB first)
                x_bits = bits[start:start+bits_per_coord][::-1]
                y_bits = bits[start+bits_per_coord:start+2*bits_per_coord][::-1]
                z_bits = bits[start+2*bits_per_coord:start+3*bits_per_coord][::-1]

                x = lo + int(x_bits, 2) * scale
                y = lo + int(y_bits, 2) * scale
                z = lo + int(z_bits, 2) * scale
                coords[f"{i}"] = (x, y, z)
            conformations.append(coords)

        return conformations


def grover_search(
    oracle: Any,
    n_qubits: int,
    n_iterations: int | None = None,
    n_shots: int = 1000,
    search_space_size: int | None = None,
    n_solutions: int | None = None,
) -> SearchResult:
    """Run Grover search to find conformations satisfying constraints.

    Args:
        oracle: GroverOracle instance (or quantum circuit).
        n_qubits: Number of qubits.
        n_iterations: Number of Grover iterations. If None, computed
            from search_space_size and n_solutions.
        n_shots: Number of measurement shots.
        search_space_size: Total number of possible states (2^n_qubits if None).
        n_solutions: Estimated number of valid states.

    Returns:
        SearchResult with measured states and counts.

    Raises:
        ImportError: If qiskit is not installed.
    """
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        raise ImportError(
            "qiskit and qiskit-aer are required for Grover search. "
            "Install with: pip install qiskit qiskit-aer"
        )

    # Compute optimal number of iterations
    if n_iterations is None:
        N = search_space_size or 2**n_qubits
        M = n_solutions or max(1, N // 100)  # Assume 1% are valid
        import math
        n_iterations = max(1, int(math.pi / 4 * math.sqrt(N / M)))

    # Build the full Grover circuit
    qc = _build_grover_circuit(oracle, n_qubits, n_iterations)

    # Run on simulator
    simulator = AerSimulator()
    result = simulator.run(qc, shots=n_shots).result()
    counts = result.get_counts(qc)

    # Sort by frequency
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    return SearchResult(
        measurements=counts,
        n_shots=n_shots,
        n_unique=len(counts),
        top_states=sorted_counts[:100],  # Top 100 states
    )


def _build_grover_circuit(
    oracle: Any,
    n_qubits: int,
    n_iterations: int,
) -> Any:
    """Build the complete Grover search circuit.

    Structure:
        |0⟩ → H⊗n → (Oracle → Diffuser) × R → Measure

    The oracle circuit may have more qubits than n_qubits (ancilla qubits).
    Only data qubits get Hadamard initialization, diffusion, and measurement.

    Args:
        oracle: Oracle circuit or GroverOracle.
        n_qubits: Number of data qubits.
        n_iterations: Number of Grover iterations.

    Returns:
        Complete quantum circuit.
    """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    # Get oracle circuit
    if hasattr(oracle, 'build'):
        oracle_circuit = oracle.build()
    else:
        oracle_circuit = oracle

    # Oracle may have ancilla qubits beyond n_qubits
    oracle_n_qubits = oracle_circuit.num_qubits

    # Build circuit with oracle's full qubit count
    qr = QuantumRegister(oracle_n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "c")  # Only measure data qubits
    qc = QuantumCircuit(qr, cr)

    # Initialize: Hadamard only data qubits
    for i in range(n_qubits):
        qc.h(qr[i])

    # Grover iterations
    for _ in range(n_iterations):
        # Apply oracle (acts on all qubits including ancilla)
        qc.compose(oracle_circuit, inplace=True)

        # Diffusion operator on data qubits only
        _add_diffusion(qc, [qr[i] for i in range(n_qubits)])

    # Measure only data qubits
    for i in range(n_qubits):
        qc.measure(qr[i], cr[i])

    return qc


def _add_diffusion(qc: Any, qubits: list) -> None:
    """Add Grover diffusion operator: 2|s⟩⟨s| - I.

    Where |s⟩ = H⊗n|0⟩ is the uniform superposition.

    Args:
        qc: Quantum circuit.
        qubits: List of data qubits to apply diffusion on.
    """
    # H on all qubits
    for q in qubits:
        qc.h(q)

    # X on all qubits
    for q in qubits:
        qc.x(q)

    # Multi-controlled Z
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])

    # X on all qubits
    for q in qubits:
        qc.x(q)

    # H on all qubits
    for q in qubits:
        qc.h(q)


def classical_grover_simulation(
    n_qubits: int,
    oracle_fn: Any,
    n_iterations: int | None = None,
    n_shots: int = 1000,
) -> SearchResult:
    """Classical simulation of Grover search (for testing/validation).

    Simulates the quantum search classically by:
    1. Generating random candidate states
    2. Checking which satisfy the oracle
    3. Amplifying valid states proportionally

    This is NOT quantum - it's a classical approximation for testing.

    Args:
        n_qubits: Number of qubits (defines search space 2^n).
        oracle_fn: Callable that takes a bitstring and returns True/False.
        n_iterations: Ignored (classical simulation).
        n_shots: Number of random samples.

    Returns:
        SearchResult with sampled states.
    """
    import random

    measurements = {}
    for _ in range(n_shots):
        # Generate random state
        bits = ''.join(random.choice('01') for _ in range(n_qubits))

        # Check oracle
        if oracle_fn(bits):
            # Valid state - count it
            measurements[bits] = measurements.get(bits, 0) + 1
        else:
            # Invalid - still record with low probability
            if random.random() < 0.01:  # 1% chance to record invalid
                measurements[bits] = measurements.get(bits, 0) + 1

    sorted_counts = sorted(measurements.items(), key=lambda x: x[1], reverse=True)

    return SearchResult(
        measurements=measurements,
        n_shots=n_shots,
        n_unique=len(measurements),
        top_states=sorted_counts[:100],
    )
