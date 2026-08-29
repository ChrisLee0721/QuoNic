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

import math
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
    backend: str = "qiskit",
) -> SearchResult:
    """Run Grover search to find conformations satisfying constraints.

    Args:
        oracle: GroverOracle instance (or quantum circuit).
        n_qubits: Number of data qubits.
        n_iterations: Number of Grover iterations. If None, computed
            from search_space_size and n_solutions.
        n_shots: Number of measurement shots.
        search_space_size: Total number of possible states (2^n_qubits if None).
        n_solutions: Estimated number of valid states.
        backend: "qiskit" or "pyqpanda3".

    Returns:
        SearchResult with measured states and counts.
    """
    # Compute optimal number of iterations
    if n_iterations is None:
        N = search_space_size or 2**n_qubits
        M = n_solutions or max(1, N // 100)
        n_iterations = max(1, int(math.pi / 4 * math.sqrt(N / M)))

    if backend == "pyqpanda3":
        return _grover_search_pyqpanda3(oracle, n_qubits, n_iterations, n_shots)
    else:
        return _grover_search_qiskit(oracle, n_qubits, n_iterations, n_shots)


def _grover_search_qiskit(
    oracle: Any, n_qubits: int, n_iterations: int, n_shots: int
) -> SearchResult:
    """Run Grover search on qiskit-aer."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    qc = _build_grover_circuit(oracle, n_qubits, n_iterations, backend="qiskit")
    # Only measure data qubits (not ancilla)
    measured = QuantumCircuit(qc.num_qubits, n_qubits)
    measured.compose(qc, inplace=True)
    measured.measure(list(range(n_qubits)), list(range(n_qubits)))
    simulator = AerSimulator()
    result = simulator.run(measured, shots=n_shots).result()
    counts = result.get_counts(measured)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    return SearchResult(
        measurements=counts,
        n_shots=n_shots,
        n_unique=len(counts),
        top_states=sorted_counts[:100],
    )


def _grover_search_pyqpanda3(
    oracle: Any, n_qubits: int, n_iterations: int, n_shots: int
) -> SearchResult:
    """Run Grover search on pyqpanda3."""
    import pyqpanda3 as pq

    qc = _build_grover_circuit(oracle, n_qubits, n_iterations, backend="pyqpanda3")

    # Add measurements on data qubits
    prog = pq.core.QProg()
    prog << qc
    for i in range(n_qubits):
        prog << pq.core.measure(pq.core.Qubit(i), i)

    qvm = pq.core.CPUQVM()
    qvm.run(prog, n_shots)
    counts = qvm.result().get_counts()
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    return SearchResult(
        measurements=counts,
        n_shots=n_shots,
        n_unique=len(counts),
        top_states=sorted_counts[:100],
    )


def _build_grover_circuit(
    oracle: Any,
    n_qubits: int,
    n_iterations: int,
    backend: str = "qiskit",
) -> Any:
    """Build the complete Grover search circuit.

    Structure:
        |0⟩ → H⊗n → (Oracle → Diffuser) × R → Measure

    Args:
        oracle: Oracle circuit or GroverOracle.
        n_qubits: Number of data qubits.
        n_iterations: Number of Grover iterations.
        backend: "qiskit" or "pyqpanda3".

    Returns:
        Complete quantum circuit (qiskit QuantumCircuit or pyqpanda3 QCircuit).
    """
    from ..circuit import QuoNicCircuit

    # Get oracle as QuoNicCircuit
    if hasattr(oracle, 'build'):
        oracle_qc = oracle.build(backend="quonic")
    elif isinstance(oracle, QuoNicCircuit):
        oracle_qc = oracle
    else:
        # Assume it's a qiskit circuit, wrap it
        oracle_qc = oracle

    oracle_n_qubits = oracle_qc.num_qubits

    # Build full Grover circuit as QuoNicCircuit
    grover = QuoNicCircuit(oracle_n_qubits)

    # Initialize: Hadamard on data qubits
    for i in range(n_qubits):
        grover.h(i)

    # Ancilla qubits for decomposed MCZ (if oracle uses them)
    ancilla_qubits = list(range(n_qubits, oracle_n_qubits)) if oracle_n_qubits > n_qubits else None

    # Grover iterations
    for _ in range(n_iterations):
        # Oracle (acts on all qubits including ancilla)
        grover.compose(oracle_qc)

        # Diffuser on data qubits only
        _add_diffusion(grover, list(range(n_qubits)), ancilla_qubits)

    # Export to requested backend
    if backend == "pyqpanda3":
        return grover.to_pyqpanda3()
    else:
        return grover.to_qiskit()


def _add_diffusion(qc: Any, qubits: list, ancillas: list[int] | None = None) -> None:
    """Add Grover diffusion operator: 2|s⟩⟨s| - I.

    Where |s⟩ = H⊗n|0⟩ is the uniform superposition.
    Uses decomposed MCZ when controls > 5.
    """
    for q in qubits:
        qc.h(q)
    for q in qubits:
        qc.x(q)
    controls = qubits[:-1]
    target = qubits[-1]
    if len(controls) > 5 and ancillas:
        qc.mcz_decomposed(controls, target, ancillas)
    else:
        qc.mcz(controls, target)
    for q in qubits:
        qc.x(q)
    for q in qubits:
        qc.h(q)
