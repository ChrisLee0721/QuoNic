"""Origin Quantum (pyqpanda3) backend for GCIQA.

Provides Grover search execution on Origin Quantum simulators
and hardware via pyqpanda3.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class PyQPanda3Result:
    """Result from pyqpanda3 Grover search."""

    counts: dict[str, int]
    n_shots: int
    elapsed: float
    top_states: list[tuple[str, int]]

    @property
    def n_unique(self) -> int:
        return len(self.counts)


def run_grover_pyqpanda3(
    n_data: int,
    valid_states: list[str],
    n_iterations: int | None = None,
    n_shots: int = 1000,
    simulator: str = "cpu",
) -> PyQPanda3Result:
    """Run Grover search on pyqpanda3.

    Args:
        n_data: Number of data qubits.
        valid_states: List of bitstrings to mark (as '0'/'1' strings).
        n_iterations: Grover iterations. If None, uses optimal count.
        n_shots: Number of measurement shots.
        simulator: "cpu" for CPUQVM, "mps" for MPSQVM.

    Returns:
        PyQPanda3Result with measurement counts.
    """
    import pyqpanda3 as pq

    if n_iterations is None:
        N = 2 ** n_data
        n_iterations = max(1, int(math.pi / 4 * math.sqrt(N / max(1, len(valid_states)))))

    data = [pq.core.Qubit(i) for i in range(n_data)]

    # Build oracle (phase flip on valid states)
    oracle = pq.core.QCircuit()
    for state in valid_states:
        for i, bit in enumerate(reversed(state)):
            if bit == '0':
                oracle << pq.core.X(data[i])
        oracle << _build_mcz(data)
        for i, bit in enumerate(reversed(state)):
            if bit == '0':
                oracle << pq.core.X(data[i])

    # Build diffuser (2|s><s| - I)
    diffuser = pq.core.QCircuit()
    for q in data:
        diffuser << pq.core.H(q)
        diffuser << pq.core.X(q)
    diffuser << _build_mcz(data)
    for q in data:
        diffuser << pq.core.X(q)
        diffuser << pq.core.H(q)

    # Build program
    prog = pq.core.QProg()
    for q in data:
        prog << pq.core.H(q)
    for _ in range(n_iterations):
        prog << oracle
        prog << diffuser
    for i in range(n_data):
        prog << pq.core.measure(data[i], i)

    # Run
    if simulator == "mps":
        qvm = pq.core.MPSQVM()
    else:
        qvm = pq.core.CPUQVM()

    t0 = time.time()
    qvm.run(prog, n_shots)
    result = qvm.result()
    counts = result.get_counts()
    elapsed = time.time() - t0

    top = sorted(counts.items(), key=lambda x: -x[1])

    return PyQPanda3Result(
        counts=counts,
        n_shots=n_shots,
        elapsed=elapsed,
        top_states=top,
    )


def _build_mcz(qubits):
    """Build multi-controlled Z gate (no ancilla)."""
    import pyqpanda3 as pq

    qc = pq.core.QCircuit()
    qc << pq.core.H(qubits[-1])
    x_circ = pq.core.QCircuit()
    x_circ << pq.core.X(qubits[-1])
    mcx = x_circ.control(qubits[:-1])
    qc << mcx
    qc << pq.core.H(qubits[-1])
    return qc


def build_grover_oracle_pyqpanda3(
    n_data: int,
    classical_oracle_fn,
) -> tuple:
    """Build a Grover oracle circuit on pyqpanda3 from a classical oracle function.

    For small n_data (<=16), enumerates all bitstrings and marks valid ones.
    For larger n_data, this approach is not feasible - use arithmetic oracle instead.

    Args:
        n_data: Number of data qubits.
        classical_oracle_fn: Function that takes a bitstring and returns True if valid.

    Returns:
        (oracle_circuit, valid_states) tuple.
    """
    import pyqpanda3 as pq

    if n_data > 16:
        raise ValueError(
            f"Enumeration oracle not feasible for {n_data} qubits. "
            "Use arithmetic oracle instead."
        )

    # Find valid states
    valid_states = []
    for i in range(2 ** n_data):
        bitstring = format(i, f'0{n_data}b')
        if classical_oracle_fn(bitstring):
            valid_states.append(bitstring)

    if not valid_states:
        # Empty oracle (no valid states)
        data = [pq.core.Qubit(i) for i in range(n_data)]
        return pq.core.QCircuit(), valid_states

    # Build phase oracle
    data = [pq.core.Qubit(i) for i in range(n_data)]
    oracle = pq.core.QCircuit()

    for state in valid_states:
        for i, bit in enumerate(reversed(state)):
            if bit == '0':
                oracle << pq.core.X(data[i])
        oracle << _build_mcz(data)
        for i, bit in enumerate(reversed(state)):
            if bit == '0':
                oracle << pq.core.X(data[i])

    return oracle, valid_states
