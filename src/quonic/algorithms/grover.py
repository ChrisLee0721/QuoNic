"""Grover search algorithm template.

The simplest way to search for a computational basis state is to pass the
bitstring directly (mark_state generates the oracle automatically):

    from quonic.algorithms import grover
    result = grover("11", 2, shots=1024)   # search for |11> among 2 bits

You can also provide a custom oracle (a callback that phase-flips the target
state):

    from quonic.algorithms import grover, mark_state
    result = grover(mark_state("11"), 2, shots=1024)
"""

from __future__ import annotations

import math
from typing import Callable

from .._i18n import tr
from ..backends import get_backend
from ..ir import Circuit, GateOperation
from ..result import Result

OracleCallback = Callable[[Circuit], None]


def _add_diffusion(circuit: Circuit, n: int) -> None:
    for q in range(n):
        circuit.add(GateOperation("h", (q,)))
    for q in range(n):
        circuit.add(GateOperation("x", (q,)))
    _add_phase_flip_all_ones(circuit, n)
    for q in range(n):
        circuit.add(GateOperation("x", (q,)))
    for q in range(n):
        circuit.add(GateOperation("h", (q,)))


def _add_phase_flip_all_ones(circuit: Circuit, n: int) -> None:
    # Apply a -1 phase to |11...1> (multi-controlled Z)
    if n == 1:
        circuit.add(GateOperation("z", (0,)))
    else:
        circuit.add(GateOperation("mcz", tuple(range(n))))


def mark_state(bitstring: str) -> OracleCallback:
    """Return an oracle callback that marks the computational basis state |bitstring>.

    The rightmost bit of the bitstring is qubit 0 (consistent with qshow's
    bitstring convention). Example: mark_state("11") marks |11>;
    mark_state("10") marks |10> (qubit0=0, qubit1=1).
    """
    bitstring = str(bitstring)
    if not bitstring or any(ch not in "01" for ch in bitstring):
        raise ValueError(tr("err.mark_state_bitstring", bitstring=bitstring))
    n = len(bitstring)

    def oracle(circuit: Circuit) -> None:
        # Flip the 0 bits of the target state to 1, apply the all-ones phase flip, then flip back
        for q in range(n):
            if bitstring[n - 1 - q] == "0":
                circuit.add(GateOperation("x", (q,)))
        _add_phase_flip_all_ones(circuit, n)
        for q in range(n):
            if bitstring[n - 1 - q] == "0":
                circuit.add(GateOperation("x", (q,)))

    return oracle


def grover(
    oracle: str | OracleCallback,
    n_qubits: int,
    iterations: int | None = None,
    backend: str = "auto",
    shots: int = 1024,
) -> Result:
    """Run Grover search.

    Args:
        oracle: The oracle, either a callback function oracle(circuit) or a
            bitstring (e.g. "11", equivalent to mark_state("11")).
        n_qubits: Number of qubits.
        iterations: Number of iterations, default floor(π/4 · √(2^n)).
        backend: Sampling backend (qiskit / cirq / pennylane).
        shots: Number of samples.

    Returns: Result (kind="counts"); result.counts is the sampling histogram.
    """
    if isinstance(oracle, str):
        if len(oracle) != n_qubits:
            raise ValueError(
                tr("err.oracle_len", oracle=oracle, n=len(oracle), n_qubits=n_qubits)
            )
        oracle = mark_state(oracle)

    circuit = Circuit()
    for q in range(n_qubits):
        circuit.add(GateOperation("h", (q,)))

    if iterations is None:
        iterations = int(math.pi / 4 * math.sqrt(2 ** n_qubits))

    for _ in range(iterations):
        oracle(circuit)
        _add_diffusion(circuit, n_qubits)

    return get_backend(backend).run(circuit, shots=shots)


def diffusion(n_qubits: int) -> Circuit:
    """Append the Grover diffusion operator (2|s><s| - I) to the current circuit.

    Applies the sequence H, X, multi-controlled Z, X, H to qubits 0..n_qubits-1;
    this is the core step of amplitude amplification (Grover iteration) and can be
    combined with qgate / mark_state to build a custom search:
        qgate(H, 0); qgate(H, 1); mark_state("11")(current_circuit()); diffusion(2)
    """
    from ..stack import current_circuit

    circ = current_circuit()
    _add_diffusion(circ, n_qubits)
    return circ
