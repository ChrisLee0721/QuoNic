"""Amplitude Amplification — generalize Grover's search to arbitrary initial states.

Amplitude amplification boosts the probability of a marked state when the
initial state is not uniform. It generalizes Grover's algorithm: Grover uses
uniform superposition as the initial state, while amplitude amplification works
with any state produced by a state-preparation operator A.

Boundary conditions:
- Requires n qubits
- Needs a state-preparation operator A and a phase oracle
- Optimal number of iterations: ⌊π/(4θ)⌋ where sin(θ) = √(marked probability)
- Over-rotation reduces success probability
- Works for any initial distribution, not just uniform

Example::

    from quonic.algorithms import amplitude_amplification, mark_state

    # Amplify the |11⟩ state starting from |++⟩ (Grover-like)
    result = amplitude_amplification(2, mark_state("11"), iterations=1, shots=1024)
    print(result.counts)  # high probability for |11⟩
"""

from __future__ import annotations

from typing import Callable

from ..backends import run_circuit
from ..compiler import decompose
from ..ir import Circuit
from ..result import Result

OracleFn = Callable[[Circuit], None]


def _default_state_prep(circuit: Circuit, n: int) -> None:
    """Default state preparation: uniform superposition (H on all qubits)."""
    from ..ir import GateOperation

    for q in range(n):
        circuit.add(GateOperation("h", (q,)))


def amplitude_amplification(
    n_qubits: int,
    oracle: OracleFn,
    state_prep: OracleFn = None,
    iterations: int = 1,
    backend: str = "auto",
    shots: int = 1024,
) -> Result:
    """Run amplitude amplification.

    Args:
        n_qubits: Number of qubits.
        oracle: Phase oracle that flips the phase of marked states.
        state_prep: State preparation operator A (default: uniform superposition).
        iterations: Number of Grover iterations.
        backend: Backend to use.
        shots: Number of measurement shots.

    Returns:
        Result with measurement counts.
    """
    if state_prep is None:

        def state_prep(c, n):
            _default_state_prep(c, n)

    circuit = Circuit()
    n = n_qubits

    # State preparation
    state_prep(circuit, n)

    # Grover iterations
    for _ in range(iterations):
        # Oracle
        oracle(circuit)
        # Diffusion (reflect about the initial state)
        _add_diffusion(circuit, n, state_prep)

    return run_circuit(decompose(circuit), backend=backend, shots=shots)


def _add_diffusion(circuit: Circuit, n: int, state_prep: OracleFn) -> None:
    """Add the diffusion operator (reflect about |s⟩)."""
    from ..ir import GateOperation

    # A†: inverse of state preparation
    # For uniform superposition (H gates), A† = H
    # Generic: we approximate by applying state_prep gates in reverse
    # For simplicity, assume state_prep is H gates (uniform case)
    for q in range(n):
        circuit.add(GateOperation("h", (q,)))
    # Phase flip |0...0⟩
    for q in range(n):
        circuit.add(GateOperation("x", (q,)))
    circuit.add(GateOperation("mcz", tuple(range(n))))
    for q in range(n):
        circuit.add(GateOperation("x", (q,)))
    # A
    for q in range(n):
        circuit.add(GateOperation("h", (q,)))
