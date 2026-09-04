"""Quantum Walk — quantum analog of random walk on a graph.

Discrete-time quantum walk on a cycle graph with a coin operator.
The walker moves left or right depending on the coin state.

Boundary conditions:
- Position register: n qubits (2^n positions)
- Coin register: 1 qubit
- Total: n+1 qubits
- Cycle graph: positions wrap around (mod 2^n)
- H coin gives symmetric walk; other coins give biased walks
- Walk preserves unitarity (no decoherence)

Example::

    from quonic.algorithms import quantum_walk

    result = quantum_walk(n_positions=3, steps=5, shots=1024)
    print(result.counts)  # distribution over positions
"""

from __future__ import annotations

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result


def quantum_walk(
    n_positions: int,
    steps: int = 10,
    coin: str = "h",
    backend: str = "auto",
    shots: int = 1024,
) -> Result:
    """Run a discrete-time quantum walk on a cycle graph.

    Args:
        n_positions: Number of position qubits (2^n positions).
        steps: Number of walk steps.
        coin: Coin operator ("h" for Hadamard, "x" for NOT).
        backend: Backend to use.
        shots: Number of measurement shots.

    Returns:
        Result with position distribution.
    """
    n = n_positions
    coin_qubit = n  # coin is the highest qubit
    circuit = Circuit()

    for _ in range(steps):
        # Coin flip
        if coin == "h":
            circuit.add(GateOperation("h", (coin_qubit,)))
        elif coin == "x":
            circuit.add(GateOperation("x", (coin_qubit,)))

        # Conditional shift: if coin=|1>, shift position right by 1
        # Increment position register controlled by coin qubit
        for i in range(n - 1, 0, -1):
            circuit.add(GateOperation("ccx", (coin_qubit, i - 1, i)))
        circuit.add(GateOperation("cx", (coin_qubit, 0)))

    return run_circuit(circuit, backend=backend, shots=shots)
