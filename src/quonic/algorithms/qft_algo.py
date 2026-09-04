"""QFT (Quantum Fourier Transform) — standalone algorithm template.

The QFT is the quantum analog of the discrete Fourier transform and is a core
subroutine in Shor's algorithm, QPE, and many other quantum algorithms.

Boundary conditions:
- n qubits, 2^n dimensional state space
- Uses no-swap convention (consistent with QPE)
- Gate set: H + CP (controlled phase) — all backends support these
- Complexity: O(n^2) gates
- Noise-free only: QFT is a unitary transformation, noise model not applicable

Example::

    from quonic.algorithms import qft
    result = qft(3, shots=1024)  # 3-qubit QFT on |000>
"""

from __future__ import annotations

from ..backends import run_circuit
from ..ir import Circuit
from ..qft import add_iqft, add_qft
from ..result import Result


def qft(
    n_qubits: int,
    inverse: bool = False,
    backend: str = "auto",
    shots: int = 1024,
) -> Result:
    """Run a standalone QFT (or inverse QFT) circuit.

    Args:
        n_qubits: Number of qubits.
        inverse: If True, run inverse QFT instead.
        backend: Backend to use.
        shots: Number of measurement shots.

    Returns:
        Result with measurement counts.
    """
    circuit = Circuit()
    if inverse:
        add_iqft(circuit, tuple(range(n_qubits)))
    else:
        add_qft(circuit, tuple(range(n_qubits)))
    return run_circuit(circuit, backend=backend, shots=shots)
