"""Quantum Phase Estimation (QPE) template.

Estimate the eigenphase of the single-qubit gate U = Rz(θ) acting on |1>.

Rz(θ)|1> = e^{iθ/2}|1>, so the eigenphase φ = θ/2.
QPE uses n phase bits to estimate j, satisfying j/2^n ≈ φ/(2π) = θ/(4π).

Example:
    import math
    from quonic.algorithms import qpe

    result = qpe(math.pi, n_precision=3, shots=1024)
    # Rz(π)|1> has phase π/2, φ/(2π)=1/4, j≈2 -> phase bits "010"
"""


from __future__ import annotations

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..qft import add_iqft
from ..result import Result


def _add_crz(circuit: Circuit, c: int, t: int, theta: float) -> None:
    # Controlled Rz(theta) (control c, target t)
    circuit.add(GateOperation("cx", (c, t)))
    circuit.add(GateOperation("rz", (t,), (-theta / 2,)))
    circuit.add(GateOperation("cx", (c, t)))
    circuit.add(GateOperation("rz", (t,), (theta / 2,)))


def _add_iqft(circuit: Circuit, n: int) -> None:
    # Inverse quantum Fourier transform (qubit 0 = least significant bit), delegated to the qft module
    add_iqft(circuit, list(range(n)))


def qpe(
    theta: float,
    n_precision: int,
    shots: int = 1024,
    backend: str = "auto",
) -> Result:
    """Estimate the eigenphase of Rz(theta) acting on |1>.

    Args:
        theta: Single-qubit rotation angle (radians).
        n_precision: Number of bits for the phase estimate.
        shots / backend: Sampling parameters.

    Returns: Result (kind="counts"). The rightmost n_precision bits of the
    bitstring are the phase estimate; their integer value j satisfies
    j/2^n ≈ theta/(4π) (n = n_precision).
    """
    n = n_precision
    state_qubit = n
    circuit = Circuit()
    circuit.add(GateOperation("x", (state_qubit,)))
    for j in range(n):
        circuit.add(GateOperation("h", (j,)))
    for j in range(n):
        _add_crz(circuit, j, state_qubit, theta * (2 ** (n - 1 - j)))
    _add_iqft(circuit, n)
    return run_circuit(circuit, backend=backend, shots=shots)
