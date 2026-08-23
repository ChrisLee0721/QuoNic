"""Loss functions for variational quantum algorithms.

Example::

    from quonic.ml import expectation_loss, fidelity_loss
    loss = expectation_loss(circuit, "ZZII")
    loss = fidelity_loss(circuit, target_state)
"""

from __future__ import annotations

import numpy as np

from ..backends import get_backend
from ..ir import Circuit
from ..statevector import StateVector


def expectation_loss(
    circuit: Circuit,
    observable: str,
    backend: str = "native",
) -> float:
    """Compute expectation value loss: <ψ|O|ψ>.

    Args:
        circuit: quantum circuit
        observable: Pauli string (e.g. "ZZII")
        backend: simulation backend

    Returns:
        Expectation value (to be minimized).
    """
    sv = get_backend(backend).run(circuit, return_state=True)
    return sv.expectation(observable)


def fidelity_loss(
    circuit: Circuit,
    target: StateVector | np.ndarray,
    backend: str = "native",
) -> float:
    """Compute fidelity loss: 1 - |<ψ|φ>|².

    Args:
        circuit: quantum circuit
        target: target state (StateVector or numpy array)
        backend: simulation backend

    Returns:
        Fidelity loss (to be minimized). 0 = perfect match.
    """
    sv = get_backend(backend).run(circuit, return_state=True)
    if isinstance(target, np.ndarray):
        target = StateVector(target)
    return 1.0 - sv.fidelity(target)


def cross_entropy_loss(
    circuit: Circuit,
    target_probs: np.ndarray,
    backend: str = "native",
    shots: int = 1024,
) -> float:
    """Compute cross-entropy loss between measured and target distributions.

    Args:
        circuit: quantum circuit
        target_probs: target probability distribution
        backend: simulation backend
        shots: number of shots

    Returns:
        Cross-entropy loss (to be minimized).
    """
    result = get_backend(backend).run(circuit, shots=shots)
    n = circuit.num_qubits
    measured_probs = np.zeros(2**n)
    for bs, count in result.counts.items():
        measured_probs[int(bs, 2)] = count / shots

    # Avoid log(0)
    measured_probs = np.clip(measured_probs, 1e-10, 1.0)
    target_probs = np.clip(target_probs, 1e-10, 1.0)

    return -np.sum(target_probs * np.log(measured_probs))
