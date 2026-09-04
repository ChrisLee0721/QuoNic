"""Variational Quantum Classifier (VQC) — quantum neural network for classification.

Boundary conditions:
- Parameterized quantum circuit as classifier
- Binary classification (0 or 1)
- Uses angle encoding for features
- Requires scipy for optimization
- Training loop not included (single inference call)

Example::

    from quonic.algorithms import vqc
    result = vqc(features=[0.5, 0.3], params=[1.0, 2.0, 0.5, 1.5], shots=1000)
    print(result["prediction"])  # 0 or 1
"""

from __future__ import annotations

from collections.abc import Sequence

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result


def _vqc_circuit(features: list[float], params: Sequence[float], n_qubits: int) -> Circuit:
    """Build VQC circuit: angle encoding + variational layers."""
    circuit = Circuit()

    # Angle encoding: Ry(feature_i) on qubit i
    for i, f in enumerate(features[:n_qubits]):
        circuit.add(GateOperation("ry", (i,), (f,)))

    # Variational layer: Ry(params) + entangling
    for i in range(n_qubits):
        circuit.add(GateOperation("ry", (i,), (params[i],)))
    for i in range(n_qubits - 1):
        circuit.add(GateOperation("cx", (i, i + 1)))

    # Second variational layer
    for i in range(n_qubits):
        circuit.add(GateOperation("ry", (i,), (params[n_qubits + i],)))

    return circuit


def vqc(
    features: list[float],
    params: Sequence[float],
    n_qubits: int | None = None,
    backend: str = "auto",
    shots: int = 1000,
) -> Result:
    """Run VQC inference.

    Args:
        features: Input features (length ≤ n_qubits).
        params: Variational parameters (length = 2 * n_qubits).
        n_qubits: Number of qubits (default: len(features)).
        backend: Backend to use.
        shots: Number of shots.

    Returns:
        Result with prediction (0 or 1) and probability.
    """
    if n_qubits is None:
        n_qubits = len(features)

    circuit = _vqc_circuit(features, params, n_qubits)
    result = run_circuit(circuit, backend=backend, shots=shots)

    # Classification: majority vote on first qubit
    p1 = sum(c for bs, c in result.counts.items() if bs[-1] == "1") / shots
    prediction = 1 if p1 > 0.5 else 0
    return Result.from_value(float(prediction), prediction=p1, counts=result.counts)
