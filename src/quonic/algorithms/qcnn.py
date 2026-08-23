"""Quantum Convolutional Neural Network (QCNN).

A variational quantum circuit with convolutional + pooling layers for
binary classification.  Configurable number of layers and qubits.

Boundary conditions:
- Requires scipy for optimization
- Demonstrates the QCNN concept, not a production classifier
- Parameterized circuit with trainable rotation angles

Example::

    from quonic.algorithms.qcnn import qcnn_train
    result = qcnn_train(n_qubits=4, n_layers=2, maxiter=100)
    print(f"Accuracy: {result.metadata['accuracy']:.2%}")
"""

from __future__ import annotations

from ..ir import Circuit, GateOperation
from ..result import Result


def _qcnn_circuit(
    n_qubits: int,
    n_layers: int,
    params: list[float],
) -> Circuit:
    """Build a QCNN circuit with convolutional + pooling layers.

    Structure per layer:
      1. Convolutional: Ry rotations + entangling CX ladder
      2. Pooling: Ry rotations on even qubits (reduce effective qubits)

    The final layer has a single output qubit measured for classification.
    """
    c = Circuit()
    c.allocate(n_qubits)

    param_idx = 0

    for layer in range(n_layers):
        # Convolutional layer: Ry on each qubit + CX ladder
        for q in range(n_qubits):
            c.add(GateOperation("ry", (q,), (params[param_idx],)))
            param_idx += 1

        # Entangling: CX chain
        for q in range(n_qubits - 1):
            c.add(GateOperation("cx", (q, q + 1)))

        # Pooling layer: Ry rotations (reduce information to even qubits)
        for q in range(0, n_qubits, 2):
            c.add(GateOperation("ry", (q,), (params[param_idx],)))
            param_idx += 1

    # Final measurement on qubit 0 (classification output)
    c.add(GateOperation("measure", (0,)))
    return c


def _cost_function(
    params: list[float],
    n_qubits: int,
    n_layers: int,
    X_train: list[list[int]],
    y_train: list[int],
) -> float:
    """Cost function for QCNN training.

    Measures the probability of |0> on the output qubit for each input,
    and computes the mean squared error against labels.
    """
    from ..backends import get_backend

    total_loss = 0.0
    for x, y in zip(X_train, y_train):
        circuit = _qcnn_circuit(n_qubits, n_layers, params)

        # Encode input: apply X gates for |1> pixels
        for i, pixel in enumerate(x):
            if pixel == 1:
                circuit.add(GateOperation("x", (i,)))

        # Run
        result = get_backend("native").run(circuit, shots=1024)

        # P(|0>) on qubit 0
        p0 = sum(v for k, v in result.counts.items() if k[-1] == "0") / 1024

        # MSE loss
        total_loss += (p0 - y) ** 2

    return total_loss / len(X_train)


def qcnn_train(
    n_qubits: int = 4,
    n_layers: int = 2,
    maxiter: int = 100,
    seed: int = 42,
) -> Result:
    """Train a QCNN on a simple binary classification task.

    Task: classify 4-pixel binary images (0=even, 1=odd number of 1s).

    Parameters:
        n_qubits: number of qubits (= number of input features)
        n_layers: number of convolutional + pooling layers
        maxiter: maximum optimization iterations
        seed: random seed for reproducibility

    Returns:
        Result with final loss and accuracy.
    """
    import numpy as np
    from scipy.optimize import minimize

    rng = np.random.RandomState(seed)

    # Generate training data: 4-pixel binary images
    # Label 0 = even number of 1s, label 1 = odd number of 1s
    X_train = []
    y_train = []
    for i in range(2**n_qubits):
        bits = [(i >> j) & 1 for j in range(n_qubits)]
        X_train.append(bits)
        y_train.append(sum(bits) % 2)

    # Count parameters
    n_params = n_layers * (n_qubits + n_qubits // 2)
    init_params = rng.randn(n_params) * 0.5

    # Optimize
    result = minimize(
        _cost_function,
        init_params,
        args=(n_qubits, n_layers, X_train, y_train),
        method="COBYLA",
        options={"maxiter": maxiter},
    )

    # Evaluate accuracy
    final_loss = result.fun
    accuracy = 1.0 - final_loss  # MSE loss → accuracy approximation

    return Result.from_value(
        float(final_loss),
        accuracy=float(accuracy),
        params=result.x.tolist(),
        n_qubits=n_qubits,
        n_layers=n_layers,
    )


def qcnn_demo(maxiter: int = 50) -> Result:
    """Quick QCNN demo with default settings."""
    return qcnn_train(n_qubits=4, n_layers=2, maxiter=maxiter)
