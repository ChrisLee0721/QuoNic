"""Hybrid classical-quantum model interface.

Enables joint training of classical neural networks and quantum circuits
with automatic gradient flow between classical and quantum layers.

Example::

    from quonic.ml import HybridModel, Ansatz
    from quonic.ml import QNNLayer, ClassicalLayer

    model = HybridModel([
        ClassicalLayer(4, 8, activation='relu'),
        QNNLayer(Ansatz.hardware_efficient(8, layers=3)),
        ClassicalLayer(8, 1),
    ])
    model.fit(X_train, y_train, epochs=100)
    predictions = model.predict(X_test)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ansatz import AnsatzBuilder
from .loss import expectation_loss
from .trainer import param_shift_grad


@dataclass
class HybridResult:
    """Result of hybrid model training."""

    params: np.ndarray
    loss_history: list[float]
    final_loss: float
    n_epochs: int


class ClassicalLayer:
    """Classical neural network layer.

    Args:
        in_features: input dimension
        out_features: output dimension
        activation: activation function ('relu', 'sigmoid', 'tanh', 'none')
    """

    def __init__(self, in_features: int, out_features: int, activation: str = "relu"):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weights = np.random.randn(in_features, out_features) * 0.1
        self.bias = np.zeros(out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        out = x @ self.weights + self.bias
        if self.activation == "relu":
            return np.maximum(0, out)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(out, -500, 500)))
        elif self.activation == "tanh":
            return np.tanh(out)
        return out

    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass. Returns (grad_input, grad_weights, grad_bias)."""
        # Apply activation derivative
        if self.activation == "relu":
            mask = (x @ self.weights + self.bias) > 0
            grad_output = grad_output * mask
        elif self.activation == "sigmoid":
            sig = 1 / (1 + np.exp(-np.clip(x @ self.weights + self.bias, -500, 500)))
            grad_output = grad_output * sig * (1 - sig)
        elif self.activation == "tanh":
            tanh_out = np.tanh(x @ self.weights + self.bias)
            grad_output = grad_output * (1 - tanh_out**2)

        grad_weights = x.T @ grad_output
        grad_bias = grad_output.sum(axis=0)
        grad_input = grad_output @ self.weights.T
        return grad_input, grad_weights, grad_bias

    @property
    def n_params(self) -> int:
        return self.in_features * self.out_features + self.out_features


class QNNLayer:
    """Quantum neural network layer.

    Args:
        ansatz: ansatz builder
        n_qubits: number of qubits (default: from ansatz)
        observable: measurement observable
    """

    def __init__(self, ansatz: AnsatzBuilder, n_qubits: int | None = None, observable: str = "Z"):
        self.ansatz = ansatz
        self.n_qubits = n_qubits or ansatz.n_params // (ansatz.n_params // 2)  # estimate
        self.observable = observable
        self.params = np.random.randn(ansatz.n_params) * 0.1

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: encode input, apply ansatz, measure."""
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        x_2d = x.reshape(batch_size, -1)

        results = []
        for i in range(batch_size):
            # Encode input as rotation angles
            encoded_params = self.params.copy()
            for j in range(min(len(x_2d[i]), len(encoded_params))):
                encoded_params[j] += x_2d[i][j]

            # Build and measure circuit
            circuit = self.ansatz.build(encoded_params)
            val = expectation_loss(circuit, self.observable[:min(len(self.observable), self.n_qubits)])
            results.append(val)

        return np.array(results).reshape(batch_size, 1)

    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Backward pass. Returns (grad_input, grad_params)."""
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        x_2d = x.reshape(batch_size, -1)

        grad_input = np.zeros_like(x_2d)
        grad_params = np.zeros_like(self.params)

        for i in range(batch_size):
            encoded_params = self.params.copy()
            for j in range(min(len(x_2d[i]), len(encoded_params))):
                encoded_params[j] += x_2d[i][j]

            # Compute gradient w.r.t. parameters
            def loss_fn(p):
                circuit = self.ansatz.build(p)
                return expectation_loss(circuit, self.observable[:min(len(self.observable), self.n_qubits)])

            param_grad = param_shift_grad(loss_fn, encoded_params)
            grad_params += param_grad * grad_output[i]

            # Compute gradient w.r.t. input (through encoding)
            for j in range(min(len(x_2d[i]), len(self.params))):
                grad_input[i, j] = param_grad[j] * (grad_output[i])

        return grad_input.reshape(x.shape), grad_params / batch_size

    @property
    def n_params(self) -> int:
        return self.ansatz.n_params


class HybridModel:
    """Hybrid classical-quantum model.

    Supports arbitrary sequences of classical and quantum layers with
    automatic gradient flow between them.

    Args:
        layers: list of ClassicalLayer and QNNLayer instances
    """

    def __init__(self, layers: list[Any]):
        self.layers = layers

    def fit(
        self,
        X: list[list[float]],
        y: list[float],
        epochs: int = 100,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> HybridResult:
        """Train the hybrid model.

        Args:
            X: training features
            y: training labels
            epochs: number of training epochs
            lr: learning rate
            verbose: print progress

        Returns:
            HybridResult with training history.
        """
        X_arr = np.array(X)
        y_arr = np.array(y).reshape(-1, 1)
        loss_history = []

        for epoch in range(epochs):
            # Forward pass
            activations = [X_arr]
            for layer in self.layers:
                activations.append(layer.forward(activations[-1]))

            # Compute loss
            predictions = activations[-1]
            loss = np.mean((predictions - y_arr) ** 2)
            loss_history.append(loss)

            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch:4d}: loss = {loss:.6f}")

            # Backward pass
            grad = 2 * (predictions - y_arr) / len(X)
            for i in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[i]
                if isinstance(layer, ClassicalLayer):
                    grad, grad_w, grad_b = layer.backward(activations[i], grad)
                    layer.weights -= lr * grad_w
                    layer.bias -= lr * grad_b
                elif isinstance(layer, QNNLayer):
                    grad, grad_params = layer.backward(activations[i], grad)
                    layer.params -= lr * grad_params

        return HybridResult(
            params=self._get_params(),
            loss_history=loss_history,
            final_loss=loss_history[-1] if loss_history else float("inf"),
            n_epochs=len(loss_history),
        )

    def predict(self, X: list[list[float]]) -> np.ndarray:
        """Predict using the trained model.

        Args:
            X: input features

        Returns:
            Predictions array.
        """
        x = np.array(X)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def _get_params(self) -> np.ndarray:
        """Get all parameters as a flat array."""
        params = []
        for layer in self.layers:
            if isinstance(layer, ClassicalLayer):
                params.extend(layer.weights.flatten())
                params.extend(layer.bias.flatten())
            elif isinstance(layer, QNNLayer):
                params.extend(layer.params.flatten())
        return np.array(params)

    def __repr__(self) -> str:
        layer_types = [type(layer).__name__ for layer in self.layers]
        return f"HybridModel(layers=[{', '.join(layer_types)}])"
