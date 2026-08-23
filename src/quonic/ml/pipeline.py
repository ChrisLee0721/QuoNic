"""End-to-end QML pipeline: encode → ansatz → train → predict.

Provides a high-level interface for variational quantum machine learning.

Example::

    from quonic.ml import QMLPipeline

    pipeline = QMLPipeline(n_qubits=4, layers=3, ansatz="hardware_efficient")
    pipeline.fit(X_train, y_train, maxiter=100)
    predictions = pipeline.predict(X_test)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ansatz import Ansatz
from .loss import expectation_loss
from .optimizer import AdamOptimizer, SPSAOptimizer
from .trainer import TrainResult, train


@dataclass
class QMLResult:
    """Result of a QML pipeline.

    Args:
        train_result: training optimization result
        predictions: predicted values (for regression) or class labels
    """

    train_result: TrainResult
    predictions: np.ndarray | None = None


class QMLPipeline:
    """End-to-end quantum machine learning pipeline.

    Args:
        n_qubits: number of qubits
        layers: number of ansatz layers
        ansatz: ansatz type ("hardware_efficient", "qaoa", "uccsd")
        optimizer: optimizer type ("adam", "spsa")
        lr: learning rate
    """

    def __init__(
        self,
        n_qubits: int = 4,
        layers: int = 2,
        ansatz: str = "hardware_efficient",
        optimizer: str = "adam",
        lr: float = 0.01,
    ):
        self.n_qubits = n_qubits
        self.layers = layers

        # Build ansatz
        if ansatz == "hardware_efficient":
            self.ansatz = Ansatz.hardware_efficient(n_qubits, layers)
        elif ansatz == "qaoa":
            self.ansatz = Ansatz.qaoa(n_qubits, layers)
        elif ansatz == "uccsd":
            self.ansatz = Ansatz.uccsd(n_qubits)
        else:
            raise ValueError(f"Unknown ansatz: {ansatz}")

        # Build optimizer
        if optimizer == "adam":
            self.optimizer = AdamOptimizer(lr=lr)
        elif optimizer == "spsa":
            self.optimizer = SPSAOptimizer(lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        self.params: np.ndarray | None = None
        self.train_result: TrainResult | None = None

    def fit(
        self,
        X: list[list[float]],
        y: list[float],
        maxiter: int = 100,
        observable: str = "Z",
        gradient: str = "param_shift",
        verbose: bool = False,
    ) -> QMLResult:
        """Train the QML model.

        Args:
            X: training features (list of feature vectors)
            y: training labels
            maxiter: maximum training iterations
            observable: Pauli observable for measurement
            gradient: gradient method ("param_shift", "spsa", "numerical")
            verbose: print progress

        Returns:
            QMLResult with training history.
        """
        self.optimizer.maxiter = maxiter

        def loss_fn(params):
            total_loss = 0.0
            obs = observable * self.n_qubits  # e.g. "Z" → "ZZ" for 2 qubits
            for xi, yi in zip(X, y):
                ansatz_circ = self.ansatz.build(params)
                loss = expectation_loss(ansatz_circ, obs)
                total_loss += (loss - yi) ** 2
            return total_loss / len(X)

        result = train(
            self.ansatz,
            self.optimizer,
            loss_fn,
            gradient=gradient,
            verbose=verbose,
        )

        self.params = result.params
        self.train_result = result

        predictions = self.predict(X)
        return QMLResult(train_result=result, predictions=predictions)

    def predict(self, X: list[list[float]]) -> np.ndarray:
        """Predict using the trained model.

        Args:
            X: input features

        Returns:
            Array of predictions.
        """
        if self.params is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        predictions = []
        obs = "Z" * self.n_qubits
        for xi in X:
            ansatz_circ = self.ansatz.build(self.params)
            pred = expectation_loss(ansatz_circ, obs)
            predictions.append(pred)

        return np.array(predictions)

    def __repr__(self) -> str:
        return (
            f"QMLPipeline(n_qubits={self.n_qubits}, "
            f"layers={self.layers}, "
            f"trained={self.params is not None})"
        )
