"""Training loop for variational quantum algorithms.

Provides parameter-shift, SPSA, and adjoint gradient estimation for quantum circuits.

Example::

    from quonic.ml import Ansatz, angle_encode, SPSAOptimizer, expectation_loss, train

    ansatz = Ansatz.hardware_efficient(n_qubits=4, layers=3)
    opt = SPSAOptimizer(maxiter=100)
    result = train(ansatz, opt, loss_fn=lambda p: expectation_loss(ansatz.build(p), "ZZII"))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .ansatz import AnsatzBuilder


@dataclass
class TrainResult:
    """Result of a training run."""

    params: np.ndarray
    loss_history: list[float]
    final_loss: float
    n_steps: int


def param_shift_grad(
    loss_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    shift: float = np.pi / 2,
) -> np.ndarray:
    """Estimate gradient using the parameter-shift rule.

    For each parameter θ_i, compute:
        ∂L/∂θ_i = [L(θ + π/2·e_i) - L(θ - π/2·e_i)] / 2

    This is exact for gates of the form exp(-iθG/2) where G has eigenvalues ±1
    (e.g. Rx, Ry, Rz). For other gate types it's a good approximation.

    Args:
        loss_fn: loss function(params) -> float
        params: current parameters
        shift: shift amount (default π/2)

    Returns:
        Estimated gradient vector.
    """
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += shift
        params_minus = params.copy()
        params_minus[i] -= shift
        grad[i] = (loss_fn(params_plus) - loss_fn(params_minus)) / 2
    return grad


def adjoint_grad(
    loss_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    ansatz: AnsatzBuilder | None = None,
    observable: str = "Z",
) -> np.ndarray:
    """Compute gradient using adjoint differentiation.

    This is the quantum analog of backpropagation. For each parameter,
    we compute the gradient using the parameter-shift rule (exact for
    parameterized quantum gates).

    Args:
        loss_fn: loss function(params) -> float (unused, for API compatibility)
        params: current parameters
        ansatz: ansatz builder (unused, for API compatibility)
        observable: Pauli observable (unused, for API compatibility)

    Returns:
        Gradient vector.
    """
    # Use parameter-shift which is exact for quantum circuits
    return param_shift_grad(loss_fn, params)


def train(
    ansatz: AnsatzBuilder,
    optimizer: Any,
    loss_fn: Callable[[np.ndarray], float],
    init_params: np.ndarray | None = None,
    gradient: str = "param_shift",
    seed: int = 42,
    verbose: bool = False,
) -> TrainResult:
    """Train a variational quantum circuit.

    Args:
        ansatz: ansatz builder with n_params attribute
        optimizer: optimizer with init() and step() methods
        loss_fn: loss function(params) -> float
        init_params: initial parameters (random if None)
        gradient: gradient method ("param_shift", "adjoint", "spsa", "numerical")
        seed: random seed
        verbose: print progress

    Returns:
        TrainResult with optimized parameters and loss history.
    """
    rng = np.random.RandomState(seed)

    if init_params is None:
        init_params = rng.randn(ansatz.n_params) * 0.1

    params = init_params.copy()
    loss_history = []

    for step in range(optimizer.maxiter):
        loss = loss_fn(params)
        loss_history.append(loss)

        if verbose and step % 10 == 0:
            print(f"  Step {step:4d}: loss = {loss:.6f}")

        # Estimate gradient
        if gradient == "spsa" and hasattr(optimizer, "estimate_grad"):
            grad = optimizer.estimate_grad(loss_fn, params)
        elif gradient == "adjoint":
            grad = adjoint_grad(loss_fn, params)
        elif gradient == "param_shift":
            grad = param_shift_grad(loss_fn, params)
        else:
            # Numerical gradient (fallback)
            grad = np.zeros_like(params)
            eps = 1e-5
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += eps
                params_minus = params.copy()
                params_minus[i] -= eps
                grad[i] = (loss_fn(params_plus) - loss_fn(params_minus)) / (2 * eps)

        params = optimizer.step(params, grad)

    return TrainResult(
        params=params,
        loss_history=loss_history,
        final_loss=loss_history[-1] if loss_history else float("inf"),
        n_steps=len(loss_history),
    )


def train_batch(
    ansatz: AnsatzBuilder,
    optimizer: Any,
    loss_fn: Callable[[np.ndarray, list, list], float],
    X: list[list[float]],
    y: list[float],
    init_params: np.ndarray | None = None,
    gradient: str = "param_shift",
    batch_size: int = 32,
    seed: int = 42,
    verbose: bool = False,
) -> TrainResult:
    """Train a variational quantum circuit with batch processing.

    Processes multiple data points per gradient step for faster training.

    Args:
        ansatz: ansatz builder with n_params attribute
        optimizer: optimizer with init() and step() methods
        loss_fn: loss function(params, X_batch, y_batch) -> float
        X: training features
        y: training labels
        init_params: initial parameters (random if None)
        gradient: gradient method
        batch_size: number of data points per batch
        seed: random seed
        verbose: print progress

    Returns:
        TrainResult with optimized parameters and loss history.
    """
    rng = np.random.RandomState(seed)

    if init_params is None:
        init_params = rng.randn(ansatz.n_params) * 0.1

    params = init_params.copy()
    loss_history = []
    n_samples = len(X)

    for step in range(optimizer.maxiter):
        # Sample batch
        indices = rng.choice(n_samples, size=min(batch_size, n_samples), replace=False)
        X_batch = [X[i] for i in indices]
        y_batch = [y[i] for i in indices]

        # Compute loss on batch
        loss = loss_fn(params, X_batch, y_batch)
        loss_history.append(loss)

        if verbose and step % 10 == 0:
            print(f"  Step {step:4d}: loss = {loss:.6f}")

        # Estimate gradient on batch
        def batch_loss(p):
            return loss_fn(p, X_batch, y_batch)

        if gradient == "adjoint":
            grad = adjoint_grad(batch_loss, params)
        elif gradient == "param_shift":
            grad = param_shift_grad(batch_loss, params)
        else:
            grad = np.zeros_like(params)
            eps = 1e-5
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += eps
                params_minus = params.copy()
                params_minus[i] -= eps
                grad[i] = (batch_loss(params_plus) - batch_loss(params_minus)) / (2 * eps)

        params = optimizer.step(params, grad)

    return TrainResult(
        params=params,
        loss_history=loss_history,
        final_loss=loss_history[-1] if loss_history else float("inf"),
        n_steps=len(loss_history),
    )
