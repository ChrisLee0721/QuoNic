"""Quantum optimizers — gradient-free and gradient-based optimizers for variational circuits.

Example::

    from quonic.ml import SPSAOptimizer
    opt = SPSAOptimizer(maxiter=100)
    params = opt.init(n_params)
    for step in range(100):
        loss, grad = compute_loss_and_grad(params)
        params = opt.step(params, grad)
"""

from __future__ import annotations

import numpy as np


class SPSAOptimizer:
    """Simultaneous Perturbation Stochastic Approximation (SPSA).

    Gradient-free optimizer that uses random perturbations to estimate the gradient.
    Well-suited for noisy quantum circuits.

    Args:
        maxiter: maximum number of iterations
        lr: learning rate (initial)
        c: perturbation magnitude
        a: learning rate decay
        c_decay: perturbation decay
    """

    def __init__(
        self,
        maxiter: int = 100,
        lr: float = 0.1,
        c: float = 0.1,
        a: float = 1.0,
        c_decay: float = 0.1,
    ):
        self.maxiter = maxiter
        self.lr = lr
        self.c = c
        self.a = a
        self.c_decay = c_decay
        self.step_num = 0

    def init(self, n_params: int) -> np.ndarray:
        """Initialize parameters."""
        return np.random.randn(n_params) * 0.1

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Update parameters using SPSA.

        Args:
            params: current parameters
            grad: estimated gradient (from SPSA or parameter-shift)

        Returns:
            Updated parameters.
        """
        self.step_num += 1
        lr = self.a / (self.step_num + 1) ** 0.6
        return params - lr * grad

    def estimate_grad(
        self, loss_fn, params: np.ndarray, c: float | None = None
    ) -> np.ndarray:
        """Estimate gradient using SPSA.

        Args:
            loss_fn: loss function(params) -> float
            params: current parameters
            c: perturbation magnitude (default: self.c)

        Returns:
            Estimated gradient.
        """
        if c is None:
            c = self.c
        n = len(params)
        delta = np.random.choice([-1, 1], size=n)
        params_plus = params + c * delta
        params_minus = params - c * delta
        loss_plus = loss_fn(params_plus)
        loss_minus = loss_fn(params_minus)
        return (loss_plus - loss_minus) / (2 * c) * delta


class AdamOptimizer:
    """Adam optimizer.

    Args:
        maxiter: maximum number of iterations
        lr: learning rate
        beta1: first moment decay
        beta2: second moment decay
        epsilon: numerical stability
    """

    def __init__(
        self,
        maxiter: int = 100,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.maxiter = maxiter
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.step_num = 0
        self.m = None
        self.v = None

    def init(self, n_params: int) -> np.ndarray:
        """Initialize parameters."""
        return np.random.randn(n_params) * 0.1

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Update parameters using Adam."""
        self.step_num += 1
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

        m_hat = self.m / (1 - self.beta1**self.step_num)
        v_hat = self.v / (1 - self.beta2**self.step_num)

        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class QNGOptimizer:
    """Quantum Natural Gradient optimizer.

    Uses the Fubini-Study metric to precondition the gradient.

    Args:
        maxiter: maximum number of iterations
        lr: learning rate
    """

    def __init__(self, maxiter: int = 100, lr: float = 0.01):
        self.maxiter = maxiter
        self.lr = lr
        self.step_num = 0

    def init(self, n_params: int) -> np.ndarray:
        """Initialize parameters."""
        return np.random.randn(n_params) * 0.1

    def step(self, params: np.ndarray, grad: np.ndarray, metric: np.ndarray = None) -> np.ndarray:
        """Update parameters using quantum natural gradient.

        Args:
            params: current parameters
            grad: gradient
            metric: Fubini-Study metric tensor (if available)

        Returns:
            Updated parameters.
        """
        self.step_num += 1
        if metric is None:
            # Fallback to plain gradient
            return params - self.lr * grad

        # Natural gradient: metric^{-1} @ grad
        try:
            nat_grad = np.linalg.solve(metric, grad)
        except np.linalg.LinAlgError:
            nat_grad = grad

        return params - self.lr * nat_grad
