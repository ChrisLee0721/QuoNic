"""Variational Quantum Regression (VQR) — quantum version of linear regression.

Boundary conditions:
- Uses parameterized circuit to fit data
- MSE loss function
- Requires scipy for optimization
- Minimal 2-parameter demonstration

Example::

    from quonic.algorithms import vqr
    X = [[0.1], [0.5], [0.9]]
    y = [0.2, 0.6, 0.8]
    result = vqr(X, y, maxiter=100)
"""

from __future__ import annotations

from ..result import Result
from ..simulator import StatevectorSimulator


def vqr(
    X: list[list[float]],
    y: list[float],
    n_params: int = 2,
    maxiter: int = 100,
    optimizer: str = "COBYLA",
) -> Result:
    """Run variational quantum regression.

    Args:
        X: Feature vectors.
        y: Target values.
        n_params: Number of variational parameters.
        maxiter: Maximum iterations.
        optimizer: scipy optimizer.

    Returns:
        Result with MSE loss and learned parameters.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as e:
        raise ImportError("scipy required for VQR") from e

    def predict(params, x):
        sim = StatevectorSimulator(1)
        sim.apply("ry", (0,), (params[0] * x[0] + params[1],))
        return sim.expectation("Z")

    def loss(params):
        mse = 0.0
        for xi, yi in zip(X, y):
            pred = predict(params, xi)
            mse += (pred - yi) ** 2
        return mse / len(X)

    import numpy as np
    init_params = np.random.randn(n_params) * 0.1
    result = minimize(loss, init_params, method=optimizer, options={"maxiter": maxiter})

    return Result.from_value(
        float(result.fun),
        params=[float(x) for x in result.x],
    )
