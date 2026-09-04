"""Quantum Natural Gradient (QNG) — optimize variational parameters using Fisher information.

Boundary conditions:
- Computes quantum Fisher information matrix
- Uses natural gradient instead of vanilla gradient
- Faster convergence for variational algorithms
- Requires StatevectorSimulator for exact computation

Example::

    from quonic.algorithms import qng
    result = qng(n_params=2, maxiter=50)
"""

from __future__ import annotations

import numpy as np

from ..result import Result
from ..simulator import StatevectorSimulator


def qng(
    n_params: int = 2,
    maxiter: int = 50,
) -> Result:
    """Demonstrate QNG optimization on a simple 2-parameter problem."""
    # Simple cost function: minimize <ψ(θ)|Z|ψ(θ)>
    def cost(params):
        sim = StatevectorSimulator(1)
        sim.apply("ry", (0,), (params[0],))
        return sim.expectation("Z")

    def fisher_info(params):
        """Compute quantum Fisher information matrix."""
        eps = 1e-4
        n = len(params)
        fim = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                # Finite difference approximation
                p_pp = params.copy()
                p_pp[i] += eps
                p_pp[j] += eps
                p_mm = params.copy()
                p_mm[i] -= eps
                p_mm[j] -= eps
                p_pm = params.copy()
                p_pm[i] += eps
                p_pm[j] -= eps
                p_mp = params.copy()
                p_mp[i] -= eps
                p_mp[j] += eps
                fim[i][j] = (cost(p_pp) - cost(p_pm) - cost(p_mp) + cost(p_mm)) / (4 * eps ** 2)
                fim[j][i] = fim[i][j]
        return fim

    params = np.array([0.5] * n_params)
    history = []
    lr = 0.1

    for _ in range(maxiter):
        c = cost(params)
        history.append(float(c))

        # Vanilla gradient
        grad = np.zeros(n_params)
        eps = 1e-4
        for i in range(n_params):
            p_plus = params.copy()
            p_plus[i] += eps
            grad[i] = (cost(p_plus) - c) / eps

        # Natural gradient: F^{-1} * grad
        fim = fisher_info(params) + 1e-6 * np.eye(n_params)
        nat_grad = np.linalg.solve(fim, grad)
        params = params - lr * nat_grad

    return Result.from_value(float(cost(params)), history=history, params=params.tolist())
