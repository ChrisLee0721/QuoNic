"""QAOA (Quantum Approximate Optimization Algorithm) template for MaxCut.

Given an undirected graph (edge list), variationally solve for the maximum cut
using a p=1 layer QAOA.

Example: a triangle graph (3 vertices, 3 edges)

    from quonic.algorithms import qaoa

    edges = [(0, 1), (1, 2), (0, 2)]
    result = qaoa.qaoa_maxcut(edges, 3)
    print(result["cut"])   # the maximum cut of the triangle = 2
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from .._i18n import tr
from ..result import Result
from ..simulator import StatevectorSimulator


def _pauli_z(i: int, j: int, n: int) -> str:
    s = ["I"] * n
    s[i] = "Z"
    s[j] = "Z"
    return "".join(s)


def _qaoa_state(
    n: int,
    edges: Sequence[tuple[int, int]],
    p: int,
    params: Sequence[float],
) -> StatevectorSimulator:
    gammas = params[:p]
    betas = params[p:]
    sim = StatevectorSimulator(n)
    for q in range(n):
        sim.apply("h", (q,))
    for layer in range(p):
        # Cost layer: exp(-i γ Z_i Z_j) = CX · Rz(2γ) · CX
        for i, j in edges:
            sim.apply("cx", (i, j))
            sim.apply("rz", (j,), (2 * gammas[layer],))
            sim.apply("cx", (i, j))
        # Mixing layer: Rx(2β)
        for q in range(n):
            sim.apply("rx", (q,), (2 * betas[layer],))
    return sim


def qaoa_maxcut(
    edges: list[tuple[int, int]],
    n_qubits: int,
    p: int = 1,
    init_params: Sequence[float] | None = None,
    optimizer: str = "COBYLA",
    maxiter: int = 300,
    record_history: bool = False,
) -> Result:
    """Variationally solve MaxCut for a given graph.

    Args:
        edges: Edge list [(i, j), ...].
        n_qubits: Number of vertices.
        p: Number of QAOA layers (default 1).
        init_params: Initial parameters (length 2p), all 0.1 by default.
        optimizer / maxiter: Passed to scipy.optimize.minimize.
        record_history: When True, record the energy of each step into
            metadata["history"] for plot_energy_convergence to plot the
            convergence curve (off by default).

    Returns: Result (kind="value"); result.value is the approximate maximum cut,
    and metadata contains "params" (optimal parameters), "energy" (Σ<ZiZj>), and
    optionally "history".
    """
    try:
        from scipy.optimize import minimize
    except ImportError as e:
        raise ImportError(tr("err.qaoa_scipy")) from e

    if init_params is None:
        init_params = [0.1] * (2 * p)

    def cost(params: Sequence[float]) -> float:
        sim = _qaoa_state(n_qubits, edges, p, params)
        return sum(sim.expectation(_pauli_z(i, j, n_qubits)) for i, j in edges)

    history: list[float] = []
    callback: Callable[[Any], None] | None = None
    if record_history:
        def callback(xk: Any) -> None:
            history.append(float(cost(xk)))

    result = minimize(
        cost,
        init_params,
        method=optimizer,
        options={"maxiter": maxiter},
        callback=callback,
    )
    energy = float(result.fun)
    cut = (len(edges) - energy) / 2.0
    metadata = {"params": [float(x) for x in result.x], "energy": energy}
    if record_history:
        metadata["history"] = history
    return Result.from_value(cut, **metadata)
