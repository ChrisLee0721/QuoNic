"""QAOA generic framework — solve any combinatorial optimization problem.

The user provides a cost Hamiltonian (as Pauli terms) and an optional mixer
Hamiltonian. The framework runs p layers of alternating cost/mixer unitaries
and variationally optimizes the parameters.

Boundary conditions:
- Cost Hamiltonian must be diagonal in the computational basis (Ising-type)
- Default mixer: transverse field (Rx on each qubit)
- Number of layers p controls approximation quality (p→∞ → exact)
- Uses StatevectorSimulator for exact expectation values
- For sampling-based estimation, use backend parameter (future extension)

Example::

    from quonic.algorithms import qaoa

    # MaxCut on a triangle
    edges = [(0,1), (1,2), (0,2)]
    result = qaoa.qaoa_maxcut(edges, 3, p=2)
    print(result["cut"])
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from .._i18n import tr
from ..result import Result
from ..simulator import StatevectorSimulator


def _qaoa_state(
    n: int,
    cost_terms: list[tuple[float, str]],
    p: int,
    params: Sequence[float],
    mixer_type: str = "rx",
) -> StatevectorSimulator:
    """Build the QAOA state with p layers of cost + mixer unitaries."""
    gammas = params[:p]
    betas = params[p:]
    sim = StatevectorSimulator(n)

    # Initial superposition
    for q in range(n):
        sim.apply("h", (q,))

    for layer in range(p):
        # Cost layer: exp(-i γ H_C) for each Pauli term
        for coeff, pauli in cost_terms:
            _apply_cost_term(sim, pauli, coeff * gammas[layer], n)

        # Mixer layer
        if mixer_type == "rx":
            for q in range(n):
                sim.apply("rx", (q,), (2 * betas[layer],))
        elif mixer_type == "custom":
            pass  # custom mixer applied externally

    return sim


def _apply_cost_term(
    sim: StatevectorSimulator, pauli: str, angle: float, n: int
) -> None:
    """Apply exp(-i·angle·P) for a Pauli string P.

    For ZZ terms: exp(-i·angle·Z_iZ_j) = CX · Rz(2·angle) · CX
    For single Z: exp(-i·angle·Z_i) = Rz(2·angle)
    For I terms: identity (skip)
    """
    z_qubits = [i for i, p in enumerate(pauli) if p == "Z"]
    if not z_qubits:
        return  # identity term
    if len(z_qubits) == 1:
        sim.apply("rz", (z_qubits[0],), (2 * angle,))
    elif len(z_qubits) == 2:
        sim.apply("cx", (z_qubits[0], z_qubits[1]))
        sim.apply("rz", (z_qubits[1],), (2 * angle,))
        sim.apply("cx", (z_qubits[0], z_qubits[1]))
    else:
        # Multi-Z: chain of CX + Rz
        for i in range(len(z_qubits) - 1):
            sim.apply("cx", (z_qubits[i], z_qubits[-1]))
        sim.apply("rz", (z_qubits[-1],), (2 * angle,))
        for i in range(len(z_qubits) - 2, -1, -1):
            sim.apply("cx", (z_qubits[i], z_qubits[-1]))


def qaoa(
    cost_hamiltonian: list[tuple[float, str]],
    n_qubits: int,
    p: int = 1,
    init_params: Sequence[float] | None = None,
    optimizer: str = "COBYLA",
    maxiter: int = 300,
    record_history: bool = False,
) -> Result:
    """Run QAOA with a generic cost Hamiltonian.

    Args:
        cost_hamiltonian: List of (coefficient, Pauli string) terms.
        n_qubits: Number of qubits.
        p: Number of QAOA layers.
        init_params: Initial parameters (length 2p), all 0.1 by default.
        optimizer: scipy.optimize.minimize method.
        maxiter: Maximum iterations.
        record_history: Record energy at each step.

    Returns:
        Result with optimal energy in value, params in metadata.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as e:
        raise ImportError(tr("err.qaoa_scipy")) from e

    if init_params is None:
        init_params = [0.1] * (2 * p)

    def cost(params: Sequence[float]) -> float:
        sim = _qaoa_state(n_qubits, cost_hamiltonian, p, params)
        return sum(coeff * sim.expectation(pauli) for coeff, pauli in cost_hamiltonian)

    history: list[float] = []
    callback: Callable[[Any], None] | None = None
    if record_history:
        def callback(xk: Any) -> None:
            history.append(float(cost(xk)))

    result = minimize(
        cost, init_params, method=optimizer,
        options={"maxiter": maxiter}, callback=callback,
    )
    metadata: dict[str, Any] = {"params": [float(x) for x in result.x]}
    if record_history:
        metadata["history"] = history
    return Result.from_value(float(result.fun), **metadata)
