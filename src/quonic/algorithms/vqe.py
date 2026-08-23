"""VQE (Variational Quantum Eigensolver) template.

Given a Hamiltonian expressed as Pauli terms, variationally solve for its ground
state energy using a hardware-efficient ansatz.

Example: the transverse-field Ising model H = Z⊗Z + X⊗I + I⊗X (2 qubits)

    from quonic.algorithms import vqe

    hamiltonian = [(1.0, "ZZ"), (1.0, "XI"), (1.0, "IX")]
    result = vqe(hamiltonian, 2)
    print(result["energy"])   # close to the exact ground state energy
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from .._i18n import tr
from ..result import Result
from ..simulator import StatevectorSimulator


def _ansatz_state(n: int, params: Sequence[float]) -> StatevectorSimulator:
    # Hardware-efficient ansatz: Ry layer -> CX chain -> Ry layer, 2n parameters in total
    sim = StatevectorSimulator(n)
    for q in range(n):
        sim.apply("ry", (q,), (params[q],))
    for q in range(n - 1):
        sim.apply("cx", (q, q + 1))
    for q in range(n):
        sim.apply("ry", (q,), (params[n + q],))
    return sim


def vqe(
    hamiltonian: list[tuple[float, str]],
    n_qubits: int,
    init_params: Sequence[float] | None = None,
    optimizer: str = "COBYLA",
    maxiter: int = 300,
    record_history: bool = False,
) -> Result:
    """Variationally solve for the ground state energy of a Hamiltonian.

    Args:
        hamiltonian: List [(coefficient, Pauli string), ...]; each Pauli string
            has length = n_qubits.
        n_qubits: Number of qubits.
        init_params: Initial parameters (length 2 * n_qubits), all zeros by default.
        optimizer: Method name for scipy.optimize.minimize.
        maxiter: Maximum number of iterations.
        record_history: When True, record the energy of each step into
            metadata["history"] for plot_energy_convergence to plot the
            convergence curve (off by default to avoid extra simulation overhead).

    Returns: Result (kind="value"); result.value is the optimal energy and
    result.metadata["params"] is the optimal parameters.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as e:
        raise ImportError(tr("err.vqe_scipy")) from e

    if init_params is None:
        init_params = [0.0] * (2 * n_qubits)

    def energy(params: Sequence[float]) -> float:
        sim = _ansatz_state(n_qubits, params)
        return sum(coeff * sim.expectation(pauli) for coeff, pauli in hamiltonian)

    history: list[float] = []
    callback: Callable[[Any], None] | None = None
    if record_history:
        def callback(xk: Any) -> None:
            history.append(float(energy(xk)))

    result = minimize(
        energy,
        init_params,
        method=optimizer,
        options={"maxiter": maxiter},
        callback=callback,
    )
    metadata = {"params": [float(x) for x in result.x]}
    if record_history:
        metadata["history"] = history
    return Result.from_value(float(result.fun), **metadata)
