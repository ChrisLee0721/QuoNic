"""Gradient computation for variational quantum circuits.

Provides parameter-shift and numerical gradient methods for computing
gradients of expectation values with respect to circuit parameters.

Example::

    from quonic.gradients import param_shift
    grad = param_shift(circuit, params, "ZZI")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from .ir import Circuit, GateOperation


def _bind_params(circuit: Circuit, param_map: dict[int, float]) -> Circuit:
    """Create a new circuit with parameters bound to values.

    Args:
        circuit: the parameterized circuit
        param_map: dict mapping parameter index to value

    Returns:
        A new Circuit with params replaced.
    """
    c = Circuit()
    c.allocate(circuit.num_qubits)
    for op in circuit.ops:
        if isinstance(op, GateOperation) and op.params:
            new_params = list(op.params)
            for i, p in enumerate(op.params):
                if isinstance(p, int) and p in param_map:
                    new_params[i] = param_map[p]
                elif i in param_map:
                    new_params[i] = param_map[i]
            c.add(GateOperation(op.name, op.qubits, tuple(new_params)))
        else:
            c.add(op)
    return c


def param_shift(
    circuit: Circuit,
    params: Sequence[float],
    observable: str | Callable,
    shift: float | None = None,
    backend: str = "native",
) -> Any:
    """Compute gradients via the parameter-shift rule.

    For each parameter θ_i, the gradient is:
        d<f>/dθ_i = [f(θ_i + π/2) - f(θ_i - π/2)] / 2

    This works for gates of the form exp(-iθG/2) where G has eigenvalues ±1.

    Args:
        circuit: parameterized circuit
        params: current parameter values
        observable: Pauli string (e.g. "ZZI") or callable(StateVector) -> float
        shift: shift amount (default π/2 for standard parameter-shift)
        backend: simulation backend name

    Returns:
        numpy array of gradients, one per parameter
    """
    import numpy as np

    if shift is None:
        shift = np.pi / 2

    from .backends import get_backend

    be = get_backend(backend)
    grad = np.zeros(len(params))

    for i in range(len(params)):
        # Build circuits with shifted parameters
        params_plus = list(params)
        params_plus[i] += shift
        params_minus = list(params)
        params_minus[i] -= shift

        circ_plus = _bind_params(circuit, {i: params_plus[i]})
        circ_minus = _bind_params(circuit, {i: params_minus[i]})

        sv_plus = be.run(circ_plus, return_state=True)
        sv_minus = be.run(circ_minus, return_state=True)

        if isinstance(observable, str):
            exp_plus = sv_plus.expectation(observable)
            exp_minus = sv_minus.expectation(observable)
        else:
            exp_plus = observable(sv_plus)
            exp_minus = observable(sv_minus)

        grad[i] = (exp_plus - exp_minus) / 2.0

    return grad


def numerical_gradient(
    circuit: Circuit,
    params: Sequence[float],
    observable: str | Callable,
    epsilon: float = 1e-5,
    backend: str = "native",
) -> Any:
    """Compute gradients via finite differences (numerical gradient).

    Less accurate than parameter-shift but works for any gate type.

    Args:
        circuit: parameterized circuit
        params: current parameter values
        observable: Pauli string or callable
        epsilon: finite difference step size
        backend: simulation backend name

    Returns:
        numpy array of gradients
    """
    import numpy as np

    from .backends import get_backend

    be = get_backend(backend)
    grad = np.zeros(len(params))

    for i in range(len(params)):
        params_plus = list(params)
        params_plus[i] += epsilon
        params_minus = list(params)
        params_minus[i] -= epsilon

        circ_plus = _bind_params(circuit, {i: params_plus[i]})
        circ_minus = _bind_params(circuit, {i: params_minus[i]})

        sv_plus = be.run(circ_plus, return_state=True)
        sv_minus = be.run(circ_minus, return_state=True)

        if isinstance(observable, str):
            exp_plus = sv_plus.expectation(observable)
            exp_minus = sv_minus.expectation(observable)
        else:
            exp_plus = observable(sv_plus)
            exp_minus = observable(sv_minus)

        grad[i] = (exp_plus - exp_minus) / (2 * epsilon)

    return grad
