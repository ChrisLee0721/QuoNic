"""Adjoint differentiation for quantum circuits.

Computes exact gradients by backpropagating through the quantum circuit.
This is O(1) in circuit evaluations (1 forward + 1 backward) vs O(n) for
parameter-shift rule.

The key insight: for a parameterized gate U(θ) applied to state |ψ⟩:
- Output: |ψ'⟩ = U(θ)|ψ⟩
- Gradient: ∂⟨O⟩/∂θ = 2·Re(⟨φ|(∂U/∂θ)|ψ⟩)
  where |φ⟩ is the backward-propagated adjoint state.

Example::

    from quonic.ml.adjoint import adjoint_grad

    def loss_fn(params):
        circuit = ansatz.build(params)
        return expectation_loss(circuit, "ZZ")

    grad = adjoint_grad(loss_fn, params, ansatz)
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..ir import Circuit, GateOperation


def adjoint_grad(
    loss_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    ansatz: Any,
    observable: str = "Z",
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute gradient using adjoint differentiation.

    Uses parameter-shift rule which is exact for parameterized quantum gates.
    For true O(1) adjoint, use adjoint_grad_statevector() which requires
    simulator internals.

    Args:
        loss_fn: loss function(params) -> float (unused, for API compatibility)
        params: current parameters
        ansatz: ansatz builder with build(params) method
        observable: Pauli observable for measurement
        eps: numerical stability

    Returns:
        Gradient vector of shape (n_params,).
    """
    from .trainer import param_shift_grad
    return param_shift_grad(loss_fn, params)


def adjoint_grad_exact(
    circuit: Circuit,
    observable: str,
    n_qubits: int,
) -> np.ndarray:
    """Compute exact gradient using true O(1) adjoint differentiation.

    This implements backpropagation through the quantum circuit:
    1. Forward pass: apply gates, store intermediate states
    2. Backward pass: propagate gradients backward through each gate

    This is O(1) in circuit evaluations (just 1 forward + 1 backward).

    Args:
        circuit: the circuit to differentiate
        observable: Pauli observable string
        n_qubits: number of qubits

    Returns:
        Gradient vector for each parameterized gate.
    """
    from ..simulators._statevector import StatevectorEngine

    # Find parameterized gates
    param_gates = []
    for i, op in enumerate(circuit.ops):
        if isinstance(op, GateOperation) and op.params:
            param_gates.append((i, op))

    if not param_gates:
        return np.array([])

    n_params = len(param_gates)
    grad = np.zeros(n_params)

    # Forward pass: store states
    engine = StatevectorEngine(n_qubits)
    states = [engine.state.copy()]
    for op in circuit.ops:
        if isinstance(op, GateOperation):
            engine.apply(op.name, list(op.qubits), op.params)
            states.append(engine.state.copy())

    # Build observable matrix
    obs_matrix = _build_observable(observable, n_qubits)

    # Backward pass: compute gradients
    for idx, (gate_idx, op) in enumerate(param_gates):
        psi_before = states[gate_idx]
        psi_after = states[gate_idx + 1]

        # Compute ∂U/∂θ for this gate
        dU = _gate_derivative(op.name, op.params[0], list(op.qubits), n_qubits)

        # Gradient = 2·Re(⟨ψ_after| O · dU |ψ_before⟩)
        dU_psi = dU @ psi_before
        obs_dU_psi = obs_matrix @ dU_psi
        grad[idx] = 2 * np.real(np.conj(psi_after) @ obs_dU_psi)

    return grad


def _eval_circuit(circuit: Circuit, observable: str) -> float:
    """Evaluate expectation value of observable on circuit."""
    from .loss import expectation_loss
    return expectation_loss(circuit, observable)


def adjoint_grad_statevector(
    circuit: Circuit,
    observable: str,
    n_qubits: int,
) -> np.ndarray:
    """Compute gradient using true adjoint differentiation on statevector.

    This requires access to the statevector simulator internals.
    For each parameterized gate, we:
    1. Store the state before the gate
    2. Apply the gate
    3. Compute the gradient contribution using the stored state

    Args:
        circuit: the circuit to differentiate
        observable: Pauli observable
        n_qubits: number of qubits

    Returns:
        Gradient vector for each parameterized gate.
    """
    from ..simulators._statevector import StatevectorEngine

    # Build the circuit and find parameterized gates
    param_gates = []
    for i, op in enumerate(circuit.ops):
        if isinstance(op, GateOperation) and op.params:
            param_gates.append((i, op))

    if not param_gates:
        return np.array([])

    n_params = len(param_gates)
    grad = np.zeros(n_params)

    # Forward pass: store states
    engine = StatevectorEngine(n_qubits)
    states = [engine.state.copy()]
    for op in circuit.ops:
        if isinstance(op, GateOperation):
            engine.apply(op.name, list(op.qubits), op.params)
            states.append(engine.state.copy())

    # Build observable matrix
    obs_matrix = _build_observable(observable, n_qubits)

    # Backward pass: compute gradients
    # For each parameterized gate, compute:
    # ∂⟨O⟩/∂θ = 2·Re(⟨ψ_after| O · (∂U/∂θ) |ψ_before⟩)
    for idx, (gate_idx, op) in enumerate(param_gates):
        psi_before = states[gate_idx]
        psi_after = states[gate_idx + 1]

        # Compute ∂U/∂θ for this gate
        dU = _gate_derivative(op.name, op.params[0], list(op.qubits), n_qubits)

        # Gradient = 2·Re(⟨ψ_after| O · dU |ψ_before⟩)
        dU_psi = dU @ psi_before
        obs_dU_psi = obs_matrix @ dU_psi
        grad[idx] = 2 * np.real(np.conj(psi_after) @ obs_dU_psi)

    return grad


def _gate_derivative(name: str, theta: float, qubits: list[int], n_qubits: int) -> np.ndarray:
    """Compute the derivative of a parameterized gate w.r.t. its parameter.

    Returns the full 2^n × 2^n derivative matrix.
    """
    dim = 2 ** n_qubits

    # Derivative of single-qubit gates
    if name == "ry":
        # ∂Ry/∂θ = [[-sin(θ/2)/2, -cos(θ/2)/2], [cos(θ/2)/2, -sin(θ/2)/2]]
        d_ry = np.array([
            [-np.sin(theta / 2) / 2, -np.cos(theta / 2) / 2],
            [np.cos(theta / 2) / 2, -np.sin(theta / 2) / 2]
        ], dtype=complex)
        return _embed_single_qubit_gate(d_ry, qubits[0], n_qubits)

    elif name == "rx":
        # ∂Rx/∂θ = [[-sin(θ/2)/2, -i·cos(θ/2)/2], [-i·cos(θ/2)/2, -sin(θ/2)/2]]
        d_rx = np.array([
            [-np.sin(theta / 2) / 2, -1j * np.cos(theta / 2) / 2],
            [-1j * np.cos(theta / 2) / 2, -np.sin(theta / 2) / 2]
        ], dtype=complex)
        return _embed_single_qubit_gate(d_rx, qubits[0], n_qubits)

    elif name == "rz":
        # ∂Rz/∂θ = [[-i·e^{-iθ/2}/2, 0], [0, i·e^{iθ/2}/2]]
        d_rz = np.array([
            [-1j * np.exp(-1j * theta / 2) / 2, 0],
            [0, 1j * np.exp(1j * theta / 2) / 2]
        ], dtype=complex)
        return _embed_single_qubit_gate(d_rz, qubits[0], n_qubits)

    elif name == "p":
        # ∂P(θ)/∂θ = [[0, 0], [0, i·e^{iθ}]]
        d_p = np.array([
            [0, 0],
            [0, 1j * np.exp(1j * theta)]
        ], dtype=complex)
        return _embed_single_qubit_gate(d_p, qubits[0], n_qubits)

    # For non-parameterized gates, derivative is zero
    return np.zeros((dim, dim), dtype=complex)


def _embed_single_qubit_gate(gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Embed a 2×2 gate into the full 2^n Hilbert space."""
    dim = 2 ** n_qubits
    full_gate = np.eye(dim, dtype=complex)

    # Apply gate to the specified qubit
    for i in range(dim):
        for j in range(dim):
            # Check if i and j differ only at the qubit position
            if (i >> qubit) & 1 == 0 and (j >> qubit) & 1 == 0:
                # Both have 0 at qubit position
                full_gate[i, j] = gate[0, 0]
            elif (i >> qubit) & 1 == 0 and (j >> qubit) & 1 == 1:
                full_gate[i, j] = gate[0, 1]
            elif (i >> qubit) & 1 == 1 and (j >> qubit) & 1 == 0:
                full_gate[i, j] = gate[1, 0]
            elif (i >> qubit) & 1 == 1 and (j >> qubit) & 1 == 1:
                full_gate[i, j] = gate[1, 1]

    return full_gate


def _build_observable(observable: str, n_qubits: int) -> np.ndarray:
    """Build the full observable matrix from a Pauli string."""
    pauli_map = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    # Extend observable to match n_qubits
    obs = observable
    while len(obs) < n_qubits:
        obs = "I" + obs

    # Build tensor product
    result = np.array([[1.0]], dtype=complex)
    for i in range(n_qubits):
        result = np.kron(result, pauli_map[obs[i]])

    return result
