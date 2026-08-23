"""GPU-accelerated adjoint differentiation using CuPy.

Provides the same API as adjoint.py but uses CuPy for GPU acceleration.
Falls back to numpy if CuPy is not available.

Example::

    from quonic.ml.adjoint_gpu import adjoint_grad_gpu

    grad = adjoint_grad_gpu(circuit, "ZZ", n_qubits)
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _xp():
    """Return CuPy if available, otherwise numpy."""
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
    except (ImportError, RuntimeError, ValueError):
        pass
    return np


def adjoint_grad_gpu(
    circuit: Any,
    observable: str,
    n_qubits: int,
) -> np.ndarray:
    """Compute gradient using GPU-accelerated adjoint differentiation.

    Same algorithm as adjoint_grad_statevector but uses CuPy for GPU
    acceleration when available.

    Args:
        circuit: circuit with parameterized gates
        observable: Pauli observable
        n_qubits: number of qubits

    Returns:
        Gradient vector for each parameterized gate.
    """
    xp = _xp()

    # Build the circuit and find parameterized gates
    param_gates = []
    for i, op in enumerate(circuit.ops):
        if hasattr(op, 'params') and op.params:
            param_gates.append((i, op))

    if not param_gates:
        return np.array([])

    n_params = len(param_gates)
    grad = np.zeros(n_params)

    # Forward pass: store states on GPU
    dim = 2 ** n_qubits
    state = xp.zeros(dim, dtype=complex)
    state[0] = 1.0

    states = [state.copy()]
    for op in circuit.ops:
        if hasattr(op, 'name') and op.name != 'measure':
            state = _apply_gate_gpu(xp, state, op.name, list(op.qubits), op.params, n_qubits)
            states.append(state.copy())

    # Build observable matrix on GPU
    obs_matrix = _build_observable_gpu(xp, observable, n_qubits)

    # Backward pass: compute gradients
    for idx, (gate_idx, op) in enumerate(param_gates):
        psi_before = states[gate_idx]
        psi_after = states[gate_idx + 1]

        # Compute ∂U/∂θ for this gate
        dU = _gate_derivative_gpu(xp, op.name, op.params[0], list(op.qubits), n_qubits)

        # Gradient = 2·Re(⟨ψ_after| O · dU |ψ_before⟩)
        dU_psi = dU @ psi_before
        obs_dU_psi = obs_matrix @ dU_psi
        grad_val = 2 * xp.real(xp.conj(psi_after) @ obs_dU_psi)

        # Move to CPU if needed
        if hasattr(grad_val, 'get'):
            grad_val = grad_val.get()
        grad[idx] = float(grad_val)

    return grad


def _apply_gate_gpu(xp, state, name, qubits, params, n_qubits):
    """Apply a gate to the state using GPU."""
    2 ** n_qubits
    gate = _gate_matrix_gpu(xp, name, params)
    if gate is None:
        return state

    # Apply gate to specific qubits
    if len(qubits) == 1:
        return _apply_single_qubit_gpu(xp, state, gate, qubits[0], n_qubits)
    elif len(qubits) == 2:
        return _apply_two_qubit_gpu(xp, state, gate, qubits[0], qubits[1], n_qubits)
    return state


def _apply_single_qubit_gpu(xp, state, gate, qubit, n_qubits):
    """Apply single-qubit gate using GPU."""
    dim = 2 ** n_qubits
    new_state = xp.zeros_like(state)

    for i in range(dim):
        bit = (i >> qubit) & 1
        # Find the partner index (flip the qubit bit)
        j = i ^ (1 << qubit)
        if bit == 0:
            new_state[i] += gate[0, 0] * state[i] + gate[0, 1] * state[j]
            new_state[j] += gate[1, 0] * state[i] + gate[1, 1] * state[j]

    return new_state


def _apply_two_qubit_gpu(xp, state, gate, qubit1, qubit2, n_qubits):
    """Apply two-qubit gate using GPU."""
    dim = 2 ** n_qubits
    new_state = xp.zeros_like(state)

    for i in range(dim):
        (i >> qubit1) & 1
        (i >> qubit2) & 1
        # Find all 4 basis states for these 2 qubits
        idx = [
            i,
            i ^ (1 << qubit1),
            i ^ (1 << qubit2),
            i ^ (1 << qubit1) ^ (1 << qubit2),
        ]
        # Apply 4x4 gate matrix
        for k, ik in enumerate(idx):
            for m, im in enumerate(idx):
                new_state[ik] += gate[k, m] * state[im]

    return new_state


def _gate_matrix_gpu(xp, name, params):
    """Get gate matrix on GPU."""
    if name == "h":
        return xp.array([[1, 1], [1, -1]], dtype=complex) / xp.sqrt(2)
    elif name == "x":
        return xp.array([[0, 1], [1, 0]], dtype=complex)
    elif name == "y":
        return xp.array([[0, -1j], [1j, 0]], dtype=complex)
    elif name == "z":
        return xp.array([[1, 0], [0, -1]], dtype=complex)
    elif name == "rx":
        t = params[0]
        return xp.array([[xp.cos(t/2), -1j*xp.sin(t/2)], [-1j*xp.sin(t/2), xp.cos(t/2)]], dtype=complex)
    elif name == "ry":
        t = params[0]
        return xp.array([[xp.cos(t/2), -xp.sin(t/2)], [xp.sin(t/2), xp.cos(t/2)]], dtype=complex)
    elif name == "rz":
        t = params[0]
        return xp.array([[xp.exp(-1j*t/2), 0], [0, xp.exp(1j*t/2)]], dtype=complex)
    elif name == "p":
        t = params[0]
        return xp.array([[1, 0], [0, xp.exp(1j*t)]], dtype=complex)
    return None


def _gate_derivative_gpu(xp, name, theta, qubits, n_qubits):
    """Compute gate derivative on GPU."""
    dim = 2 ** n_qubits

    if name == "ry":
        d_ry = xp.array([
            [-xp.sin(theta/2)/2, -xp.cos(theta/2)/2],
            [xp.cos(theta/2)/2, -xp.sin(theta/2)/2]
        ], dtype=complex)
        return _embed_gate_gpu(xp, d_ry, qubits[0], n_qubits)

    elif name == "rx":
        d_rx = xp.array([
            [-xp.sin(theta/2)/2, -1j*xp.cos(theta/2)/2],
            [-1j*xp.cos(theta/2)/2, -xp.sin(theta/2)/2]
        ], dtype=complex)
        return _embed_gate_gpu(xp, d_rx, qubits[0], n_qubits)

    elif name == "rz":
        d_rz = xp.array([
            [-1j*xp.exp(-1j*theta/2)/2, 0],
            [0, 1j*xp.exp(1j*theta/2)/2]
        ], dtype=complex)
        return _embed_gate_gpu(xp, d_rz, qubits[0], n_qubits)

    elif name == "p":
        d_p = xp.array([
            [0, 0],
            [0, 1j*xp.exp(1j*theta)]
        ], dtype=complex)
        return _embed_gate_gpu(xp, d_p, qubits[0], n_qubits)

    return xp.zeros((dim, dim), dtype=complex)


def _embed_gate_gpu(xp, gate, qubit, n_qubits):
    """Embed 2x2 gate into full Hilbert space on GPU."""
    dim = 2 ** n_qubits
    full_gate = xp.eye(dim, dtype=complex)

    for i in range(dim):
        for j in range(dim):
            if (i >> qubit) & 1 == 0 and (j >> qubit) & 1 == 0:
                full_gate[i, j] = gate[0, 0]
            elif (i >> qubit) & 1 == 0 and (j >> qubit) & 1 == 1:
                full_gate[i, j] = gate[0, 1]
            elif (i >> qubit) & 1 == 1 and (j >> qubit) & 1 == 0:
                full_gate[i, j] = gate[1, 0]
            elif (i >> qubit) & 1 == 1 and (j >> qubit) & 1 == 1:
                full_gate[i, j] = gate[1, 1]

    return full_gate


def _build_observable_gpu(xp, observable, n_qubits):
    """Build observable matrix on GPU."""
    pauli_map = {
        "I": xp.eye(2, dtype=complex),
        "X": xp.array([[0, 1], [1, 0]], dtype=complex),
        "Y": xp.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": xp.array([[1, 0], [0, -1]], dtype=complex),
    }

    obs = observable
    while len(obs) < n_qubits:
        obs = "I" + obs

    result = xp.array([[1.0]], dtype=complex)
    for i in range(n_qubits):
        result = xp.kron(result, pauli_map[obs[i]])

    return result
