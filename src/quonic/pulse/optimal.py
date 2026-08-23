"""GRAPE (GRadient Ascent Pulse Engineering) for optimal quantum control.

Optimizes pulse waveforms to implement a target unitary with minimum error.

Example::

    from quonic.pulse import grape_optimize
    import numpy as np

    target = np.array([[0, 1], [1, 0]], dtype=complex)  # X gate
    result = grape_optimize(target, n_steps=50, maxiter=200)
    print(f"Final fidelity: {result.fidelity:.6f}")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GRAPEResult:
    """Result of GRAPE optimization.

    Args:
        pulse: optimized pulse waveform (n_steps complex amplitudes)
        fidelity: final gate fidelity (1 - error)
        n_iter: number of iterations used
        loss_history: fidelity at each iteration
    """

    pulse: np.ndarray
    fidelity: float
    n_iter: int
    loss_history: list


def grape_optimize(
    target: np.ndarray,
    n_steps: int = 50,
    maxiter: int = 200,
    dt: float = 1.0,
    max_amp: float = 1.0,
    lr: float = 0.1,
    seed: int = 42,
    h_drift: np.ndarray | None = None,
) -> GRAPEResult:
    """Optimize a pulse to implement a target unitary using GRAPE.

    The pulse Hamiltonian is H(t) = u_x(t) * X + u_y(t) * Y + H_drift,
    where u_x and u_y are the optimized control amplitudes.

    Args:
        target: target 2x2 unitary matrix
        n_steps: number of time steps in the pulse
        maxiter: maximum optimization iterations
        dt: time step duration
        max_amp: maximum pulse amplitude
        lr: learning rate
        seed: random seed
        h_drift: drift Hamiltonian (default: 0)

    Returns:
        GRAPEResult with optimized pulse and fidelity.
    """
    rng = np.random.RandomState(seed)

    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    H_drift = h_drift if h_drift is not None else np.zeros((2, 2), dtype=complex)

    # Initialize pulse: (n_steps, 2) for [u_x, u_y]
    pulse = rng.randn(n_steps, 2) * 0.01

    loss_history = []

    for iteration in range(maxiter):
        # Forward propagation: compute U = U_n * ... * U_1
        U = I2.copy()
        propagators = []
        for k in range(n_steps):
            H_k = H_drift + pulse[k, 0] * X + pulse[k, 1] * Y
            U_k = _expm_i(H_k * dt)
            propagators.append(U_k)
            U = U_k @ U

        # Fidelity: |Tr(U†_target @ U)|^2 / 4
        overlap = np.trace(target.conj().T @ U)
        fidelity = float(np.abs(overlap) ** 2 / 4)
        loss_history.append(fidelity)

        if fidelity > 1 - 1e-10:
            break

        # Gradient via finite differences (more robust than analytic)
        grad = np.zeros_like(pulse)
        eps = 1e-5
        for k in range(n_steps):
            for c in range(2):
                pulse[k, c] += eps
                U_plus = I2.copy()
                for j in range(n_steps):
                    H_j = H_drift + pulse[j, 0] * X + pulse[j, 1] * Y
                    U_plus = _expm_i(H_j * dt) @ U_plus
                f_plus = np.abs(np.trace(target.conj().T @ U_plus)) ** 2 / 4

                pulse[k, c] -= 2 * eps
                U_minus = I2.copy()
                for j in range(n_steps):
                    H_j = H_drift + pulse[j, 0] * X + pulse[j, 1] * Y
                    U_minus = _expm_i(H_j * dt) @ U_minus
                f_minus = np.abs(np.trace(target.conj().T @ U_minus)) ** 2 / 4

                pulse[k, c] += eps  # restore
                grad[k, c] = (f_plus - f_minus) / (2 * eps)

        # Gradient ascent (maximize fidelity)
        pulse += lr * grad

        # Clamp amplitude
        pulse = np.clip(pulse, -max_amp, max_amp)

    # Final fidelity
    U_final = I2.copy()
    for k in range(n_steps):
        H_k = H_drift + pulse[k, 0] * X + pulse[k, 1] * Y
        U_final = _expm_i(H_k * dt) @ U_final

    final_overlap = np.trace(target.conj().T @ U_final)
    final_fidelity = float(np.abs(final_overlap) ** 2 / 4)

    # Combine u_x + i*u_y into complex waveform
    waveform = pulse[:, 0] + 1j * pulse[:, 1]

    return GRAPEResult(
        pulse=waveform,
        fidelity=final_fidelity,
        n_iter=len(loss_history),
        loss_history=loss_history,
    )


def krotov_optimize(
    target: np.ndarray,
    n_steps: int = 50,
    maxiter: int = 200,
    dt: float = 1.0,
    max_amp: float = 1.0,
    lambda_a: float = 1.0,
    seed: int = 42,
    h_drift: np.ndarray | None = None,
) -> GRAPEResult:
    """Optimize a pulse using Krotov's method.

    Krotov's method updates pulse amplitudes iteratively using forward-backward
    propagation. It's more stable than GRAPE for large systems.

    Args:
        target: target 2x2 unitary matrix
        n_steps: number of time steps
        maxiter: maximum iterations
        dt: time step duration
        max_amp: maximum pulse amplitude
        lambda_a: update scaling factor
        seed: random seed
        h_drift: drift Hamiltonian

    Returns:
        GRAPEResult with optimized pulse and fidelity.
    """
    rng = np.random.RandomState(seed)

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    H_drift = h_drift if h_drift is not None else np.zeros((2, 2), dtype=complex)

    pulse = rng.randn(n_steps, 2) * 0.01
    loss_history = []

    for iteration in range(maxiter):
        # Forward propagation
        U_fwd = I2.copy()
        fwd_props = []
        for k in range(n_steps):
            H_k = H_drift + pulse[k, 0] * X + pulse[k, 1] * Y
            U_k = _expm_i(H_k * dt)
            fwd_props.append(U_k)
            U_fwd = U_k @ U_fwd

        overlap = np.trace(target.conj().T @ U_fwd)
        fidelity = float(np.abs(overlap) ** 2 / 4)
        loss_history.append(fidelity)

        if fidelity > 1 - 1e-10:
            break

        # Backward propagation: target† @ U_fwd gives the error
        # Krotov update: Δu_k ∝ Im(Tr(B_k† @ σ_k))
        # where B_k = -i * H_control, σ_k = forward_state @ backward_state†
        U_bwd = target.conj().T  # start from target†
        for k in range(n_steps - 1, -1, -1):
            H_k = H_drift + pulse[k, 0] * X + pulse[k, 1] * Y
            U_k = _expm_i(H_k * dt)
            # Compute sigma_k = fwd_props[k] @ U_bwd
            sigma = fwd_props[k] @ U_bwd
            # Update: Δu_x = lambda * Im(Tr(X @ sigma))
            delta_x = lambda_a * np.imag(np.trace(X @ sigma))
            delta_y = lambda_a * np.imag(np.trace(Y @ sigma))
            pulse[k, 0] += delta_x
            pulse[k, 1] += delta_y
            U_bwd = U_k.conj().T @ U_bwd

        pulse = np.clip(pulse, -max_amp, max_amp)

    # Final fidelity
    U_final = I2.copy()
    for k in range(n_steps):
        H_k = H_drift + pulse[k, 0] * X + pulse[k, 1] * Y
        U_final = _expm_i(H_k * dt) @ U_final

    final_fidelity = float(np.abs(np.trace(target.conj().T @ U_final)) ** 2 / 4)
    waveform = pulse[:, 0] + 1j * pulse[:, 1]

    return GRAPEResult(
        pulse=waveform,
        fidelity=final_fidelity,
        n_iter=len(loss_history),
        loss_history=loss_history,
    )


def _expm_i(H: np.ndarray) -> np.ndarray:
    """Compute exp(-iH) for a 2x2 Hermitian matrix using eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvecs @ np.diag(np.exp(-1j * eigvals)) @ eigvecs.conj().T
