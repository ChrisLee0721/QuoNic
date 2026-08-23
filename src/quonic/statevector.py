"""StateVector / MixedState — wrappers around quantum state representations.

StateVector: pure state (numpy complex array, 2^n amplitudes).
MixedState: mixed state (density matrix, 2^n × 2^n).

Both support ``probabilities()`` and ``expectation()``.
Only StateVector supports ``amplitude()`` and ``fidelity()``.

Example::

    sv = qshow(return_state=True)
    print(sv.amplitude("011"))
    print(sv.expectation("ZZI"))
"""

from __future__ import annotations

from typing import Any


class StateVector:
    """Quantum state vector with convenience methods.

    Attributes:
        data: numpy complex array of shape (2^n,)
        n: number of qubits
    """

    def __init__(self, data: Any) -> None:
        import numpy as np

        self.data = np.asarray(data, dtype=complex)
        self.n = int(np.log2(len(self.data)))

    def amplitude(self, bitstring: str) -> complex:
        """Return the amplitude for a given bitstring (qubit 0 = rightmost)."""
        idx = int(bitstring, 2)
        return complex(self.data[idx])

    def probabilities(self) -> dict[str, float]:
        """Return the probability distribution over all bitstrings."""
        import numpy as np

        probs = np.abs(self.data) ** 2
        return {format(i, f"0{self.n}b"): float(p) for i, p in enumerate(probs)}

    def expectation(self, pauli: str) -> float:
        """Compute the expectation value <ψ|P|ψ> for a Pauli-string observable.

        The pauli string is read left-to-right as qubit (n-1) down to qubit 0.
        Each character is one of 'I', 'X', 'Y', 'Z'.
        Example: "ZZI" for n=3 means Z on qubit 2, Z on qubit 1, I on qubit 0.
        """
        import numpy as np

        if len(pauli) != self.n:
            raise ValueError(f"Pauli string length {len(pauli)} != n_qubits {self.n}")

        pauli_matrices = {
            "I": np.eye(2, dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        }

        # Build the full Pauli operator
        op = np.array([[1.0]], dtype=complex)
        # pauli[0] = qubit n-1 (MSB), pauli[-1] = qubit 0 (LSB)
        for ch in reversed(pauli):
            op = np.kron(op, pauli_matrices[ch])

        return float(np.real(self.data.conj() @ op @ self.data))

    def fidelity(self, other: StateVector) -> float:
        """Compute the state fidelity |<ψ|φ>|²."""
        import numpy as np

        return float(np.abs(np.vdot(self.data, other.data)) ** 2)

    def __repr__(self) -> str:
        probs = self.probabilities()
        nonzero = {k: f"{v:.4f}" for k, v in probs.items() if v > 1e-8}
        return f"StateVector(n={self.n}, {nonzero})"

    def __len__(self) -> int:
        return len(self.data)


class MixedState:
    """Density matrix wrapper for mixed quantum states.

    Returned when ``return_state=True`` with noise enabled.
    Supports ``probabilities()`` and ``expectation()`` but not ``amplitude()``
    (mixed states don't have well-defined amplitudes).

    Attributes:
        rho: numpy complex array of shape (2^n, 2^n)
        n: number of qubits
    """

    def __init__(self, rho: Any) -> None:
        import numpy as np

        self.rho = np.asarray(rho, dtype=complex)
        dim = self.rho.shape[0]
        self.n = int(np.log2(dim))

    def probabilities(self) -> dict[str, float]:
        """Return the probability distribution (diagonal of density matrix)."""
        import numpy as np

        diag = np.real(np.diag(self.rho))
        return {format(i, f"0{self.n}b"): float(max(0, p)) for i, p in enumerate(diag)}

    def expectation(self, pauli: str) -> float:
        """Compute Tr(ρ · P) for a Pauli-string observable."""
        import numpy as np

        if len(pauli) != self.n:
            raise ValueError(f"Pauli string length {len(pauli)} != n_qubits {self.n}")

        pauli_matrices = {
            "I": np.eye(2, dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        }

        op = np.array([[1.0]], dtype=complex)
        for ch in reversed(pauli):
            op = np.kron(op, pauli_matrices[ch])

        return float(np.real(np.trace(self.rho @ op)))

    def purity(self) -> float:
        """Compute Tr(ρ²).  Pure state → 1.0, maximally mixed → 1/2^n."""
        import numpy as np

        return float(np.real(np.trace(self.rho @ self.rho)))

    def fidelity(self, other: Any) -> float:
        """Compute fidelity with another state (StateVector or MixedState)."""
        import numpy as np

        if isinstance(other, StateVector):
            return float(np.real(other.data.conj() @ self.rho @ other.data))
        if isinstance(other, MixedState):
            # Fidelity = Tr(sqrt(sqrt(ρ) σ sqrt(ρ)))²
            sqrt_rho = _matrix_sqrt(self.rho)
            inner = sqrt_rho @ other.rho @ sqrt_rho
            sqrt_inner = _matrix_sqrt(inner)
            return float(np.real(np.trace(sqrt_inner)) ** 2)
        raise TypeError(f"Cannot compute fidelity with {type(other)}")

    def amplitude(self, bitstring: str) -> complex:
        raise NotImplementedError(
            "MixedState does not have well-defined amplitudes. "
            "Use probabilities() or expectation() instead."
        )

    def __repr__(self) -> str:
        probs = self.probabilities()
        nonzero = {k: f"{v:.4f}" for k, v in probs.items() if v > 1e-8}
        purity = self.purity()
        return f"MixedState(n={self.n}, purity={purity:.4f}, {nonzero})"


def _matrix_sqrt(m: Any) -> Any:
    """Compute the matrix square root via eigendecomposition."""
    import numpy as np

    eigvals, eigvecs = np.linalg.eigh(m)
    eigvals = np.maximum(eigvals, 0)  # clip negative eigenvalues
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T
