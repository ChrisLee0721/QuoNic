"""Stabilizer formalism for quantum error correction.

Provides stabilizer code definition, syndrome computation, and logical operators.

Example::

    from quonic.qec import StabilizerCode
    code = StabilizerCode(["ZZII", "IIZZ", "XIII", "IIXI"])
    syndrome = code.syndrome_vector(state)
"""

from __future__ import annotations

import numpy as np

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_matrix(pauli_str: str) -> np.ndarray:
    """Build the full 2^n × 2^n matrix for a Pauli string like 'XIZI'."""
    mat = np.array([[1.0]], dtype=complex)
    for ch in pauli_str:
        mat = np.kron(mat, _PAULI[ch])
    return mat


class StabilizerCode:
    """Stabilizer code defined by a set of stabilizer generators.

    Args:
        stabilizers: list of Pauli strings (e.g. ["ZZII", "IIZZ"])
        logical_ops: optional dict of logical operators {"X": ["XIII"], "Z": ["ZZII"]}
    """

    def __init__(
        self,
        stabilizers: list[str],
        logical_ops: dict[str, list[str]] | None = None,
    ):
        self.stabilizers = stabilizers
        self.n_qubits = len(stabilizers[0]) if stabilizers else 0
        self.n_stabilizers = len(stabilizers)
        self.n_data = self.n_qubits - self.n_stabilizers
        self.distance = self.n_qubits - self.n_stabilizers  # conservative estimate
        self.logical_ops = logical_ops or {}

        # Pre-compute stabilizer matrices for fast syndrome extraction
        self._stab_matrices = [_pauli_matrix(s) for s in stabilizers]

    def syndrome_vector(self, state: np.ndarray) -> list[int]:
        """Compute syndrome bits for a given state vector.

        For each stabilizer generator g_i, the syndrome bit is:
            s_i = 0 if g_i|ψ⟩ = +|ψ⟩  (eigenvalue +1)
            s_i = 1 if g_i|ψ⟩ = -|ψ⟩  (eigenvalue -1)

        Args:
            state: state vector (2^n complex array)

        Returns:
            List of syndrome bits (0 or 1).
        """
        syndrome = []
        state_flat = state.ravel()
        for mat in self._stab_matrices:
            # Compute ⟨ψ|g_i|ψ⟩ — real part gives eigenvalue for stabilizer states
            val = np.real(np.conj(state_flat) @ mat @ state_flat)
            # Normalize: eigenvalue is ±1 for stabilizer states
            norm = np.real(np.conj(state_flat) @ state_flat)
            if norm > 0:
                val /= norm
            # +1 eigenvalue → syndrome 0, -1 eigenvalue → syndrome 1
            syndrome.append(0 if val > 0 else 1)
        return syndrome

    def is_valid(self, syndrome: list[int]) -> bool:
        """Check if syndrome indicates no error."""
        return all(s == 0 for s in syndrome)

    def detect_error(self, syndrome: list[int]) -> str | None:
        """Detect which Pauli error occurred based on syndrome.

        Returns the Pauli string of the most likely error, or None if no error.
        """
        if self.is_valid(syndrome):
            return None

        # Build syndrome lookup: for each single-qubit Pauli error, compute its syndrome
        n = self.n_qubits
        paulis = ["X", "Y", "Z"]

        best_match = None
        best_weight = n + 1

        for q in range(n):
            for p in paulis:
                error_str = "I" * q + p + "I" * (n - q - 1)
                error_syndrome = self._compute_error_syndrome(error_str)
                if error_syndrome == syndrome:
                    weight = 1  # single-qubit error
                    if weight < best_weight:
                        best_weight = weight
                        best_match = error_str

        return best_match

    def _compute_error_syndrome(self, error_str: str) -> list[int]:
        """Compute the syndrome that a given error would produce.

        For stabilizer g and error E: the syndrome bit is 0 if gE = Eg (commute),
        1 if gE = -Eg (anti-commute).
        """
        syndrome = []
        for stab in self.stabilizers:
            anticommute = False
            for a, b in zip(stab, error_str):
                if a == "I" or b == "I":
                    continue
                if a == b:
                    continue
                # Different non-identity Paulis anti-commute (X,Y,Z are all distinct)
                # But XZ = -ZX, XY = -YX, YZ = -ZY
                # So distinct Paulis on the same qubit → anti-commute
                anticommute = not anticommute
            syndrome.append(1 if anticommute else 0)
        return syndrome

    def logical_operator(self, name: str) -> np.ndarray | None:
        """Get the matrix for a logical operator.

        Args:
            name: operator name (e.g. "X", "Z")

        Returns:
            2^n × 2^n matrix, or None if not defined.
        """
        ops = self.logical_ops.get(name)
        if not ops:
            return None
        # Product of all logical operator strings
        mat = np.eye(2**self.n_qubits, dtype=complex)
        for op_str in ops:
            mat = mat @ _pauli_matrix(op_str)
        return mat

    def __repr__(self) -> str:
        return (
            f"StabilizerCode(n_qubits={self.n_qubits}, "
            f"n_stabilizers={self.n_stabilizers}, "
            f"distance={self.distance})"
        )
