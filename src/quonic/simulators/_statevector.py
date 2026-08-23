"""Statevector engine: 2^n complex amplitude vector, exact simulation (including non-Clifford gates).

Conventions: qubit 0 is the least-significant bit (rightmost in the bitstring).
Multi-qubit gates are implemented with the "diagonal phase + H" trick, avoiding
the index-order ambiguity of two-qubit matrices.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .._i18n import tr
from ._gates import _H, single


class StatevectorEngine:
    def __init__(self, num_qubits: int) -> None:
        self.n: int = num_qubits
        self.state: Any = np.zeros(2 ** num_qubits, dtype=complex)
        self.state[0] = 1.0  # |0...0>

    def _apply_single(self, u: Any, q: int) -> None:
        hi = 2 ** (self.n - q - 1)
        lo = 2 ** q
        s = self.state.reshape(hi, 2, lo)
        self.state = np.einsum("ij,ajk->aik", u, s).reshape(-1)

    def _apply_custom(self, matrix: np.ndarray, qubits: Sequence[int]) -> None:
        """Apply an arbitrary unitary matrix to the specified qubits."""
        dim = matrix.shape[0]
        n_qubits = int(np.log2(dim))
        other_qubits = [q for q in range(self.n) if q not in qubits]

        if not other_qubits:
            # All qubits are target qubits — simple matrix-vector multiply
            self.state = matrix @ self.state
            return

        perm = list(qubits) + other_qubits
        state_t = self.state.reshape([2] * self.n).transpose(perm)
        state_flat = state_t.reshape(dim, -1)
        result = matrix @ state_flat
        result = result.reshape([2] * n_qubits + [2 ** len(other_qubits)])
        inv_perm = [0] * self.n
        for i, q in enumerate(perm):
            inv_perm[q] = i
        result = result.transpose(inv_perm)
        self.state = result.reshape(-1)

    def _apply_phase(self, qubits: Sequence[int], angle: float) -> None:
        """Apply an e^{i·angle} phase (diagonal gate) to the basis states where all these qubits are |1>."""
        idx = np.arange(2 ** self.n)
        mask = np.ones(2 ** self.n, dtype=bool)
        for q in qubits:
            mask &= ((idx >> q) & 1).astype(bool)
        self.state[mask] *= np.exp(1j * angle)

    def _swap(self, a: int, b: int) -> None:
        if a == b:
            return
        idx = np.arange(2 ** self.n)
        ia = (idx >> a) & 1
        ib = (idx >> b) & 1
        mask = ~((1 << a) | (1 << b))
        perm = (idx & mask) | (ia << b) | (ib << a)
        self.state = self.state[perm]

    def _cswap(self, anc: int, a: int, b: int) -> None:
        """CSWAP: swap qubits a,b when ancilla anc is |1>."""
        if a == b:
            return
        idx = np.arange(2 ** self.n)
        anc_set = ((idx >> anc) & 1).astype(bool)
        ia = (idx >> a) & 1
        ib = (idx >> b) & 1
        mask = ~((1 << a) | (1 << b))
        # For states where ancilla=1, swap a and b
        new_idx = idx.copy()
        swapped = (idx & mask) | (ia << b) | (ib << a)
        new_idx[anc_set] = swapped[anc_set]
        self.state = self.state[new_idx]

    def apply(
        self, name: str, qubits: Sequence[int], params: tuple[float, ...] = ()
    ) -> None:
        from ..gates import _GATE_REGISTRY

        name = name.lower()
        if name == "measure":
            return
        # Check custom gate registry first
        if name in _GATE_REGISTRY and _GATE_REGISTRY[name].matrix is not None:
            self._apply_custom(_GATE_REGISTRY[name].matrix, qubits)
            return
        if name in ("i", "h", "x", "y", "z", "rx", "ry", "rz", "p"):
            self._apply_single(single(name, params), qubits[0])
        elif name == "cx":
            self._apply_single(_H, qubits[1])
            self._apply_phase(qubits, np.pi)
            self._apply_single(_H, qubits[1])
        elif name == "cz":
            self._apply_phase(qubits, np.pi)
        elif name == "cp":
            self._apply_phase(qubits, params[0])
        elif name == "ccx":
            self._apply_single(_H, qubits[2])
            self._apply_phase(qubits, np.pi)
            self._apply_single(_H, qubits[2])
        elif name == "cswap":
            # CSWAP(ancilla, i, j): swap qubits i,j when ancilla=|1>
            self._cswap(qubits[0], qubits[1], qubits[2])
        elif name == "swap":
            self._swap(qubits[0], qubits[1])
        elif name == "mcz":
            self._apply_phase(qubits, np.pi)
        else:
            raise ValueError(tr("err.statevector_gate", name=name))

    def sample(self, shots: int) -> dict[str, int]:
        probs = np.abs(self.state) ** 2
        probs = probs / probs.sum()
        idx = np.random.choice(2 ** self.n, size=shots, p=probs)
        counts: dict[str, int] = {}
        fmt = f"0{self.n}b"
        for i in idx:
            bs = format(int(i), fmt)
            counts[bs] = counts.get(bs, 0) + 1
        return counts

    def measure_qubit(self, q: int) -> int:
        """Mid-circuit measurement of qubit q: collapse by amplitude probabilities and return the 0/1 outcome."""
        idx = np.arange(2 ** self.n)
        bit = (idx >> q) & 1
        p0 = float(np.sum(np.abs(self.state[bit == 0]) ** 2))
        outcome = 0 if np.random.random() < p0 else 1
        self.state[bit != outcome] = 0.0
        norm = np.linalg.norm(self.state)
        if norm > 0.0:
            self.state /= norm
        return outcome
