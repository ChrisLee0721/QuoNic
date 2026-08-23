"""Statevector simulator — uses numpy to compute expectation values exactly, for VQE / QAOA.

Convention: qubit 0 is the least-significant bit (the rightmost of the bitstring), consistent with the three sampling backends.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._i18n import tr

_I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)

_PAULI = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def _rotation(axis: str, theta: float) -> Any:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    if axis == "x":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if axis == "y":
        return np.array([[c, -s], [s, c]], dtype=complex)
    if axis == "z":
        return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)
    raise ValueError(tr("err.unknown_axis", axis=axis))


class StatevectorSimulator:
    def __init__(self, num_qubits: int) -> None:
        self.n: int = num_qubits
        self.state: Any = np.zeros(2 ** num_qubits, dtype=complex)
        self.state[0] = 1.0  # |0...0>

    def _apply_single(self, u: Any, q: int) -> None:
        a = 2 ** q
        k = 2 ** (self.n - q - 1)
        s = self.state.reshape(a, 2, k)
        self.state = np.einsum("ij,ajk->aik", u, s).reshape(-1)

    def _apply_phase(self, qubits: Sequence[int]) -> None:
        # apply a -1 phase to basis states where "all these qubits are |1>" (implementing multi-controlled Z)
        idx = np.arange(2 ** self.n)
        mask = np.ones(2 ** self.n, dtype=bool)
        for q in qubits:
            mask &= ((idx >> q) & 1).astype(bool)
        self.state = np.where(mask, -self.state, self.state)

    def apply(self, name: str, qubits: Sequence[int], params: Sequence[float] = ()) -> None:
        name = name.lower()
        if name == "h":
            self._apply_single(_H, qubits[0])
        elif name in ("x", "y", "z"):
            self._apply_single(_PAULI[name.upper()], qubits[0])
        elif name in ("rx", "ry", "rz"):
            self._apply_single(_rotation(name[1], params[0]), qubits[0])
        elif name == "cx":
            self._apply_single(_H, qubits[1])
            self._apply_phase((qubits[0], qubits[1]))
            self._apply_single(_H, qubits[1])
        elif name == "cz":
            self._apply_phase((qubits[0], qubits[1]))
        elif name == "ccx":
            self._apply_single(_H, qubits[2])
            self._apply_phase(tuple(qubits))
            self._apply_single(_H, qubits[2])
        elif name == "mcz":
            self._apply_phase(tuple(qubits))
        else:
            raise ValueError(tr("err.sim_unsupported_gate", name=name))

    def expectation(self, pauli_string: str) -> float:
        """Compute <ψ| O |ψ>, where O is the Pauli product described by pauli_string.

        pauli_string[i] acts on qubit i (e.g. "ZZ" means Z⊗Z).
        """
        if len(pauli_string) != self.n:
            raise ValueError(
                tr("err.pauli_len", actual=len(pauli_string), expected=self.n)
            )
        other = StatevectorSimulator(self.n)
        other.state = self.state.copy()
        for q, p in enumerate(pauli_string):
            if p != "I":
                other._apply_single(_PAULI[p], q)
        return float(np.real(np.vdot(self.state, other.state)))
