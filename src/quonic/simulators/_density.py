"""Density matrix engine: 2^n × 2^n density matrix, supports depolarizing noise.

Conventions: qubit 0 is the least-significant bit. Noise is applied after every
logical gate (depolarizing channel).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

import numpy as np

from .._i18n import tr
from ..noise import NoiseModel, resolve_noise
from ._gates import _H, single

_I = np.eye(2, dtype=complex)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_PAULIS = (_I, _X, _Y, _Z)


class DensityMatrixEngine:
    def __init__(
        self,
        num_qubits: int,
        noise: NoiseModel | float | None = None,
    ) -> None:
        self.n: int = num_qubits
        self.noise: NoiseModel = resolve_noise(noise)
        self.rho: Any = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)
        self.rho[0, 0] = 1.0

    @staticmethod
    def _apply_single_to(rho: Any, u: Any, q: int, n: int) -> Any:
        hi = 2 ** (n - q - 1)
        lo = 2 ** q
        r = rho.reshape(hi, 2, lo, hi, 2, lo)
        uc = u.conj().T
        return np.einsum("AQBCRD,qQ,Rr->AqBCrD", r, u, uc).reshape(2 ** n, 2 ** n)

    def _apply_single(self, u: Any, q: int) -> None:
        self.rho = self._apply_single_to(self.rho, u, q, self.n)

    def _apply_custom(self, matrix: Any, qubits: Sequence[int]) -> None:
        """Apply an arbitrary unitary matrix to the density matrix."""
        dim = matrix.shape[0]
        n_qubits = int(np.log2(dim))
        other_qubits = [q for q in range(self.n) if q not in qubits]

        if not other_qubits:
            # All qubits are target qubits
            self.rho = matrix @ self.rho @ matrix.conj().T
            return

        # Reorder: target qubits first
        perm = list(qubits) + other_qubits
        inv_perm = [0] * self.n
        for i, q in enumerate(perm):
            inv_perm[q] = i

        # Apply U ⊗ I to rho: ρ' = U ρ U†
        dim_other = 2 ** len(other_qubits)
        # For multi-qubit: reshape, apply, reshape back
        rho_flat = self.rho.reshape(dim, dim_other, dim, dim_other)
        result = np.einsum("ij,jkln,ml->ikmn", matrix, rho_flat, matrix.conj())
        result = result.reshape([2] * n_qubits + [2] * len(other_qubits) + [2] * n_qubits + [2] * len(other_qubits))
        # Transpose back
        self.rho = result.transpose(inv_perm + [self.n + p for p in inv_perm]).reshape(-1, -1)

    def _apply_phase(self, qubits: Sequence[int], angle: float) -> None:
        idx = np.arange(2 ** self.n)
        mask = np.ones(2 ** self.n, dtype=bool)
        for q in qubits:
            mask &= ((idx >> q) & 1).astype(bool)
        phase = np.zeros(2 ** self.n)
        phase[mask] = angle
        self.rho *= np.exp(1j * (phase[:, None] - phase[None, :]))

    def _swap(self, a: int, b: int) -> None:
        if a == b:
            return
        idx = np.arange(2 ** self.n)
        ia = (idx >> a) & 1
        ib = (idx >> b) & 1
        mask = ~((1 << a) | (1 << b))
        perm = (idx & mask) | (ia << b) | (ib << a)
        self.rho = self.rho[perm][:, perm]

    def _cswap(self, anc: int, a: int, b: int) -> None:
        """CSWAP: swap qubits a,b when ancilla anc is |1>."""
        if a == b:
            return
        idx = np.arange(2 ** self.n)
        anc_set = ((idx >> anc) & 1).astype(bool)
        ia = (idx >> a) & 1
        ib = (idx >> b) & 1
        mask = ~((1 << a) | (1 << b))
        new_idx = idx.copy()
        swapped = (idx & mask) | (ia << b) | (ib << a)
        new_idx[anc_set] = swapped[anc_set]
        self.rho = self.rho[new_idx][:, new_idx]

    def _depolarize_single(self, q: int, p: float) -> None:
        rho = self.rho
        result = (1.0 - p) * rho
        for pauli in (_X, _Y, _Z):
            result += (p / 3.0) * self._apply_single_to(rho, pauli, q, self.n)
        self.rho = result

    def _depolarize_double(self, q0: int, q1: int, p: float) -> None:
        rho = self.rho
        result = (1.0 - p) * rho
        for pa in _PAULIS:
            for pb in _PAULIS:
                if pa is _I and pb is _I:
                    continue
                tmp = rho
                if pa is not _I:
                    tmp = self._apply_single_to(tmp, pa, q0, self.n)
                if pb is not _I:
                    tmp = self._apply_single_to(tmp, pb, q1, self.n)
                result += (p / 15.0) * tmp
        self.rho = result

    def _noise_after(self, qubits: Sequence[int]) -> None:
        if not self.noise.enabled:
            return
        nq = len(qubits)
        if nq == 1 and self.noise.single > 0.0:
            self._depolarize_single(qubits[0], self.noise.single)
        elif nq == 2 and self.noise.double > 0.0:
            self._depolarize_double(qubits[0], qubits[1], self.noise.double)

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
            self._cswap(qubits[0], qubits[1], qubits[2])
        elif name == "swap":
            self._swap(qubits[0], qubits[1])
        elif name == "mcz":
            self._apply_phase(qubits, np.pi)
        else:
            raise ValueError(tr("err.density_gate", name=name))
        self._noise_after(qubits)

    def sample(self, shots: int) -> dict[str, int]:
        probs = np.real(np.diag(self.rho))
        probs = np.clip(probs, 0.0, None)
        probs = probs / probs.sum()
        idx = np.random.choice(2 ** self.n, size=shots, p=probs)
        p = self.noise.readout
        if p > 0.0:
            # readout error: flip each measured bit independently with probability p
            flips = np.random.random((shots, self.n)) < p
            mask = (flips * (1 << np.arange(self.n))).sum(axis=1)
            idx = idx ^ mask.astype(int)
        counts: dict[str, int] = {}
        fmt = f"0{self.n}b"
        for i in idx:
            bs = format(int(i), fmt)
            counts[bs] = counts.get(bs, 0) + 1
        return counts

    def measure_qubit(self, q: int) -> int:
        """Mid-circuit measurement of qubit q: collapse the density matrix by
        diagonal probabilities and return the 0/1 outcome.

        Implemented with the projector P_outcome·ρ·P_outcome (keeping 2^n
        dimensions and zeroing the rows/columns where qubit q != outcome), then
        normalized by the trace.
        """
        idx = np.arange(2 ** self.n)
        bit = (idx >> q) & 1
        diag = np.real(np.diag(self.rho))
        p0 = float(np.sum(diag[bit == 0]))
        outcome = 0 if np.random.random() < p0 else 1
        keep = bit == outcome
        self.rho[~keep, :] = 0.0
        self.rho[:, ~keep] = 0.0
        tr = np.real(np.trace(self.rho))
        if tr > 0.0:
            self.rho /= tr
        return outcome

    def expectation(self, pauli_string: str) -> float:
        """Compute ⟨O⟩ = Tr(ρ·O) for the Pauli product O described by pauli_string.

        pauli_string[i] acts on qubit i (qubit 0 is the least-significant bit),
        matching StatevectorSimulator.expectation.
        """
        if len(pauli_string) != self.n:
            raise ValueError(
                tr("err.pauli_len", actual=len(pauli_string), expected=self.n)
            )
        _P = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
        o = functools.reduce(np.kron, [_P[p] for p in reversed(pauli_string)])
        return float(np.real(np.trace(self.rho @ o)))
