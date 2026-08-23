"""Stabilizer engine: Aaronson–Gottesman Clifford tableau (polynomial scaling).

Supports only the basic Clifford gate set {h, x, y, z, cx, cz}; encountering a
non-Clifford gate (arbitrary-angle rotation / ccx / cp, etc.) or mcz raises an
error, and the scheduler falls back to another method.

Conventions: qubit 0 is the least-significant bit. The tableau has 2n rows (the
first n rows are destabilizers, the last n rows are stabilizers), with columns =
x[0..n-1] | z[0..n-1] | phase (mod 4: 0=+1, 1=i, 2=-1, 3=-i).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .._i18n import tr


class StabilizerEngine:
    def __init__(self, num_qubits: int) -> None:
        self.n: int = num_qubits
        # rows 0..n-1 are destabilizers (X_i), rows n..2n-1 are stabilizers (Z_i)
        self.x: Any = np.zeros((2 * num_qubits, num_qubits), dtype=bool)
        self.z: Any = np.zeros((2 * num_qubits, num_qubits), dtype=bool)
        self.phase: Any = np.zeros(2 * num_qubits, dtype=int)
        for i in range(num_qubits):
            self.x[i, i] = True
            self.z[num_qubits + i, i] = True

    # ------------------------------------------------------------------
    # gate operations (act on all 2n rows)
    # ------------------------------------------------------------------
    def _h(self, q: int) -> None:
        mask = self.x[:, q] & self.z[:, q]
        self.x[:, q], self.z[:, q] = self.z[:, q].copy(), self.x[:, q].copy()
        self.phase = (self.phase + 2 * mask.astype(int)) % 4

    def _s(self, q: int) -> None:
        mask = self.x[:, q] & self.z[:, q]
        self.z[:, q] ^= self.x[:, q]
        self.phase = (self.phase + 2 * mask.astype(int)) % 4

    def _x(self, q: int) -> None:
        self.phase = (self.phase + 2 * self.z[:, q].astype(int)) % 4

    def _y(self, q: int) -> None:
        self.phase = (self.phase + 2 * (self.x[:, q] ^ self.z[:, q]).astype(int)) % 4

    def _z(self, q: int) -> None:
        self.phase = (self.phase + 2 * self.x[:, q].astype(int)) % 4

    def _cx(self, a: int, b: int) -> None:
        r = self.x[:, a] & self.z[:, b] & (self.x[:, b] ^ self.z[:, a] ^ True)
        self.phase = (self.phase + 2 * r.astype(int)) % 4
        self.x[:, b] ^= self.x[:, a]
        self.z[:, a] ^= self.z[:, b]

    def apply(
        self, name: str, qubits: Sequence[int], params: tuple[float, ...] = ()
    ) -> None:
        name = name.lower()
        if name == "measure":
            return
        if name == "h":
            self._h(qubits[0])
        elif name == "x":
            self._x(qubits[0])
        elif name == "y":
            self._y(qubits[0])
        elif name == "z":
            self._z(qubits[0])
        elif name == "cx":
            self._cx(qubits[0], qubits[1])
        elif name == "cz":
            self._h(qubits[1])
            self._cx(qubits[0], qubits[1])
            self._h(qubits[1])
        elif name == "swap":
            self._cx(qubits[0], qubits[1])
            self._cx(qubits[1], qubits[0])
            self._cx(qubits[0], qubits[1])
        else:
            raise ValueError(tr("err.stabilizer_gate", name=name))

    # ------------------------------------------------------------------
    # measurement + projection
    # ------------------------------------------------------------------
    def _rowsum(self, i: int, p: int) -> None:
        """row i = row i * row p (Pauli multiplication, with phase)."""
        xi, zi = self.x[i], self.z[i]
        xp, zp = self.x[p], self.z[p]
        per_q = (
            (xi & zi).astype(int)
            + (xp & zp).astype(int)
            + 2 * (zi & xp).astype(int)
            - ((xi ^ xp) & (zi ^ zp)).astype(int)
        )
        inc = int(per_q.sum() % 4)
        self.phase[i] = (self.phase[i] + self.phase[p] + inc) % 4
        self.x[i] = xi ^ xp
        self.z[i] = zi ^ zp

    @staticmethod
    def _bit(g: Any, c: int, n: int) -> bool:
        """The c-th bit of the symplectic vector g=(x, z): c<n is the x region, c>=n is the z region."""
        if c < n:
            return bool(g[0][c])
        return bool(g[1][c - n])

    @staticmethod
    def _mul(g: Any, h: Any) -> tuple[Any, Any, int]:
        """Pauli multiplication g·h, returns (x^, z^, phase mod 4)."""
        x1, z1, p1 = g
        x2, z2, p2 = h
        per_q = (
            (x1 & z1).astype(int)
            + (x2 & z2).astype(int)
            + 2 * (z1 & x2).astype(int)
            - ((x1 ^ x2) & (z1 ^ z2)).astype(int)
        )
        inc = int(per_q.sum() % 4)
        return x1 ^ x2, z1 ^ z2, (p1 + p2 + inc) % 4

    def _deterministic_outcome(self, q: int) -> int:
        """When no stabilizer row contains X_q, the measurement is deterministic; use Gaussian elimination to find the sign of Z_q."""
        n = self.n
        gens: list[Any] = [
            [self.x[i].copy(), self.z[i].copy(), int(self.phase[i])]
            for i in range(n, 2 * n)
        ]
        pivot = [-1] * (2 * n)
        for i in range(n):
            first = next((c for c in range(2 * n) if self._bit(gens[i], c, n)), None)
            if first is None:
                continue
            pivot[first] = i
            for j in range(n):
                if j != i and self._bit(gens[j], first, n):
                    gens[j] = self._mul(gens[j], gens[i])
        target: Any = [np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), 0]
        target[1][q] = True
        for c in range(2 * n):
            if self._bit(target, c, n):
                i = pivot[c]
                if i < 0:
                    raise RuntimeError(tr("err.stabilizer_measure"))
                target = self._mul(target, gens[i])
        return 0 if target[2] == 0 else 1

    def _measure(self, q: int) -> int:
        n = self.n
        p: int | None = None
        for i in range(n, 2 * n):
            if self.x[i, q]:
                p = i
                break
        if p is None:
            return self._deterministic_outcome(q)
        outcome = int(np.random.randint(2))
        for i in range(2 * n):
            if i != p and self.x[i, q]:
                self._rowsum(i, p)
        # destabilizer row (p-n) = old stabilizer row p; stabilizer row p = ±Z_q
        self.x[p - n] = self.x[p]
        self.z[p - n] = self.z[p]
        self.phase[p - n] = self.phase[p]
        self.x[p] = False
        self.z[p] = False
        self.z[p, q] = True
        self.phase[p] = 2 * outcome
        return outcome

    def sample(self, shots: int) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _ in range(shots):
            engine = self._copy()
            bits = [engine._measure(q) for q in range(self.n)]
            bs = "".join(str(b) for b in reversed(bits))
            counts[bs] = counts.get(bs, 0) + 1
        return counts

    def _copy(self) -> StabilizerEngine:
        e = StabilizerEngine(self.n)
        e.x = self.x.copy()
        e.z = self.z.copy()
        e.phase = self.phase.copy()
        return e
