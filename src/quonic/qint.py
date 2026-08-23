"""QInt — quantum integer register.

A QInt occupies n_bits consecutive qubits in the current circuit, supporting classical loading,
uniform superposition, and quantum addition (QFT addition), finally measured with qshow().

Example:
    from quonic import QInt, qshow

    x = QInt(3, value=5)   # |5> = |101>
    x.h()                  # uniform superposition over |0>..|7>
    x += 3                 # add 3 to each component (mod 8)
    qshow()                # measure and display
"""

from __future__ import annotations

import math

from typing_extensions import Self

from ._i18n import tr
from .gates import H, Rz, X
from .ir import Circuit
from .qft import add_iqft, add_qft
from .qgate import qgate
from .qif import controlled
from .stack import current_circuit


class QInt:
    """Quantum integer register.

    Parameters:
        n_bits: bit width.
        value: initial classical value (None means |0>, equivalent to value=0).
    """

    def __init__(self, n_bits: int, value: int | None = None) -> None:
        if not isinstance(n_bits, int) or n_bits < 1:
            raise ValueError(tr("err.qint_n_bits", n_bits=n_bits))
        self.n_bits: int = n_bits
        base = current_circuit().num_qubits
        current_circuit().allocate(base + n_bits)
        self.qubits: tuple[int, ...] = tuple(range(base, base + n_bits))
        if value is not None:
            self.load(value)

    def load(self, value: int) -> QInt:
        """Classical load: set the register to |value>."""
        value = int(value)
        if not 0 <= value < 2 ** self.n_bits:
            raise ValueError(
                tr("err.qint_value_range", n_bits=self.n_bits, max=2 ** self.n_bits, value=value)
            )
        for j in range(self.n_bits):
            if (value >> j) & 1:
                qgate(X, self.qubits[j])
        return self

    def h(self) -> QInt:
        """Apply a Hadamard to each bit, producing a uniform superposition of 2**n_bits basis states."""
        for q in self.qubits:
            qgate(H, q)
        return self

    def superpose(self) -> QInt:
        """Alias of h(): uniform superposition."""
        return self.h()

    def add(self, k: int) -> QInt:
        """Quantum addition: |a> -> |a + k mod 2**n_bits> (k is a classical constant).

        Implemented with QFT addition (Draper addition); k may be any integer (auto-reduced modulo).
        """
        k = int(k) % 2 ** self.n_bits
        add_qft(current_circuit(), self.qubits)
        for j in range(self.n_bits):
            qgate(Rz(2 * math.pi * k / 2 ** (j + 1)), self.qubits[j])
        add_iqft(current_circuit(), self.qubits)
        return self

    def __iadd__(self, k: int) -> Self:
        return self.add(k)

    def sub(self, k: int) -> QInt:
        """Quantum subtraction: |a> -> |a - k mod 2**n_bits> (equivalent to adding -k)."""
        return self.add(-int(k))

    def __isub__(self, k: int) -> Self:
        return self.sub(k)

    def __int__(self) -> int:
        raise TypeError(tr("err.qint_superposition"))

    def lt(self, k: int) -> int:
        """Comparison: returns a flag bit that is 1 when x < k (x is left unchanged)."""
        from .compare import qlt

        return qlt(self, k)

    def eq(self, k: int) -> int:
        """Comparison: returns a flag bit that is 1 when x == k (x is left unchanged)."""
        from .compare import qeq

        return qeq(self, k)

    def gt(self, k: int) -> int:
        """Comparison: returns a flag bit that is 1 when x > k (x is left unchanged)."""
        from .compare import qgt

        return qgt(self, k)

    def mul(self, k: int) -> QInt:
        """Multiplication: returns a new QInt whose value is |x * k mod 2**n_bits> (x is left unchanged)."""
        return mul(self, k)

    def __repr__(self) -> str:
        return f"QInt({self.n_bits} bits, qubits={list(self.qubits)})"


def _add_quantum(
    circuit: Circuit,
    a_qubits: tuple[int, ...],
    b_qubits: tuple[int, ...],
    shift: int = 0,
) -> None:
    """Quantum-quantum addition: |a>|b> -> |a>|b + (a << shift) mod 2**n> (Draper addition).

    Unlike _add_const (adding a classical constant), here each bit of a acts as a controlled phase condition,
    implemented with controlled(Rz) rotations. shift is used for "shift-and-add" multiplication.
    """
    n = len(b_qubits)
    add_qft(circuit, b_qubits)
    for j in range(n):
        for i in range(n):
            p = i + shift
            if p <= j:
                theta = 2 * math.pi / (2 ** (j - p + 1))
                controlled(Rz(theta), a_qubits[i], b_qubits[j])
    add_iqft(circuit, b_qubits)


def mul(x: QInt, k: int) -> QInt:
    """Quantum multiplication: returns a new QInt whose value is |x * k mod 2**n>, with x left unchanged.

    Implemented via "shift-and-add + quantum-quantum Draper addition": for each set bit b of k, accumulate
    x << b into a zero-initialized result register. The result is stored in a clean register rather than
    overwritten in place, so it holds for any k (including even k; in-place multiplication by an even k is not reversible).

    Example:
        a = QInt(2, value=1)     # |1>
        p = mul(a, 3)            # p = |3>
        qshow()                  # read a, p
    """
    n = x.n_bits
    k = int(k) % (2 ** n)
    result = QInt(n)
    for b in range(n):
        if (k >> b) & 1:
            _add_quantum(current_circuit(), x.qubits, result.qubits, shift=b)
    return result
