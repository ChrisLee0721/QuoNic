"""Comparators: qlt / qeq / qgt — compare a quantum register with a classical constant, storing the result in a flag bit.

All three comparisons return a new ancilla bit (flag bit); the quantum register x is left unchanged before and after the comparison:

    from quonic import QInt, qlt, qeq, qgt, qshow

    x = QInt(3); x.h()          # |0..7> uniform superposition
    lt = qlt(x, 4)              # lt flag bit, 1 when x < 4
    qshow()                     # measure all bits, read the lt flag

Implementation notes:
- qeq is exact: whether x - k mod 2^n is zero (flip all X and detect all-zero with a multi-controlled Z).
- qlt uses an (n+1)-bit two's complement: the sign bit (the n-th bit) of x - k indicates x < k; the sign bit comes from a
  clean sign ancilla, and x and the ancilla are uncomputed/restored after the comparison.
- qgt = NOT qlt(k+1) (x > k ⟺ x >= k+1 ⟺ not x < k+1).

All based on QFT addition (the same Draper addition as QInt.add). numpy is only used indirectly at runtime
via add_qft / gate matrices, guaranteeing zero-cost `import quonic`.
"""

from __future__ import annotations

import math
from typing import Any

from ._i18n import tr
from .gates import CX, H, Rz, X
from .ir import Circuit, GateOperation
from .qft import add_iqft, add_qft
from .qgate import qgate
from .stack import current_circuit


def _alloc_ancilla() -> int:
    """Allocate a clean |0> ancilla bit and return its index."""
    circ = current_circuit()
    q = circ.num_qubits
    circ.allocate(q + 1)
    return q


def _add_const(circuit: Circuit, qubits: tuple[int, ...], k: int) -> None:
    """QFT addition: |a> -> |a + k mod 2**len(qubits)>; k may be negative."""
    n = len(qubits)
    k = int(k) % (2 ** n)
    add_qft(circuit, qubits)
    for j in range(n):
        qgate(Rz(2 * math.pi * k / 2 ** (j + 1)), qubits[j])
    add_iqft(circuit, qubits)


def _check_qint(x: Any) -> None:
    if not (hasattr(x, "n_bits") and hasattr(x, "qubits")):
        raise TypeError(tr("err.compare_qint", type=type(x).__name__))


def qeq(x: Any, k: int) -> int:
    """flag == 1 iff x == k (exact, x is left unchanged). Returns the flag bit index."""
    _check_qint(x)
    flag = _alloc_ancilla()
    _add_const(current_circuit(), x.qubits, -int(k))
    for q in x.qubits:
        qgate(X, q)
    qgate(H, flag)
    current_circuit().add(GateOperation("mcz", x.qubits + (flag,)))
    qgate(H, flag)
    for q in x.qubits:
        qgate(X, q)
    _add_const(current_circuit(), x.qubits, int(k))
    return flag


def qlt(x: Any, k: int) -> int:
    """flag == 1 iff x < k (x is left unchanged). Returns the flag bit index."""
    _check_qint(x)
    sign = _alloc_ancilla()
    flag = _alloc_ancilla()
    qubits = x.qubits + (sign,)
    _add_const(current_circuit(), qubits, -int(k))
    qgate(CX, sign, flag)
    _add_const(current_circuit(), qubits, int(k))
    return flag


def qgt(x: Any, k: int) -> int:
    """flag == 1 iff x > k (x is left unchanged). Returns the flag bit index."""
    flag = qlt(x, int(k) + 1)
    qgate(X, flag)
    return flag


__all__ = ["qeq", "qgt", "qlt"]
