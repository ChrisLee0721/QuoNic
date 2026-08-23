"""@oracle decorator: compile a classical predicate into a Grover phase oracle.

     from quonic.algorithms import grover, oracle

     @oracle(3)
     def f(x):
         return x == 5            # mark |101> (qubit0 is the least significant bit)

     result = grover(f, 3)        # search for the unique state satisfying f
"""

from __future__ import annotations

from typing import Any, Callable

from .._i18n import tr
from ..ir import Circuit
from .grover import mark_state


def oracle(n_qubits: int) -> Callable[[Callable[[int], bool]], Any]:
    """Decorator: turn a classical predicate f(x) -> bool into a Grover phase oracle.

    f takes an integer x (0 <= x < 2**n_qubits) and returns True to mark that
    state. Every state satisfying f(x)=True receives a -1 phase.

    The returned decorator wraps f into an oracle(circuit) callback that can be
    passed directly to grover(), or to quantum_counting() (which reads its
    .marked attribute).

    Args:
        n_qubits: Number of qubits the predicate acts on.
    """
    if not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError(tr("err.oracle_n_qubits_positive", n_qubits=n_qubits))

    def decorator(f: Callable[[int], bool]) -> Any:
        marked: tuple[int, ...] = tuple(x for x in range(2 ** n_qubits) if f(x))

        def phase_oracle(circuit: Circuit) -> None:
            for x in marked:
                mark_state(format(x, f"0{n_qubits}b"))(circuit)

        phase_oracle.marked = marked
        phase_oracle.n_qubits = n_qubits
        phase_oracle.__name__ = getattr(f, "__name__", "oracle")
        return phase_oracle

    return decorator
