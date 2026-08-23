"""Global circuit stack.

qgate() adds gates to the "current circuit". By default there is a single global circuit;
future custom gates / function scopes will push / pop new circuits, hence a stack rather than a single variable.
"""

from __future__ import annotations

from ._i18n import tr
from .ir import Circuit


class CircuitStack:
    def __init__(self) -> None:
        self._stack: list[Circuit] = [Circuit()]

    @property
    def current(self) -> Circuit:
        return self._stack[-1]

    def push(self) -> None:
        self._stack.append(Circuit())

    def pop(self) -> Circuit:
        if len(self._stack) == 1:
            raise RuntimeError(tr("err.stack_empty"))
        return self._stack.pop()

    def reset(self) -> None:
        self._stack = [Circuit()]


_default = CircuitStack()


def current_circuit() -> Circuit:
    return _default.current


def reset() -> None:
    _default.reset()


def push() -> None:
    """Push a new circuit scope (used by cwhile etc. to capture the loop body)."""
    _default.push()


def pop() -> Circuit:
    """Pop the current circuit scope and return that circuit."""
    return _default.pop()
