"""qgate — add a quantum gate to the current circuit."""

from __future__ import annotations

from ._i18n import tr
from .gates import Gate, GateName, resolve
from .ir import GateOperation
from .stack import current_circuit


def qgate(gate: Gate | GateName, *qubits: int) -> GateOperation:
    g = resolve(gate)
    qubits = tuple(int(q) for q in qubits)
    if len(qubits) != g.num_qubits:
        raise ValueError(
            tr("err.qgate_arity", name=g.name, expected=g.num_qubits, actual=len(qubits), qubits=qubits)
        )
    op = GateOperation(name=g.name, qubits=qubits, params=g.params)
    current_circuit().add(op)
    return op
