"""Translator protocol: each gate or classical-control operation declares its
translation for the three library backends (qiskit / cirq / pennylane).

The in-house native backend is not a translator target: it is data-driven
(``engine.apply(name, ...)``) and executes classical control shot-by-shot.
"""

from __future__ import annotations

from typing import Any


class Translator:
    name: str = ""

    def to_qiskit(self, qc: Any, op: Any, cregs: dict[str, Any]) -> None:
        """Emit this operation onto a Qiskit QuantumCircuit in place.

        ``cregs`` maps a named classical register to its ClassicalRegister
        (multi-bit) or its qubit index (single-bit alias), maintained by the
        qiskit backend across ops.
        """
        raise NotImplementedError

    def to_cirq(
        self, cirq: Any, op: Any, qubits: list[Any], cregs: dict[str, Any]
    ) -> list[Any]:
        """Return the list of Cirq operations for this operation.

        ``cregs`` maps a named classical register to {bit_index: measurement key}.
        """
        raise NotImplementedError

    def to_pennylane(self, qml: Any, op: Any, cregs: dict[str, Any]) -> None:
        """Emit this operation inside a PennyLane qnode in place.

        ``cregs`` maps a named classical register to {bit_index: measured value}.
        """
        raise NotImplementedError
