"""Identity gate translator."""

from __future__ import annotations

from typing import Any

from .base import Translator


class IdentityTranslator(Translator):
    name = "i"

    def to_qiskit(self, qc: Any, op: Any, cregs: dict[str, int]) -> None:
        qc.id(op.qubits[0])

    def to_cirq(
        self, cirq: Any, op: Any, qubits: list[Any], cregs: dict[str, str]
    ) -> list[Any]:
        return [cirq.I(qubits[op.qubits[0]])]

    def to_pennylane(self, qml: Any, op: Any, cregs: dict[str, Any]) -> None:
        qml.Identity(wires=op.qubits[0])
