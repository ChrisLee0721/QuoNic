"""Hadamard gate translator."""

from __future__ import annotations

from typing import Any

from .base import Translator


class HadamardTranslator(Translator):
    name = "h"

    def to_qiskit(self, qc: Any, op: Any, cregs: dict[str, int]) -> None:
        qc.h(op.qubits[0])

    def to_cirq(
        self, cirq: Any, op: Any, qubits: list[Any], cregs: dict[str, str]
    ) -> list[Any]:
        return [cirq.H(qubits[op.qubits[0]])]

    def to_pennylane(self, qml: Any, op: Any, cregs: dict[str, Any]) -> None:
        qml.Hadamard(wires=op.qubits[0])
