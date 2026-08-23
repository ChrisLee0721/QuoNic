"""Controlled gate translators (cx / cz / ccx)."""

from __future__ import annotations

from typing import Any

from .base import Translator


class CXTranslator(Translator):
    name = "cx"

    def to_qiskit(self, qc: Any, op: Any, cregs: dict[str, int]) -> None:
        qc.cx(op.qubits[0], op.qubits[1])

    def to_cirq(
        self, cirq: Any, op: Any, qubits: list[Any], cregs: dict[str, str]
    ) -> list[Any]:
        return [cirq.CNOT(qubits[op.qubits[0]], qubits[op.qubits[1]])]

    def to_pennylane(self, qml: Any, op: Any, cregs: dict[str, Any]) -> None:
        qml.CNOT(wires=[op.qubits[0], op.qubits[1]])


class CZTranslator(Translator):
    name = "cz"

    def to_qiskit(self, qc: Any, op: Any, cregs: dict[str, int]) -> None:
        qc.cz(op.qubits[0], op.qubits[1])

    def to_cirq(
        self, cirq: Any, op: Any, qubits: list[Any], cregs: dict[str, str]
    ) -> list[Any]:
        return [cirq.CZ(qubits[op.qubits[0]], qubits[op.qubits[1]])]

    def to_pennylane(self, qml: Any, op: Any, cregs: dict[str, Any]) -> None:
        qml.CZ(wires=[op.qubits[0], op.qubits[1]])


class CCXTranslator(Translator):
    name = "ccx"

    def to_qiskit(self, qc: Any, op: Any, cregs: dict[str, int]) -> None:
        qc.ccx(op.qubits[0], op.qubits[1], op.qubits[2])

    def to_cirq(
        self, cirq: Any, op: Any, qubits: list[Any], cregs: dict[str, str]
    ) -> list[Any]:
        return [cirq.CCNOT(qubits[op.qubits[0]], qubits[op.qubits[1]], qubits[op.qubits[2]])]

    def to_pennylane(self, qml: Any, op: Any, cregs: dict[str, Any]) -> None:
        qml.Toffoli(wires=[op.qubits[0], op.qubits[1], op.qubits[2]])
