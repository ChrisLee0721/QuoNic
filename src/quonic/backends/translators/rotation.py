"""Parameterized rotation / phase gate translators (rx / ry / rz / cp / p)."""

from __future__ import annotations

import math
from typing import Any

from .base import Translator


class RxTranslator(Translator):
    name = "rx"

    def to_qiskit(self, qc: Any, op: Any, cregs: dict[str, int]) -> None:
        qc.rx(op.params[0], op.qubits[0])

    def to_cirq(
        self, cirq: Any, op: Any, qubits: list[Any], cregs: dict[str, str]
    ) -> list[Any]:
        return [cirq.rx(op.params[0])(qubits[op.qubits[0]])]

    def to_pennylane(self, qml: Any, op: Any, cregs: dict[str, Any]) -> None:
        qml.RX(op.params[0], wires=op.qubits[0])


class RyTranslator(Translator):
    name = "ry"

    def to_qiskit(self, qc: Any, op: Any, cregs: dict[str, int]) -> None:
        qc.ry(op.params[0], op.qubits[0])

    def to_cirq(
        self, cirq: Any, op: Any, qubits: list[Any], cregs: dict[str, str]
    ) -> list[Any]:
        return [cirq.ry(op.params[0])(qubits[op.qubits[0]])]

    def to_pennylane(self, qml: Any, op: Any, cregs: dict[str, Any]) -> None:
        qml.RY(op.params[0], wires=op.qubits[0])


class RzTranslator(Translator):
    name = "rz"

    def to_qiskit(self, qc: Any, op: Any, cregs: dict[str, int]) -> None:
        qc.rz(op.params[0], op.qubits[0])

    def to_cirq(
        self, cirq: Any, op: Any, qubits: list[Any], cregs: dict[str, str]
    ) -> list[Any]:
        return [cirq.rz(op.params[0])(qubits[op.qubits[0]])]

    def to_pennylane(self, qml: Any, op: Any, cregs: dict[str, Any]) -> None:
        qml.RZ(op.params[0], wires=op.qubits[0])


class CpTranslator(Translator):
    name = "cp"

    def to_qiskit(self, qc: Any, op: Any, cregs: dict[str, int]) -> None:
        qc.cp(op.params[0], op.qubits[0], op.qubits[1])

    def to_cirq(
        self, cirq: Any, op: Any, qubits: list[Any], cregs: dict[str, str]
    ) -> list[Any]:
        return [
            cirq.CZPowGate(exponent=op.params[0] / math.pi).on(
                qubits[op.qubits[0]], qubits[op.qubits[1]]
            )
        ]

    def to_pennylane(self, qml: Any, op: Any, cregs: dict[str, Any]) -> None:
        qml.ControlledPhaseShift(op.params[0], wires=[op.qubits[0], op.qubits[1]])


class PTranslator(Translator):
    name = "p"

    def to_qiskit(self, qc: Any, op: Any, cregs: dict[str, int]) -> None:
        qc.p(op.params[0], op.qubits[0])

    def to_cirq(
        self, cirq: Any, op: Any, qubits: list[Any], cregs: dict[str, str]
    ) -> list[Any]:
        return [
            cirq.ZPowGate(exponent=op.params[0] / math.pi).on(qubits[op.qubits[0]])
        ]

    def to_pennylane(self, qml: Any, op: Any, cregs: dict[str, Any]) -> None:
        qml.PhaseShift(op.params[0], wires=op.qubits[0])
