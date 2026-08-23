"""Swap gate translator."""

from __future__ import annotations

from typing import Any

from .base import Translator


class SwapTranslator(Translator):
    name = "swap"

    def to_qiskit(self, qc: Any, op: Any, cregs: dict[str, int]) -> None:
        qc.swap(op.qubits[0], op.qubits[1])

    def to_cirq(
        self, cirq: Any, op: Any, qubits: list[Any], cregs: dict[str, str]
    ) -> list[Any]:
        return [cirq.SWAP(qubits[op.qubits[0]], qubits[op.qubits[1]])]

    def to_pennylane(self, qml: Any, op: Any, cregs: dict[str, Any]) -> None:
        qml.SWAP(wires=[op.qubits[0], op.qubits[1]])
