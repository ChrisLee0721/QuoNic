"""Multi-controlled-Z gate translator."""

from __future__ import annotations

import math
from typing import Any

from .base import Translator


class MczTranslator(Translator):
    name = "mcz"

    def to_qiskit(self, qc: Any, op: Any, cregs: dict[str, int]) -> None:
        q = op.qubits
        qc.mcp(math.pi, list(q[:-1]), q[-1])

    def to_cirq(
        self, cirq: Any, op: Any, qubits: list[Any], cregs: dict[str, str]
    ) -> list[Any]:
        q = op.qubits
        return [
            cirq.ControlledGate(cirq.Z, num_controls=len(q) - 1).on(
                *(qubits[i] for i in q)
            )
        ]

    def to_pennylane(self, qml: Any, op: Any, cregs: dict[str, Any]) -> None:
        target = op.qubits[-1]
        qml.Hadamard(wires=target)
        qml.MultiControlledX(wires=list(op.qubits))
        qml.Hadamard(wires=target)
