"""Classical measurement translator (measure a qubit into a named creg).

Single- and multi-bit cregs are supported on all three library backends. Qiskit
builds a real ClassicalRegister; cirq measures into a per-bit measurement key;
pennylane stores a per-bit MeasurementValue.
"""

from __future__ import annotations

from typing import Any

from .base import Translator


class CMeasureTranslator(Translator):
    name = "cmeasure"

    def to_qiskit(self, qc: Any, op: Any, cregs: dict[str, Any]) -> None:
        cr = cregs.get(op.creg)
        if cr is not None and not isinstance(cr, int):
            # multi-bit register: measure into the named ClassicalRegister bit
            qc.measure(op.qubit, cr[op.bit])
        else:
            # single-bit alias: measure into the qubit's own flat classical bit
            qc.measure(op.qubit, op.qubit)
            cregs[op.creg] = op.qubit

    def to_cirq(
        self, cirq: Any, op: Any, qubits: list[Any], cregs: dict[str, Any]
    ) -> list[Any]:
        key = f"m{op.qubit}"
        bits = cregs.setdefault(op.creg, {})
        bits[op.bit] = key
        return [cirq.measure(qubits[op.qubit], key=key)]

    def to_pennylane(self, qml: Any, op: Any, cregs: dict[str, Any]) -> None:
        mv = qml.measure(wires=op.qubit)
        bits = cregs.setdefault(op.creg, {})
        bits[op.bit] = mv
