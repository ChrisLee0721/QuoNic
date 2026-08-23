"""Classical-if translator (measure-then-branch).

An int control measures that qubit first and branches on the result; a str control
reads a value already stored by a preceding ``cmeasure`` op (single-bit register,
then when == 1); a CRegCondition reads a named multi-bit register and branches on
``register == value``. All three forms are supported on qiskit, cirq and pennylane.
"""

from __future__ import annotations

from typing import Any

from ...ir import CRegCondition
from .base import Translator


class CifTranslator(Translator):
    name = "cif"

    def to_qiskit(self, qc: Any, op: Any, cregs: dict[str, Any]) -> None:
        from . import TRANSLATORS  # deferred import to avoid a cycle

        if isinstance(op.control, int):
            qc.measure(op.control, op.control)
            clbit = qc.clbits[op.control]
            with qc.if_test((clbit, 1)):
                TRANSLATORS[op.then_op.name].to_qiskit(qc, op.then_op, cregs)
            with qc.if_test((clbit, 0)):
                TRANSLATORS[op.else_op.name].to_qiskit(qc, op.else_op, cregs)
            return

        if isinstance(op.control, CRegCondition):
            cond = op.control
            if cond.width > 1:
                cr = cregs[cond.creg]  # a qiskit ClassicalRegister
                with qc.if_test((cr, cond.value)) as else_:
                    TRANSLATORS[op.then_op.name].to_qiskit(qc, op.then_op, cregs)
                with else_:
                    TRANSLATORS[op.else_op.name].to_qiskit(qc, op.else_op, cregs)
                return
            # width == 1 register with an explicit value: alias to a single clbit
            clbit = qc.clbits[cregs.get(cond.creg, 0)]
            with qc.if_test((clbit, cond.value)):
                TRANSLATORS[op.then_op.name].to_qiskit(qc, op.then_op, cregs)
            with qc.if_test((clbit, 1 - cond.value)):
                TRANSLATORS[op.else_op.name].to_qiskit(qc, op.else_op, cregs)
            return

        # str control (single-bit creg alias): then on == 1, else on == 0
        clbit = qc.clbits[cregs.get(op.control, 0)]
        with qc.if_test((clbit, 1)):
            TRANSLATORS[op.then_op.name].to_qiskit(qc, op.then_op, cregs)
        with qc.if_test((clbit, 0)):
            TRANSLATORS[op.else_op.name].to_qiskit(qc, op.else_op, cregs)

    def to_cirq(
        self, cirq: Any, op: Any, qubits: list[Any], cregs: dict[str, Any]
    ) -> list[Any]:
        from . import TRANSLATORS  # deferred import to avoid a cycle

        if isinstance(op.control, int):
            import sympy

            key = f"m{op.control}"
            ops = [cirq.measure(qubits[op.control], key=key)]
            then_ops = TRANSLATORS[op.then_op.name].to_cirq(cirq, op.then_op, qubits, cregs)
            else_ops = TRANSLATORS[op.else_op.name].to_cirq(cirq, op.else_op, qubits, cregs)
            for t in then_ops:
                ops.append(t.with_classical_controls(key))
            for e in else_ops:
                ops.append(e.with_classical_controls(sympy.Eq(sympy.Symbol(key), 0)))
            return ops

        import sympy

        if isinstance(op.control, CRegCondition):
            creg_name = op.control.creg
            width = op.control.width
            value = op.control.value
        else:  # str: single-bit register, then when == 1
            creg_name = op.control
            width = 1
            value = 1

        bits = cregs[creg_name]
        exprs = [sympy.Eq(sympy.Symbol(bits[i]), (value >> i) & 1) for i in range(width)]
        condition = exprs[0] if width == 1 else sympy.And(*exprs)

        then_ops = TRANSLATORS[op.then_op.name].to_cirq(cirq, op.then_op, qubits, cregs)
        else_ops = TRANSLATORS[op.else_op.name].to_cirq(cirq, op.else_op, qubits, cregs)
        ops = []
        for t in then_ops:
            ops.append(t.with_classical_controls(condition))
        for e in else_ops:
            ops.append(e.with_classical_controls(sympy.Not(condition)))
        return ops

    def to_pennylane(self, qml: Any, op: Any, cregs: dict[str, Any]) -> None:
        from . import TRANSLATORS  # deferred import to avoid a cycle

        if isinstance(op.control, int):
            m = qml.measure(wires=op.control)

            def then_fn() -> None:
                TRANSLATORS[op.then_op.name].to_pennylane(qml, op.then_op, cregs)

            def else_fn() -> None:
                TRANSLATORS[op.else_op.name].to_pennylane(qml, op.else_op, cregs)

            qml.cond(m == 1, then_fn, else_fn)()
            return

        if isinstance(op.control, CRegCondition):
            creg_name = op.control.creg
            width = op.control.width
            value = op.control.value
        else:  # str: single-bit register, then when == 1
            creg_name = op.control
            width = 1
            value = 1

        bits = cregs[creg_name]
        condition = None
        for i in range(width):
            cmp = bits[i] == ((value >> i) & 1)
            condition = cmp if condition is None else (condition & cmp)

        def then_fn() -> None:
            TRANSLATORS[op.then_op.name].to_pennylane(qml, op.then_op, cregs)

        def else_fn() -> None:
            TRANSLATORS[op.else_op.name].to_pennylane(qml, op.else_op, cregs)

        qml.cond(condition, then_fn, else_fn)()
