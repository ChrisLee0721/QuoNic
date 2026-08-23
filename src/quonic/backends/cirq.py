"""Cirq backend adapter."""

from __future__ import annotations

from .._i18n import tr
from ..ir import Circuit
from ..noise import NoiseModel, resolve_noise
from ..result import Result
from .base import Backend
from .translators import TRANSLATORS


def _translate_custom_gate_cirq(cirq, op, qubits):
    """Translate a custom gate to Cirq operations."""
    from ..gates import _GATE_REGISTRY

    if op.name in _GATE_REGISTRY and _GATE_REGISTRY[op.name].matrix is not None:
        import numpy as np

        matrix = np.asarray(_GATE_REGISTRY[op.name].matrix, dtype=complex)
        gate = cirq.MatrixGate(matrix)
        return [gate.on(*[qubits[q] for q in op.qubits])]
    raise ValueError(tr("err.cirq_gate", name=op.name))


class CirqBackend(Backend):
    name = "cirq"
    methods = frozenset({"statevector"})

    def run(
        self,
        circuit: Circuit,
        shots: int = 1024,
        noise: NoiseModel | float | None = None,
        method: str = "statevector",
    ) -> Result:
        if method == "gpu":
            raise NotImplementedError(tr("err.no_gpu", name=self.name))
        try:
            import cirq
        except ImportError as e:
            raise ImportError(tr("err.cirq_missing")) from e

        nm = resolve_noise(noise)
        n = circuit.num_qubits
        qubits = [cirq.LineQubit(i) for i in range(n)]
        ops = []
        cregs = {}
        for op in circuit.ops:
            if op.name in TRANSLATORS:
                ops.extend(TRANSLATORS[op.name].to_cirq(cirq, op, qubits, cregs))
            else:
                ops.extend(_translate_custom_gate_cirq(cirq, op, qubits))
            if nm.enabled and op.name != "measure":
                nq = len(op.qubits)
                if nq == 1 and nm.single > 0.0:
                    ops.append(cirq.depolarize(nm.single).on(qubits[op.qubits[0]]))
                elif nq == 2 and nm.double > 0.0:
                    ops.append(
                        cirq.depolarize(nm.double, n_qubits=2).on(
                            *[qubits[i] for i in op.qubits]
                        )
                    )

        for q in circuit.unmeasured_qubits():
            ops.append(cirq.measure(qubits[q], key=f"m{q}"))

        simulator = cirq.Simulator()
        result = simulator.run(cirq.Circuit(ops), repetitions=shots)

        counts = {}
        for r in range(shots):
            bits = [result.measurements[f"m{q}"][r][0] for q in range(n)]
            bitstring = "".join(str(int(b)) for b in reversed(bits))
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return Result.from_counts(counts, shots)
