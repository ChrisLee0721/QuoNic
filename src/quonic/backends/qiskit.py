"""Qiskit backend adapter."""

from __future__ import annotations

from .._i18n import tr
from ..ir import Circuit, CRegCondition
from ..noise import NoiseModel, resolve_noise
from ..result import Result
from .base import Backend
from .translators import TRANSLATORS


def _translate_custom_gate_qiskit(qc, op) -> None:
    """Translate a custom gate (from _GATE_REGISTRY) to a Qiskit unitary gate."""
    from ..gates import _GATE_REGISTRY

    if op.name in _GATE_REGISTRY and _GATE_REGISTRY[op.name].matrix is not None:
        import numpy as np

        matrix = np.asarray(_GATE_REGISTRY[op.name].matrix, dtype=complex)
        qc.unitary(matrix, list(op.qubits))
    else:
        raise ValueError(
            tr("err.qiskit_gate", name=op.name)
        )


class QiskitBackend(Backend):
    name = "qiskit"
    methods = frozenset(
        {"statevector", "stabilizer", "matrix_product_state", "density_matrix", "gpu"}
    )
    _CAPABILITIES = {"noise": True, "ctrl": True, "mid_measure": False, "gpu": True}

    def run(
        self,
        circuit: Circuit,
        shots: int = 1024,
        noise: NoiseModel | float | None = None,
        method: str = "statevector",
    ) -> Result:
        try:
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator
        except ImportError as e:
            raise ImportError(tr("err.qiskit_missing")) from e

        nm = resolve_noise(noise)
        qc = QuantumCircuit(circuit.num_qubits, circuit.num_qubits)
        # A named single-bit classical bit is an alias for some qubit's measurement
        # result: map it to that bit's own classical bit, so get_counts outputs a flat
        # bitstring (with no named registers) consistent with the native backend.
        # Multi-bit registers (width > 1) instead get a real qiskit ClassicalRegister.
        cregs = {}
        from qiskit import ClassicalRegister

        widths = {}
        for op in circuit.ops:
            if op.name == "cmeasure" and op.bit > 0:
                w = op.bit + 1
                widths[op.creg] = max(widths.get(op.creg, 1), w)
            elif op.name == "cwhile" and op.width > 1:
                widths[op.creg] = max(widths.get(op.creg, 1), op.width)
            elif op.name == "cif" and isinstance(op.control, CRegCondition):
                if op.control.width > 1:
                    widths[op.control.creg] = max(
                        widths.get(op.control.creg, 1), op.control.width
                    )
        for name, width in widths.items():
            cr = ClassicalRegister(width, name=name)
            qc.add_register(cr)
            cregs[name] = cr

        for op in circuit.ops:
            if op.name in TRANSLATORS:
                TRANSLATORS[op.name].to_qiskit(qc, op, cregs)
            else:
                _translate_custom_gate_qiskit(qc, op)

        # Auto-complete: any qubit without an explicit measure is measured at the end
        for q in circuit.unmeasured_qubits():
            qc.measure(q, q)

        # Noise simulation requires the density-matrix method; stabilizer / MPS do not support general noise models
        if nm.enabled:
            method = "density_matrix"

        simulator = AerSimulator(method=method)
        run_kwargs = {}
        if nm.enabled:
            from qiskit_aer.noise import NoiseModel as QiskitNoiseModel
            from qiskit_aer.noise import depolarizing_error

            qnm = QiskitNoiseModel()
            single_gates = ["h", "x", "y", "z", "rx", "ry", "rz"]
            double_gates = ["cx", "cz", "swap"]
            if nm.single > 0.0:
                qnm.add_all_qubit_quantum_error(
                    depolarizing_error(nm.single, 1), single_gates
                )
            if nm.double > 0.0:
                qnm.add_all_qubit_quantum_error(
                    depolarizing_error(nm.double, 2), double_gates
                )
            if nm.readout > 0.0:
                from qiskit_aer.noise import ReadoutError

                p = nm.readout
                qnm.add_all_qubit_readout_error(
                    ReadoutError([[1.0 - p, p], [p, 1.0 - p]]),
                    list(range(circuit.num_qubits)),
                )
            run_kwargs["noise_model"] = qnm

        result = simulator.run(qc, shots=shots, **run_kwargs).result()
        counts = result.get_counts()
        return Result.from_counts(counts, shots)
