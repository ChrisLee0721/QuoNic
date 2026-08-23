"""IBM Quantum backend adapter.

⚠️  UNTESTED: This backend has not been tested on real hardware.
   Code is provided as-is. Use at your own risk.

Submits circuits to IBM Quantum hardware via qiskit-ibm-runtime.

Prerequisites:
    pip install 'quonic[ibm]'
    # or: pip install qiskit-ibm-runtime

Usage:
    qshow(backend='ibm', device='ibm_brisbane')
"""

from __future__ import annotations

from typing import Any, ClassVar

from .._i18n import tr
from ..ir import Circuit
from ..noise import NoiseModel
from ..result import Result
from .base import Backend
from .translators import TRANSLATORS


class IBMBackend(Backend):
    name = "ibm"
    methods = frozenset({"statevector"})
    _CAPABILITIES: ClassVar[dict[str, bool]] = {"noise": False, "ctrl": False, "mid_measure": False, "gpu": False}

    def __init__(self, device: str = "ibm_brisbane") -> None:
        self.device = device

    def run(
        self,
        circuit: Circuit,
        shots: int = 1024,
        noise: NoiseModel | float | None = None,
        method: str = "statevector",
        return_state: bool = False,
    ) -> Any:
        if noise is not None:
            raise ValueError(tr("err.qi_noise"))
        if return_state:
            raise NotImplementedError(tr("err.engine_no_sv", name=self.name))

        try:
            from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
            from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        except ImportError as e:
            raise ImportError(tr("err.ibm_missing")) from e

        # Build Qiskit circuit
        n = circuit.num_qubits
        qr = QuantumRegister(n, "q")
        cr = ClassicalRegister(n, "c")
        qc = QuantumCircuit(qr, cr)

        cregs = {}
        for op in circuit.ops:
            if op.name in TRANSLATORS:
                TRANSLATORS[op.name].to_qiskit(qc, op, cregs)

        # Auto-measure unmeasured qubits
        for q in circuit.unmeasured_qubits():
            qc.measure(q, q)

        # Submit to IBM Quantum
        service = QiskitRuntimeService()
        backend = service.backend(self.device)
        sampler = Sampler(backend)
        job = sampler.run(qc, shots=shots)
        result = job.result()

        # Convert to QuoNic Result
        counts = result.quasi_dists[0].binary_probabilities()
        counts = {k: int(v * shots) for k, v in counts.items()}
        return Result.from_counts(counts, shots)
