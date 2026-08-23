"""QuEra backend adapter.

⚠️  UNTESTED: This backend has not been tested on real hardware.
   Code is provided as-is. Use at your own risk.

Submits circuits to QuEra neutral-atom quantum hardware.

Prerequisites:
    pip install 'quonic[quera]'
    # or: pip install qurry (QuEra's Python SDK)

Usage:
    qshow(backend='quera', device='Aquila')
"""

from __future__ import annotations

from typing import Any

from .._i18n import tr
from ..ir import Circuit
from ..noise import NoiseModel
from ..result import Result
from .base import Backend


class QuEraBackend(Backend):
    name = "quera"
    methods = frozenset({"statevector"})
    _CAPABILITIES = {"noise": False, "ctrl": False, "mid_measure": False, "gpu": False}

    def __init__(self, device: str = "Aquila") -> None:
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
            raise ValueError(tr("err.quera_noise"))
        if return_state:
            raise NotImplementedError(tr("err.engine_no_sv", name=self.name))

        try:
            from qurry import QuantumCircuit as QC
            from qurry import QuEraDevice
        except ImportError as e:
            raise ImportError(tr("err.quera_missing")) from e

        # Build QuEra circuit
        qc = QC(circuit.num_qubits)
        for op in circuit.ops:
            if op.name == "measure":
                continue
            _translate_gate(qc, op)

        # Submit to QuEra
        device = QuEraDevice(self.device)
        job = device.run(qc, shots=shots)
        result = job.get_counts()

        return Result.from_counts(result, shots)


def _translate_gate(qc, op):
    """Translate a QuoNic gate to a QuEra gate."""
    name = op.name
    q = op.qubits

    if name == "h":
        qc.h(q[0])
    elif name == "x":
        qc.x(q[0])
    elif name == "y":
        qc.y(q[0])
    elif name == "z":
        qc.z(q[0])
    elif name == "rx":
        qc.rx(q[0], op.params[0])
    elif name == "ry":
        qc.ry(q[0], op.params[0])
    elif name == "rz":
        qc.rz(q[0], op.params[0])
    elif name == "measure":
        pass
    else:
        raise ValueError(tr("err.quera_gate", name=name))
