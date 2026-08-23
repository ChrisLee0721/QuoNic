"""IonQ backend adapter.

⚠️  UNTESTED: This backend has not been tested on real hardware.
   Code is provided as-is. Use at your own risk.

Submits circuits to IonQ cloud simulator or hardware via the IonQ API.

Prerequisites:
    pip install 'quonic[ionq]'
    # or: pip install ionq-cirq (or ionq-qiskit)

Usage:
    qshow(backend='ionq', device='ionq_simulator')
"""

from __future__ import annotations

from typing import Any, ClassVar

from .._i18n import tr
from ..ir import Circuit
from ..noise import NoiseModel
from ..result import Result
from .base import Backend


class IonQBackend(Backend):
    name = "ionq"
    methods = frozenset({"statevector"})
    _CAPABILITIES: ClassVar[dict[str, bool]] = {"noise": False, "ctrl": False, "mid_measure": False, "gpu": False}

    def __init__(self, device: str = "ionq_simulator") -> None:
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
            raise ValueError(tr("err.ionq_noise"))
        if return_state:
            raise NotImplementedError(tr("err.engine_no_sv", name=self.name))

        try:
            import ionq
        except ImportError as e:
            raise ImportError(tr("err.ionq_missing")) from e

        # Build IonQ circuit
        g = ionq.Gate()
        for op in circuit.ops:
            if op.name == "measure":
                continue
            _translate_gate(g, op)

        # Submit to IonQ
        if "simulator" in self.device:
            result = g.simulate(shots=shots)
        else:
            job = g.submit(device=self.device, shots=shots)
            result = job.get_counts()

        return Result.from_counts(result, shots)


def _translate_gate(g, op):
    """Translate a QuoNic gate to an IonQ gate."""
    name = op.name
    q = op.qubits

    if name == "h":
        g.h(q[0])
    elif name == "x":
        g.x(q[0])
    elif name == "y":
        g.y(q[0])
    elif name == "z":
        g.z(q[0])
    elif name == "cx":
        g.cnot(q[0], q[1])
    elif name == "rx":
        g.rx(q[0], op.params[0])
    elif name == "ry":
        g.ry(q[0], op.params[0])
    elif name == "rz":
        g.rz(q[0], op.params[0])
    elif name == "measure":
        pass
    else:
        raise ValueError(tr("err.ionq_gate", name=name))
