"""Xanadu (Strawberry Fields) backend adapter.

⚠️  UNTESTED: This backend has not been tested on real hardware.
   Code is provided as-is. Use at your own risk.

Submits circuits to Xanadu photonic quantum hardware.

Prerequisites:
    pip install 'quonic[xanadu]'
    # or: pip install strawberryfields

Usage:
    qshow(backend='xanadu', device='X8')
"""

from __future__ import annotations

from typing import Any

from .._i18n import tr
from ..ir import Circuit
from ..noise import NoiseModel
from ..result import Result
from .base import Backend


class XanaduBackend(Backend):
    name = "xanadu"
    methods = frozenset({"statevector"})
    _CAPABILITIES = {"noise": False, "ctrl": False, "mid_measure": False, "gpu": False}

    def __init__(self, device: str = "X8") -> None:
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
            raise ValueError(tr("err.xanadu_noise"))
        if return_state:
            raise NotImplementedError(tr("err.engine_no_sv", name=self.name))

        try:
            import strawberryfields as sf
            from strawberryfields import Program as SFProgram
            from strawberryfields import ops as SO
        except ImportError as e:
            raise ImportError(tr("err.xanadu_missing")) from e

        # Build Strawberry Fields program
        prog = SFProgram(circuit.num_qubits)
        with prog.context as q:
            for op in circuit.ops:
                if op.name == "measure":
                    continue
                _translate_gate(SO, q, op)

        # Submit to Xanadu
        eng = sf.RemoteEngine(self.device)
        result = eng.run(prog, shots=shots)
        counts = result.get_counts()

        return Result.from_counts(counts, shots)


def _translate_gate(SO, q, op):
    """Translate a QuoNic gate to a Strawberry Fields gate."""
    name = op.name
    qargs = op.qubits

    if name == "x":
        SO.X | q[qargs[0]]
    elif name == "z":
        SO.Z | q[qargs[0]]
    elif name == "s":
        SO.S | q[qargs[0]]
    elif name == "rx":
        SO.Rx(op.params[0]) | q[qargs[0]]
    elif name == "rz":
        SO.Rz(op.params[0]) | q[qargs[0]]
    elif name == "bs":
        SO.BSgate(op.params[0], op.params[1]) | (q[qargs[0]], q[qargs[1]])
    elif name == "measure":
        pass
    else:
        raise ValueError(tr("err.xanadu_gate", name=name))
