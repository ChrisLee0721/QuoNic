"""Rigetti backend adapter.

⚠️  UNTESTED: This backend has not been tested on real hardware.
   Code is provided as-is. Use at your own risk.

Submits circuits to Rigetti quantum hardware via pyQuil.

Prerequisites:
    pip install 'quonic[rigetti]'
    # or: pip install pyquil

Usage:
    qshow(backend='rigetti', device='Aspen-M-3')
"""

from __future__ import annotations

from typing import Any

from .._i18n import tr
from ..ir import Circuit
from ..noise import NoiseModel
from ..result import Result
from .base import Backend


class RigettiBackend(Backend):
    name = "rigetti"
    methods = frozenset({"statevector"})
    _CAPABILITIES = {"noise": False, "ctrl": False, "mid_measure": False, "gpu": False}

    def __init__(self, device: str = "Aspen-M-3") -> None:
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
            raise ValueError(tr("err.rigetti_noise"))
        if return_state:
            raise NotImplementedError(tr("err.engine_no_sv", name=self.name))

        try:
            from pyquil import Program, get_qc
        except ImportError as e:
            raise ImportError(tr("err.rigetti_missing")) from e

        # Build pyQuil program
        prog = Program()
        for op in circuit.ops:
            if op.name == "measure":
                continue
            _translate_gate(prog, op)

        # Submit to Rigetti
        qc = get_qc(self.device)
        result = qc.run(prog, shots=shots)
        counts = {}
        for row in result.readout_data.get("ro", []):
            bs = "".join(str(int(b)) for b in row)
            counts[bs] = counts.get(bs, 0) + 1

        return Result.from_counts(counts, shots)


def _translate_gate(prog, op):
    """Translate a QuoNic gate to a pyQuil gate."""
    from pyquil.gates import CNOT, CZ, RX, RY, RZ, SWAP
    from pyquil.gates import H as pH
    from pyquil.gates import X as pX
    from pyquil.gates import Y as pY
    from pyquil.gates import Z as pZ

    name = op.name
    q = op.qubits

    if name == "h":
        prog += pH(q[0])
    elif name == "x":
        prog += pX(q[0])
    elif name == "y":
        prog += pY(q[0])
    elif name == "z":
        prog += pZ(q[0])
    elif name == "cx":
        prog += CNOT(q[0], q[1])
    elif name == "cz":
        prog += CZ(q[0], q[1])
    elif name == "rx":
        prog += RX(op.params[0], q[0])
    elif name == "ry":
        prog += RY(op.params[0], q[0])
    elif name == "rz":
        prog += RZ(op.params[0], q[0])
    elif name == "swap":
        prog += SWAP(q[0], q[1])
    elif name == "measure":
        pass
    else:
        raise ValueError(tr("err.rigetti_gate", name=name))
