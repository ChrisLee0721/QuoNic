"""Azure Quantum backend adapter.

⚠️  UNTESTED: This backend has not been tested on real hardware.
   Code is provided as-is. Use at your own risk.

Submits circuits to Azure Quantum (IonQ, Quantinuum, Microsoft simulators).

Prerequisites:
    pip install 'quonic[azure]'
    # or: pip install azure-quantum

Usage:
    qshow(backend='azure', device='ionq_simulator')
"""

from __future__ import annotations

from typing import Any, ClassVar

from .._i18n import tr
from ..ir import Circuit
from ..noise import NoiseModel
from ..result import Result
from .base import Backend


class AzureBackend(Backend):
    name = "azure"
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
            raise ValueError(tr("err.azure_noise"))
        if return_state:
            raise NotImplementedError(tr("err.engine_no_sv", name=self.name))

        try:
            from azure.quantum import Workspace
        except ImportError as e:
            raise ImportError(tr("err.azure_missing")) from e

        # Build circuit (IonQ JSON format)
        circuit_dict = _build_ionq_circuit(circuit)

        # Submit to Azure Quantum
        workspace = Workspace()
        target = workspace.get_targets(name=self.device)
        job = target.submit(circuit_dict, shots=shots)
        result = job.get_results()

        counts = {}
        for bitstring, prob in result["histogram"].items():
            counts[bitstring] = int(prob * shots)

        return Result.from_counts(counts, shots)


def _build_ionq_circuit(circuit: Circuit) -> dict:
    """Build IonQ JSON circuit format."""
    gates = []
    for op in circuit.ops:
        if op.name == "measure":
            continue
        gate = {"target": op.qubits[0] if len(op.qubits) == 1 else list(op.qubits)}
        gate["gate"] = op.name.upper()
        if op.params:
            gate["rotation"] = op.params[0]
        gates.append(gate)

    return {
        "circuit": gates,
        "nqubits": circuit.num_qubits,
    }
