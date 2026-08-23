"""PennyLane backend adapter."""

from __future__ import annotations

from typing import Any, List, Optional, Union

from .._i18n import tr
from ..ir import Circuit
from ..noise import NoiseModel, resolve_noise
from ..result import Result
from .base import Backend
from .translators import TRANSLATORS


def _two_qubit_depolarizing_kraus(p: float) -> List[Any]:
    """The 16 Kraus operators of the two-qubit depolarizing channel (consistent with Qiskit's depolarizing_error(p, 2))."""
    import numpy as np

    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = (I2, X, Y, Z)
    kraus = [np.sqrt(1 - 15 * p / 16) * np.kron(I2, I2)]
    s = np.sqrt(p / 16)
    for a in paulis:
        for b in paulis:
            if a is I2 and b is I2:
                continue
            kraus.append(s * np.kron(a, b))
    return kraus


def _set_shots(qml: Any, qnode: Any, shots: int) -> Any:
    """Set shots across versions.

    In PennyLane 0.44+ set_shots is a transform (set_shots(qnode, shots=...));
    in earlier versions (0.36–0.42) it is a decorator (@set_shots(shots=...)). Older
    versions support Python 3.9/3.10 while newer ones require 3.11+, so we do runtime
    compatibility here.
    """
    try:
        return qml.set_shots(qnode, shots=shots)
    except TypeError:
        return qml.set_shots(shots=shots)(qnode)


def _translate_custom_gate_pennylane(qml: Any, op: Any) -> None:
    """Translate a custom gate to PennyLane QubitUnitary."""
    from ..gates import _GATE_REGISTRY

    if op.name in _GATE_REGISTRY and _GATE_REGISTRY[op.name].matrix is not None:
        import numpy as np

        matrix = np.asarray(_GATE_REGISTRY[op.name].matrix, dtype=complex)
        qml.QubitUnitary(matrix, wires=list(op.qubits))
    else:
        raise ValueError(
            tr("err.pennylane_gate", name=op.name)
        )


class PennyLaneBackend(Backend):
    name = "pennylane"
    methods = frozenset({"statevector"})

    def run(
        self,
        circuit: Circuit,
        shots: int = 1024,
        noise: Optional[Union[NoiseModel, float, int]] = None,
        method: str = "statevector",
    ) -> Result:
        if method == "gpu":
            raise NotImplementedError(tr("err.no_gpu", name=self.name))
        try:
            import pennylane as qml
        except ImportError as e:
            raise ImportError(tr("err.pennylane_missing")) from e

        nm = resolve_noise(noise)
        n = circuit.num_qubits
        device_name = "default.mixed" if nm.enabled else "default.qubit"
        dev = qml.device(device_name, wires=n)

        two_qubit_kraus = None
        if nm.enabled and nm.double > 0.0:
            two_qubit_kraus = _two_qubit_depolarizing_kraus(nm.double)

        @qml.qnode(dev)
        def qnode() -> Any:
            cregs = {}
            for op in circuit.ops:
                if op.name in TRANSLATORS:
                    TRANSLATORS[op.name].to_pennylane(qml, op, cregs)
                else:
                    _translate_custom_gate_pennylane(qml, op)
                if nm.enabled and op.name != "measure":
                    if len(op.qubits) == 1 and nm.single > 0.0:
                        qml.DepolarizingChannel(nm.single, wires=op.qubits[0])
                    elif len(op.qubits) == 2 and two_qubit_kraus is not None:
                        qml.QubitChannel(two_qubit_kraus, wires=list(op.qubits))
            return qml.counts()

        qnode = _set_shots(qml, qnode, shots)

        raw = qnode()
        # PennyLane's bitstring has wire0 at the most significant position;
        # reverse it to the Qiskit convention (qubit0 at the least significant position)
        counts = {}
        for bitstring, count in raw.items():
            key = str(bitstring)[::-1]
            counts[key] = counts.get(key, 0) + count
        return Result.from_counts(counts, shots)
