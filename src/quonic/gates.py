"""Built-in quantum gates.

Gate objects are the primary API (consistent with the Qiskit/Cirq style):
    from quonic.gates import H, X, CX
    qgate(H, 0)

qgate() accepts either a gate object or a gate name string (e.g. qgate("h", 0)).
Parameterized gates (Rx/Ry/Rz) are factory functions that return gate objects with parameters:
    from quonic.gates import Rx
    qgate(Rx(0.5), 0)

Custom gates with arbitrary unitary matrices:
    import numpy as np
    u3 = Gate("u3", matrix=np.array([[a, b], [c, d]]))
    qgate(u3, 0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ._i18n import tr

# Global registry for custom gates (name -> Gate with matrix)
_GATE_REGISTRY: dict[str, Gate] = {}


@dataclass(frozen=True)
class Gate:
    name: str
    num_qubits: int
    params: tuple[float, ...] = field(default_factory=tuple)
    matrix: Any | None = None  # 2^n × 2^n unitary matrix (numpy array)

    @classmethod
    def from_matrix(cls, name: str, matrix: Any, **kwargs) -> Gate:
        """Create a custom gate from a unitary matrix.

        The matrix must be 2^n × 2^n for some n. The number of qubits is
        inferred from the matrix dimension.

        Example:
            import numpy as np
            my_gate = Gate.from_matrix("u3", np.array([[a, b], [c, d]]))
            qgate(my_gate, 0)
        """
        import math

        dim = matrix.shape[0]
        n_qubits = int(math.log2(dim))
        if 2**n_qubits != dim:
            raise ValueError(f"Matrix dimension {dim} is not a power of 2")
        gate = cls(name=name, num_qubits=n_qubits, matrix=matrix, **kwargs)
        _GATE_REGISTRY[name] = gate
        return gate


H = Gate("h", 1)
X = Gate("x", 1)
Y = Gate("y", 1)
Z = Gate("z", 1)
I = Gate("i", 1)  # identity gate, standard symbol
CX = Gate("cx", 2)
CZ = Gate("cz", 2)
CCX = Gate("ccx", 3)
SWAP = Gate("swap", 2)
MEASURE = Gate("measure", 1)


def CP(theta: float) -> Gate:
    """Controlled-phase gate: applies phase e^{i*theta} to |11>."""
    return Gate("cp", 2, (_angle(theta),))


def _angle(theta: float) -> float:
    try:
        return float(theta)
    except (TypeError, ValueError):
        raise TypeError(
            tr("err.gate_angle", theta=theta, type=type(theta).__name__)
        ) from None


def Rx(theta: float) -> Gate:
    return Gate("rx", 1, (_angle(theta),))


def Ry(theta: float) -> Gate:
    return Gate("ry", 1, (_angle(theta),))


def Rz(theta: float) -> Gate:
    return Gate("rz", 1, (_angle(theta),))


_BY_NAME = {g.name: g for g in (H, X, Y, Z, I, CX, CZ, CCX, SWAP, MEASURE)}

# the allowed values of a gate name string. IDEs (Pylance) use this to autocomplete gate names inside qgate("...").
GateName = Literal["h", "x", "y", "z", "i", "cx", "cz", "ccx", "swap", "measure"]


_GATE_ALIASES = {
    "hadamard": "h",
    "cnot": "cx",
    "toffoli": "ccx",
    "fredkin": "swap",
    "pauli_x": "x",
    "pauli_y": "y",
    "pauli_z": "z",
    "identity": "i",
    "phase": "p",
    "controlled_z": "cz",
}


def resolve(gate: Gate | GateName) -> Gate:
    """Resolve a gate object or gate name string into a Gate object."""
    if isinstance(gate, Gate):
        return gate
    if isinstance(gate, str):
        name = gate.strip().lower()
        if name in _BY_NAME:
            return _BY_NAME[name]
        # Check aliases
        if name in _GATE_ALIASES:
            return _BY_NAME[_GATE_ALIASES[name]]
        # Fuzzy match
        import difflib
        all_names = list(_BY_NAME.keys()) + list(_GATE_ALIASES.keys())
        matches = difflib.get_close_matches(name, all_names, n=1, cutoff=0.4)
        if matches:
            target = matches[0]
            if target in _GATE_ALIASES:
                target = _GATE_ALIASES[target]
            raise ValueError(
                tr("err.unknown_gate", gate=gate, gates=", ".join(sorted(_BY_NAME)))
                + f" Did you mean '{target}'?"
            )
        raise ValueError(
            tr("err.unknown_gate", gate=gate, gates=", ".join(sorted(_BY_NAME)))
        )
    raise TypeError(tr("err.qgate_arg", type=type(gate).__name__))


__all__ = [
    "CCX",
    "CP",
    "CX",
    "CZ",
    "MEASURE",
    "SWAP",
    "Gate",
    "GateName",
    "H",
    "I",
    "Rx",
    "Ry",
    "Rz",
    "X",
    "Y",
    "Z",
    "get_gate_registry",
    "resolve",
]


def get_gate_registry() -> dict[str, Gate]:
    """Return the global custom gate registry."""
    return dict(_GATE_REGISTRY)
