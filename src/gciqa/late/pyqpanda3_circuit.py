"""pyqpanda3 circuit wrapper that mimics qiskit's API.

Allows GroverOracle to build circuits for Origin Quantum hardware
without modifying the oracle arithmetic code.

Usage:
    qc = PyQpandaCircuit(n_qubits)
    qc.cx(0, 1)      # CNOT
    qc.x(0)           # X gate
    qc.ccx(0, 1, 2)   # Toffoli
    qc.h(0)           # Hadamard
    qc.z(0)           # Z gate

    prog = qc.to_qprog()
"""

from __future__ import annotations

from typing import Any


class PyQpandaQubit:
    """Wrapper around pyqpanda3 Qubit that supports integer indexing."""

    def __init__(self, index: int):
        import pyqpanda3 as pq
        self._qubit = pq.core.Qubit(index)
        self._index = index

    @property
    def index(self) -> int:
        return self._index

    def __repr__(self):
        return f"Qubit({self._index})"


class PyQpandaRegister:
    """Register of qubits, mimicking qiskit's QuantumRegister."""

    def __init__(self, size: int, name: str = ""):
        self._qubits = [PyQpandaQubit(i) for i in range(size)]
        self._name = name
        self._size = size

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._qubits[i] for i in range(*index.indices(self._size))]
        return self._qubits[index]

    def __len__(self):
        return self._size

    def __iter__(self):
        return iter(self._qubits)


class PyQpandaCircuit:
    """Circuit wrapper that mimics qiskit's QuantumCircuit API using pyqpanda3.

    Supports the gate operations used by GroverOracle:
    - cx(a, b): CNOT
    - x(a): Pauli-X
    - ccx(a, b, c): Toffoli (CCX)
    - h(a): Hadamard
    - z(a): Pauli-Z
    - compose(other): Append another circuit
    """

    def __init__(self, n_qubits: int, name: str = ""):
        import pyqpanda3 as pq
        self._pq = pq
        self._n_qubits = n_qubits
        self._operations: list[tuple] = []
        self._name = name

    @property
    def num_qubits(self) -> int:
        return self._n_qubits

    def _resolve_qubit(self, q) -> Any:
        """Convert PyQpandaQubit or int to pyqpanda3 Qubit."""
        if isinstance(q, PyQpandaQubit):
            return q._qubit
        elif isinstance(q, int):
            return self._pq.core.Qubit(q)
        return q

    def cx(self, control, target) -> None:
        """CNOT gate."""
        c = self._resolve_qubit(control)
        t = self._resolve_qubit(target)
        self._operations.append(("cx", c, t))

    def x(self, qubit) -> None:
        """Pauli-X gate."""
        q = self._resolve_qubit(qubit)
        self._operations.append(("x", q))

    def ccx(self, ctrl1, ctrl2, target) -> None:
        """Toffoli (CCX) gate."""
        c1 = self._resolve_qubit(ctrl1)
        c2 = self._resolve_qubit(ctrl2)
        t = self._resolve_qubit(target)
        self._operations.append(("ccx", c1, c2, t))

    def h(self, qubit) -> None:
        """Hadamard gate."""
        q = self._resolve_qubit(qubit)
        self._operations.append(("h", q))

    def z(self, qubit) -> None:
        """Pauli-Z gate."""
        q = self._resolve_qubit(qubit)
        self._operations.append(("z", q))

    def compose(self, other: PyQpandaCircuit) -> None:
        """Append another circuit's operations."""
        self._operations.extend(other._operations)

    def depth(self) -> int:
        """Estimate circuit depth (simplified)."""
        return len(self._operations)

    def to_qcircuit(self) -> Any:
        """Convert to pyqpanda3 QCircuit."""
        qc = self._pq.core.QCircuit()
        for op in self._operations:
            gate = op[0]
            if gate == "cx":
                qc << self._pq.core.CNOT(op[1], op[2])
            elif gate == "x":
                qc << self._pq.core.X(op[1])
            elif gate == "ccx":
                qc << self._pq.core.TOFFOLI(op[1], op[2], op[3])
            elif gate == "h":
                qc << self._pq.core.H(op[1])
            elif gate == "z":
                qc << self._pq.core.Z(op[1])
        return qc

    def to_qprog(self, measure: bool = True) -> Any:
        """Convert to pyqpanda3 QProg with optional measurement."""
        prog = self._pq.core.QProg()
        prog << self.to_qcircuit()

        if measure:
            for i in range(self._n_qubits):
                prog << self._pq.core.measure(self._pq.core.Qubit(i), i)

        return prog

    def run(self, shots: int = 1000, simulator: str = "cpu") -> dict[str, int]:
        """Build and run the circuit, return measurement counts."""
        prog = self.to_qprog(measure=True)

        if simulator == "mps":
            qvm = self._pq.core.MPSQVM()
        else:
            qvm = self._pq.core.CPUQVM()

        qvm.run(prog, shots)
        return qvm.result().get_counts()
