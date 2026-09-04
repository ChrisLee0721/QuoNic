"""QuoNic quantum circuit abstraction.

Backend-agnostic circuit representation. The oracle builds circuits
using this API, then exports to qiskit/pyqpanda3/hardware for execution.

Usage::

    from quonic.circuit import QuoNicCircuit

    qc = QuoNicCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.ccx(0, 1, 2)
    qc.z(2)

    # Export to qiskit
    qiskit_qc = qc.to_qiskit()

    # Export to pyqpanda3
    pq_prog = qc.to_pyqpanda3()

    # Run on qiskit-aer
    counts = qc.run(backend="qiskit", shots=1000)

    # Run on pyqpanda3
    counts = qc.run(backend="pyqpanda3", shots=1000)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GateOp:
    """A single gate operation."""
    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()


class QuoNicCircuit:
    """Backend-agnostic quantum circuit.

    Gates are stored as a list of GateOp objects. The circuit can be
    exported to qiskit or pyqpanda3 for execution.

    Supported gates:
        h(i)          — Hadamard on qubit i
        x(i)          — Pauli-X on qubit i
        z(i)          — Pauli-Z on qubit i
        cx(c, t)      — CNOT: control c, target t
        ccx(c1, c2, t) — Toffoli: controls c1,c2, target t
        mcx(ctrls, t) — Multi-controlled X
        mcz(ctrls, t) — Multi-controlled Z (via H-MCX-H)
    """

    def __init__(self, n_qubits: int):
        self._n_qubits = n_qubits
        self._ops: list[GateOp] = []

    @property
    def num_qubits(self) -> int:
        return self._n_qubits

    @property
    def ops(self) -> list[GateOp]:
        return self._ops

    def depth(self) -> int:
        return len(self._ops)

    # ─── Gate operations ────────────────────────────────────────────────

    def h(self, qubit: int) -> QuoNicCircuit:
        self._ops.append(GateOp("h", (qubit,)))
        return self

    def x(self, qubit: int) -> QuoNicCircuit:
        self._ops.append(GateOp("x", (qubit,)))
        return self

    def z(self, qubit: int) -> QuoNicCircuit:
        self._ops.append(GateOp("z", (qubit,)))
        return self

    def cx(self, control: int, target: int) -> QuoNicCircuit:
        self._ops.append(GateOp("cx", (control, target)))
        return self

    def ccx(self, ctrl1: int, ctrl2: int, target: int) -> QuoNicCircuit:
        self._ops.append(GateOp("ccx", (ctrl1, ctrl2, target)))
        return self

    def mcx(self, controls: list[int], target: int) -> QuoNicCircuit:
        """Multi-controlled X gate."""
        self._ops.append(GateOp("mcx", (*controls, target)))
        return self

    def mcz(self, controls: list[int], target: int) -> QuoNicCircuit:
        """Multi-controlled Z gate (via H-MCX-H)."""
        self.h(target)
        self.mcx(controls, target)
        self.h(target)
        return self

    def mcz_decomposed(self, controls: list[int], target: int, ancillas: list[int]) -> QuoNicCircuit:
        """Decomposed MCZ using Toffoli cascade with ancilla qubits.

        For hardware with limited control qubits (e.g., max 5).
        Uses len(controls)-2 ancilla qubits and O(len(controls)) Toffoli gates.

        Args:
            controls: Control qubit indices.
            target: Target qubit index.
            ancillas: Ancilla qubit indices (need len(controls)-2).
        """
        n = len(controls)
        if n <= 1:
            if n == 1:
                self.cz(controls[0], target)
            else:
                self.z(target)
            return self
        if n == 2:
            self.h(target)
            self.ccx(controls[0], controls[1], target)
            self.h(target)
            return self

        # Fan-in: build cascade on ancillas
        # Need n-2 ancillas for n controls
        n_anc = n - 2
        self.ccx(controls[0], controls[1], ancillas[0])
        for k in range(1, n_anc):
            self.ccx(controls[k + 1], ancillas[k - 1], ancillas[k])

        # MCZ via last control + last ancilla → target
        # Must include controls[-1] which the cascade missed
        self.h(target)
        self.ccx(controls[-1], ancillas[n_anc - 1], target)
        self.h(target)

        # Fan-out: uncompute ancillas (reverse order)
        for k in range(n_anc - 1, 0, -1):
            self.ccx(controls[k + 1], ancillas[k - 1], ancillas[k])
        self.ccx(controls[0], controls[1], ancillas[0])

        return self

    def cz(self, control: int, target: int) -> QuoNicCircuit:
        """Controlled-Z gate."""
        self.h(target)
        self.cx(control, target)
        self.h(target)
        return self

    def compose(self, other: QuoNicCircuit) -> QuoNicCircuit:
        """Append another circuit's operations."""
        self._ops.extend(other._ops)
        return self

    # ─── Backend export ─────────────────────────────────────────────────

    def to_qiskit(self) -> Any:
        """Export to qiskit QuantumCircuit."""
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(self._n_qubits)
        for op in self._ops:
            if op.name == "h":
                qc.h(op.qubits[0])
            elif op.name == "x":
                qc.x(op.qubits[0])
            elif op.name == "z":
                qc.z(op.qubits[0])
            elif op.name == "cx":
                qc.cx(op.qubits[0], op.qubits[1])
            elif op.name == "ccx":
                qc.ccx(op.qubits[0], op.qubits[1], op.qubits[2])
            elif op.name == "mcx":
                controls = list(op.qubits[:-1])
                target = op.qubits[-1]
                qc.mcx(controls, target)
        return qc

    def to_pyqpanda3(self) -> Any:
        """Export to pyqpanda3 QProg."""
        import pyqpanda3 as pq
        qc = pq.core.QCircuit()
        for op in self._ops:
            if op.name == "h":
                qc << pq.core.H(pq.core.Qubit(op.qubits[0]))
            elif op.name == "x":
                qc << pq.core.X(pq.core.Qubit(op.qubits[0]))
            elif op.name == "z":
                qc << pq.core.Z(pq.core.Qubit(op.qubits[0]))
            elif op.name == "cx":
                qc << pq.core.CNOT(
                    pq.core.Qubit(op.qubits[0]),
                    pq.core.Qubit(op.qubits[1]),
                )
            elif op.name == "ccx":
                qc << pq.core.TOFFOLI(
                    pq.core.Qubit(op.qubits[0]),
                    pq.core.Qubit(op.qubits[1]),
                    pq.core.Qubit(op.qubits[2]),
                )
            elif op.name == "mcx":
                controls = [pq.core.Qubit(q) for q in op.qubits[:-1]]
                target = pq.core.Qubit(op.qubits[-1])
                x_circ = pq.core.QCircuit()
                x_circ << pq.core.X(target)
                mcx = x_circ.control(controls)
                qc << mcx
        return qc

    # ─── Execution ──────────────────────────────────────────────────────

    def run(self, backend: str = "qiskit", shots: int = 1000) -> dict[str, int]:
        """Run the circuit and return measurement counts.

        Args:
            backend: "qiskit" or "pyqpanda3".
            shots: Number of measurement shots.

        Returns:
            Dict mapping bitstring to count.
        """
        if backend == "qiskit":
            return self._run_qiskit(shots)
        elif backend == "pyqpanda3":
            return self._run_pyqpanda3(shots)
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

    def _run_qiskit(self, shots: int) -> dict[str, int]:
        from qiskit_aer import AerSimulator
        qc = self.to_qiskit()
        qc.measure_all()
        sim = AerSimulator()
        result = sim.run(qc, shots=shots).result()
        return result.get_counts()

    def _run_pyqpanda3(self, shots: int) -> dict[str, int]:
        import pyqpanda3 as pq
        prog = pq.core.QProg()
        prog << self.to_pyqpanda3()
        for i in range(self._n_qubits):
            prog << pq.core.measure(pq.core.Qubit(i), i)
        qvm = pq.core.CPUQVM()
        qvm.run(prog, shots)
        return qvm.result().get_counts()
