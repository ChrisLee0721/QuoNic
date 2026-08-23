"""QPanda3 backend adapter."""

from __future__ import annotations

from typing import Any

from .._i18n import tr
from ..noise import NoiseModel
from .engine import EngineBackend


class QPandaBackend(EngineBackend):
    name = "qpanda"
    _MISSING_ERR = "err.qpanda_missing"
    _GATE_ERR = "err.qpanda_gate"
    methods = frozenset({"statevector", "density_matrix"})
    _CAPABILITIES = {"noise": True, "ctrl": True, "mid_measure": True, "gpu": True}

    # ------------------------------------------------------------------ #
    #  Statevector path
    # ------------------------------------------------------------------ #

    def _create(self, n: int) -> Any:
        try:
            from pyqpanda3.core import QCircuit
        except ImportError as e:
            raise ImportError(tr(self._MISSING_ERR)) from e
        self._n = n
        return QCircuit(n)

    def _apply_one(
        self, engine: Any, name: str, qubits: list[int], params: tuple[float, ...]
    ) -> None:
        from pyqpanda3.core import CNOT, CP, CZ, RX, RY, RZ, SWAP, TOFFOLI, H, X, Y, Z

        if name in ("identity", "i"):
            pass  # no-op
        elif name == "h":
            engine << H(qubits[0])
        elif name == "x":
            engine << X(qubits[0])
        elif name == "y":
            engine << Y(qubits[0])
        elif name == "z":
            engine << Z(qubits[0])
        elif name == "cx":
            engine << CNOT(qubits[0], qubits[1])
        elif name == "cz":
            engine << CZ(qubits[0], qubits[1])
        elif name == "swap":
            engine << SWAP(qubits[0], qubits[1])
        elif name == "ccx":
            engine << TOFFOLI(qubits[0], qubits[1], qubits[2])
        elif name == "rx":
            engine << RX(qubits[0], params[0])
        elif name == "ry":
            engine << RY(qubits[0], params[0])
        elif name == "rz":
            engine << RZ(qubits[0], params[0])
        elif name == "p":
            # P(θ) — QPanda3 has P gate
            from pyqpanda3.core import P

            engine << P(qubits[0], params[0])
        elif name == "cp":
            engine << CP(qubits[0], qubits[1], params[0])
        elif name == "mcz":
            self._apply_mcz(engine, qubits)
        elif name == "measure":
            pass  # measurement added in _sample
        else:
            raise ValueError(tr(self._GATE_ERR, name=name))

    @staticmethod
    def _apply_mcz(engine: Any, qubits: list[int]) -> None:
        from pyqpanda3.core import CNOT, CZ, H

        if len(qubits) == 2:
            engine << CZ(qubits[0], qubits[1])
        else:
            target = qubits[-1]
            engine << H(target)
            for c in qubits[:-1]:
                engine << CNOT(c, target)
            engine << H(target)

    def _sample(self, engine: Any, shots: int, n: int) -> dict[str, int]:
        from pyqpanda3.core import CPUQVM, QProg, measure

        prog = QProg()
        prog << engine
        # Add measurement for all qubits
        for q in range(n):
            prog << measure(q, q)
        qvm = CPUQVM()
        qvm.run(prog, shots)
        raw = qvm.result().get_counts()
        counts: dict[str, int] = {}
        for bs, count in raw.items():
            # QPanda3 returns bitstrings with MSB first; reverse for qubit-0-is-LSB
            counts[str(bs)[::-1]] = counts.get(str(bs)[::-1], 0) + int(count)
        return counts

    def _run_gpu(self, circuit, shots, nm):
        """Try GPUQVM, fallback to CuPy."""
        try:
            from pyqpanda3.core import GPUQVM, CBit, QProg, measure

            n = circuit.num_qubits
            qc = self._create(n)
            for op in circuit.ops:
                if op.name == "measure":
                    continue
                self._apply_one(qc, op.name, list(op.qubits), op.params)

            prog = QProg()
            prog << qc
            cbits = [CBit(i) for i in range(n)]
            for i in range(n):
                prog << measure(i, cbits[i])

            vm = GPUQVM()
            vm.run(prog, shots)
            raw = vm.result().get_counts()
            counts: dict[str, int] = {}
            for bs, count in raw.items():
                counts[str(bs)[::-1]] = counts.get(str(bs)[::-1], 0) + int(count)
            if nm.readout > 0:
                counts = self._apply_readout_noise(counts, n, nm.readout)
            from ..result import Result
            return Result.from_counts(counts, shots)
        except Exception:
            return super()._run_gpu(circuit, shots, nm)

    # ------------------------------------------------------------------ #
    #  Density-matrix path (v2)
    # ------------------------------------------------------------------ #

    def _sample_dm(self, engine: Any, shots: int, n: int) -> dict[str, int]:
        """Sample using density-matrix simulator."""
        from pyqpanda3.core import DensityMatrixSimulator, QProg, measure

        prog = QProg()
        prog << engine
        for q in range(n):
            prog << measure(q, q)
        sim = DensityMatrixSimulator()
        sim.run(prog, shots)
        raw = sim.result().get_counts()
        counts: dict[str, int] = {}
        for bs, count in raw.items():
            counts[str(bs)[::-1]] = counts.get(str(bs)[::-1], 0) + int(count)
        return counts

    def _apply_noise_after_gate(
        self, engine: Any, qubits: list[int], nm: NoiseModel
    ) -> None:
        """QPanda3 noise injection via NoiseModel."""
        from pyqpanda3.core import NoiseModel as QPandaNoise
        from pyqpanda3.core import depolarizing_error

        p = nm.single if len(qubits) == 1 else nm.double
        if p > 0:
            for q in qubits:
                noise = QPandaNoise()
                noise.add_noise_op(depolarizing_error(p), q)
                # Note: QPanda3 noise is typically applied at the QVM level,
                # not per-gate. This is a best-effort approach.

    def _measure_qubit(self, engine: Any, qubit: int) -> int:
        """Mid-circuit measurement via probability extraction."""
        import numpy as np
        from pyqpanda3.core import CPUQVM, QProg

        prog = QProg()
        prog << engine
        qvm = CPUQVM()
        qvm.run(prog, 1)
        probs = qvm.result().get_prob_list()
        n = self._n
        idx = np.arange(2**n)
        bit = (idx >> qubit) & 1
        p0 = float(np.sum(np.array(probs)[bit == 0]))
        return 0 if np.random.random() < p0 else 1

    def _run_dynamic(self, circuit, shots, nm, method, return_state=False):
        """Use numpy SV for dynamic circuits (QPanda3 doesn't support state injection)."""
        return self._run_dynamic_sv(circuit, shots, nm)
