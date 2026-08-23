"""Qulacs backend adapter."""

from __future__ import annotations

from typing import Any, ClassVar

from .._i18n import tr
from .engine import EngineBackend


class QulacsBackend(EngineBackend):
    name = "qulacs"
    _MISSING_ERR = "err.qulacs_missing"
    _GATE_ERR = "err.qulacs_gate"
    methods = frozenset({"statevector", "density_matrix"})
    _CAPABILITIES: ClassVar[dict[str, bool]] = {"noise": True, "ctrl": True, "mid_measure": True, "gpu": True}

    # ------------------------------------------------------------------ #
    #  Statevector path (v1)
    # ------------------------------------------------------------------ #

    def _create(self, n: int) -> Any:
        try:
            import qulacs
        except ImportError as e:
            raise ImportError(tr(self._MISSING_ERR)) from e
        return qulacs.QuantumCircuit(n)

    def _apply_one(
        self, engine: Any, name: str, qubits: list[int], params: tuple[float, ...]
    ) -> None:
        from qulacs import gate

        # Check custom gate registry first
        from ..gates import _GATE_REGISTRY
        if name in _GATE_REGISTRY and _GATE_REGISTRY[name].matrix is not None:
            import numpy as np
            mat = np.asarray(_GATE_REGISTRY[name].matrix, dtype=complex)
            engine.add_gate(gate.DenseMatrix(qubits, mat))
            return

        if name in ("identity", "i"):
            engine.add_gate(gate.Identity(qubits[0]))
        elif name == "h":
            engine.add_gate(gate.H(qubits[0]))
        elif name == "x":
            engine.add_gate(gate.X(qubits[0]))
        elif name == "y":
            engine.add_gate(gate.Y(qubits[0]))
        elif name == "z":
            engine.add_gate(gate.Z(qubits[0]))
        elif name == "cx":
            engine.add_gate(gate.CNOT(qubits[0], qubits[1]))
        elif name == "cz":
            engine.add_gate(gate.CZ(qubits[0], qubits[1]))
        elif name == "swap":
            engine.add_gate(gate.SWAP(qubits[0], qubits[1]))
        elif name == "ccx":
            engine.add_gate(gate.TOFFOLI(qubits[0], qubits[1], qubits[2]))
        elif name == "rx":
            engine.add_gate(gate.RX(qubits[0], params[0]))
        elif name == "ry":
            engine.add_gate(gate.RY(qubits[0], params[0]))
        elif name == "rz":
            engine.add_gate(gate.RZ(qubits[0], params[0]))
        elif name == "p":
            import numpy as np
            mat = np.array([[1.0, 0.0], [0.0, np.exp(1j * params[0])]])
            engine.add_gate(gate.DenseMatrix(qubits[0], mat))
        elif name == "cp":
            import numpy as np
            p_mat = np.array([[1.0, 0.0], [0.0, np.exp(1j * params[0])]])
            engine.add_gate(gate.CNOT(qubits[0], qubits[1]))
            engine.add_gate(gate.DenseMatrix(qubits[1], p_mat))
            engine.add_gate(gate.CNOT(qubits[0], qubits[1]))
        elif name == "mcz":
            self._apply_mcz(engine, qubits)
        elif name == "measure":
            pass  # handled in _sample
        else:
            raise ValueError(tr(self._GATE_ERR, name=name))

    @staticmethod
    def _apply_mcz(engine: Any, qubits: list[int]) -> None:
        from qulacs import gate
        if len(qubits) == 2:
            engine.add_gate(gate.CZ(qubits[0], qubits[1]))
        elif len(qubits) == 3:
            engine.add_gate(gate.TOFFOLI(qubits[0], qubits[1], qubits[2]))
            engine.add_gate(gate.Z(qubits[2]))
            engine.add_gate(gate.TOFFOLI(qubits[0], qubits[1], qubits[2]))
        else:
            n = len(qubits)
            target = qubits[-1]
            engine.add_gate(gate.H(target))
            for i in range(n - 1):
                engine.add_gate(gate.CNOT(qubits[i], target))
            engine.add_gate(gate.H(target))

    def _sample(self, engine: Any, shots: int, n: int) -> dict[str, int]:
        from qulacs import QuantumState
        state = QuantumState(n)
        engine.update_quantum_state(state)
        raw = state.sampling(shots)
        counts: dict[str, int] = {}
        for val in raw:
            bs = format(val, f"0{n}b")[::-1]
            counts[bs] = counts.get(bs, 0) + 1
        return counts

    def _get_statevector(self, engine: Any, n: int) -> Any:
        from qulacs import QuantumState
        state = QuantumState(n)
        engine.update_quantum_state(state)
        return state.get_vector()

    def _get_density_matrix(self, engine: Any, n: int) -> Any:
        from qulacs import DensityMatrix
        dm = DensityMatrix(n)
        engine.update_quantum_state(dm)
        return dm.get_matrix()

    # ------------------------------------------------------------------ #
    #  GPU path — try QuantumStateGpu, fallback to CuPy
    # ------------------------------------------------------------------ #

    def _run_gpu(self, circuit, shots, nm):
        import importlib.util
        if importlib.util.find_spec("qulacs.gate") and hasattr(
            __import__("qulacs", fromlist=["QuantumStateGpu"]), "QuantumStateGpu"
        ):
            return self._run_gpu_native(circuit, shots, nm)
        return super()._run_gpu(circuit, shots, nm)

    def _run_gpu_native(self, circuit, shots, nm):
        """Native qulacs GPU execution (when QuantumStateGpu is available)."""
        from qulacs import QuantumCircuit as QC
        from qulacs import QuantumStateGpu

        n = circuit.num_qubits
        qc = QC(n)
        for op in circuit.ops:
            if op.name == "measure":
                continue
            self._apply_one(qc, op.name, list(op.qubits), op.params)

        state = QuantumStateGpu(n)
        qc.update_quantum_state(state)
        raw = state.sampling(shots)
        counts: dict[str, int] = {}
        for val in raw:
            bs = format(val, f"0{n}b")[::-1]
            counts[bs] = counts.get(bs, 0) + 1
        if nm.readout > 0:
            counts = self._apply_readout_noise(counts, n, nm.readout)
        from ..result import Result
        return Result.from_counts(counts, shots)

    # ------------------------------------------------------------------ #
    #  Dynamic path (v2) — stateful engine for mid-circuit measurement
    # ------------------------------------------------------------------ #

    def _run_dynamic(self, circuit, shots, nm, method, return_state=False):
        """Override: use stateful engine that collapses on measurement."""
        from qulacs import QuantumState

        counts: dict[str, int] = {}
        n = circuit.num_qubits

        for _ in range(shots):
            state = QuantumState(n)
            cregs: dict[str, int] = {}
            self._execute_shot_stateful(state, circuit.ops, cregs, n)
            # Sample the final state
            raw = state.sampling(1)
            bs = format(raw[0], f"0{n}b")[::-1]
            counts[bs] = counts.get(bs, 0) + 1

        if nm.readout > 0:
            counts = self._apply_readout_noise(counts, n, nm.readout)
        from ..result import Result
        return Result.from_counts(counts, shots)

    def _execute_shot_stateful(self, state, ops, cregs, n):
        """Execute ops on a QuantumState, collapsing on measurement."""

        for op in ops:
            name = op.name
            if name == "cmeasure":
                outcome = self._measure_and_collapse(state, op.qubit, n)
                v = cregs.get(op.creg, 0)
                cregs[op.creg] = (v & ~(1 << op.bit)) | (outcome << op.bit)
            elif name == "cif":
                if isinstance(op.control, int):
                    outcome = self._measure_and_collapse(state, op.control, n)
                    hit = outcome == 1
                elif isinstance(op.control, __import__("quonic.ir", fromlist=["CRegCondition"]).CRegCondition):
                    hit = cregs.get(op.control.creg, 0) == op.control.value
                else:
                    hit = cregs.get(op.control, 0) == 1
                branch = op.then_op if hit else op.else_op
                self._apply_gate_stateful(state, branch.name, list(branch.qubits), branch.params, n)
            elif name == "cwhile":
                iters = 0
                while cregs.get(op.creg, 0) != op.until:
                    self._execute_shot_stateful(state, op.body, cregs, n)
                    iters += 1
                    if iters > 100000:
                        raise RuntimeError("cwhile limit exceeded")
            elif name == "measure":
                pass
            else:
                self._apply_gate_stateful(state, name, list(op.qubits), op.params, n)

    def _apply_gate_stateful(self, state, name, qubits, params, n):
        """Apply a gate directly to a QuantumState."""
        import numpy as np
        from qulacs import QuantumCircuit
        from qulacs import gate as G

        # Build a tiny circuit with one gate, apply to state
        qc = QuantumCircuit(n)
        if name in ("identity", "i"):
            qc.add_gate(G.Identity(qubits[0]))
        elif name == "h":
            qc.add_gate(G.H(qubits[0]))
        elif name == "x":
            qc.add_gate(G.X(qubits[0]))
        elif name == "y":
            qc.add_gate(G.Y(qubits[0]))
        elif name == "z":
            qc.add_gate(G.Z(qubits[0]))
        elif name == "cx":
            qc.add_gate(G.CNOT(qubits[0], qubits[1]))
        elif name == "cz":
            qc.add_gate(G.CZ(qubits[0], qubits[1]))
        elif name == "swap":
            qc.add_gate(G.SWAP(qubits[0], qubits[1]))
        elif name == "ccx":
            qc.add_gate(G.TOFFOLI(qubits[0], qubits[1], qubits[2]))
        elif name == "rx":
            qc.add_gate(G.RX(qubits[0], params[0]))
        elif name == "ry":
            qc.add_gate(G.RY(qubits[0], params[0]))
        elif name == "rz":
            qc.add_gate(G.RZ(qubits[0], params[0]))
        elif name == "p":
            mat = np.array([[1.0, 0.0], [0.0, np.exp(1j * params[0])]])
            qc.add_gate(G.DenseMatrix(qubits[0], mat))
        elif name == "cp":
            p_mat = np.array([[1.0, 0.0], [0.0, np.exp(1j * params[0])]])
            qc.add_gate(G.CNOT(qubits[0], qubits[1]))
            qc.add_gate(G.DenseMatrix(qubits[1], p_mat))
            qc.add_gate(G.CNOT(qubits[0], qubits[1]))
        elif name == "mcz":
            if len(qubits) == 2:
                qc.add_gate(G.CZ(qubits[0], qubits[1]))
            else:
                target = qubits[-1]
                qc.add_gate(G.H(target))
                for i in range(len(qubits) - 1):
                    qc.add_gate(G.CNOT(qubits[i], target))
                qc.add_gate(G.H(target))
        elif name in ("measure", "i"):
            return
        else:
            return  # skip unknown gates in dynamic mode
        qc.update_quantum_state(state)

    def _measure_and_collapse(self, state, qubit, n):
        """Measure qubit, collapse state, return outcome."""
        import numpy as np

        sv = state.get_vector()
        idx = np.arange(2**n)
        bit = (idx >> qubit) & 1
        p0 = float(np.sum(np.abs(sv[bit == 0]) ** 2))
        outcome = 0 if np.random.random() < p0 else 1

        # Collapse: zero out the non-measured branches, renormalize
        sv_new = sv.copy()
        sv_new[bit != outcome] = 0.0
        norm = np.linalg.norm(sv_new)
        if norm > 0:
            sv_new /= norm
        # Write back to state using load()
        state.load(sv_new)
        return outcome

    # ------------------------------------------------------------------ #
    #  Density-matrix path (v2)
    # ------------------------------------------------------------------ #

    def _create_dm(self, n: int) -> Any:
        try:
            import qulacs
        except ImportError as e:
            raise ImportError(tr(self._MISSING_ERR)) from e
        return (qulacs.QuantumCircuit(n), qulacs.DensityMatrix(n))

    def _apply_one_dm(self, engine, name, qubits, params):
        circuit, _dm = engine
        self._apply_one(circuit, name, qubits, params)

    def _sample_dm(self, engine, shots, n):
        circuit, dm = engine
        dm.set_zero_state()
        circuit.update_quantum_state(dm)
        raw = dm.sampling(shots)
        counts: dict[str, int] = {}
        for val in raw:
            bs = format(val, f"0{n}b")[::-1]
            counts[bs] = counts.get(bs, 0) + 1
        return counts

    def _apply_noise_after_gate(self, engine, qubits, nm):
        from qulacs import gate
        circuit, _dm = engine
        p = nm.single if len(qubits) == 1 else nm.double
        if p > 0:
            for q in qubits:
                circuit.add_gate(gate.DepolarizingNoise(q, p))

    def _measure_qubit(self, engine, qubit):
        """Mid-circuit measurement with state collapse."""
        import numpy as np
        from qulacs import QuantumState

        if isinstance(engine, tuple):
            circuit, dm = engine
            n = dm.get_qubit_count()
            dm.set_zero_state()
            circuit.update_quantum_state(dm)
            data = dm.get_matrix()
            diag = np.real(np.diag(data))
        else:
            n = engine.get_qubit_count()
            state = QuantumState(n)
            engine.update_quantum_state(state)
            sv = state.get_vector()
            diag = np.abs(sv) ** 2

        idx = np.arange(2**n)
        bit = (idx >> qubit) & 1
        p0 = float(np.sum(diag[bit == 0]))
        return 0 if np.random.random() < p0 else 1
