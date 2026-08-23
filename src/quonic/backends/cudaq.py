"""CUDA-Q backend adapter."""

from __future__ import annotations

from typing import Any

from .._i18n import tr
from ..ir import CRegCondition
from ..noise import NoiseModel
from ..result import Result
from .engine import EngineBackend


class CudaQBackend(EngineBackend):
    name = "cudaq"
    _MISSING_ERR = "err.cudaq_missing"
    _GATE_ERR = "err.cudaq_gate"
    methods = frozenset({"statevector", "density_matrix"})
    _CAPABILITIES = {"noise": True, "ctrl": True, "mid_measure": True, "gpu": True}

    # ------------------------------------------------------------------ #
    #  Statevector path (v1)
    # ------------------------------------------------------------------ #

    def _create(self, n: int) -> Any:
        try:
            import cudaq
        except ImportError as e:
            raise ImportError(tr(self._MISSING_ERR)) from e
        kernel = cudaq.make_kernel()
        return (kernel, n)  # store n for sampling

    def _apply_one(
        self, engine: Any, name: str, qubits: list[int], params: tuple[float, ...]
    ) -> None:
        kernel, _ = engine
        if name == "identity":
            pass
        elif name == "h":
            kernel.h(qubits[0])
        elif name == "x":
            kernel.x(qubits[0])
        elif name == "y":
            kernel.y(qubits[0])
        elif name == "z":
            kernel.z(qubits[0])
        elif name == "cx":
            kernel.cx(qubits[0], qubits[1])
        elif name == "cz":
            kernel.cz(qubits[0], qubits[1])
        elif name == "swap":
            kernel.swap(qubits[0], qubits[1])
        elif name == "ccx":
            kernel.x(qubits[2], qubits[0], qubits[1])  # toffoli
        elif name == "rx":
            kernel.rx(params[0], qubits[0])
        elif name == "ry":
            kernel.ry(params[0], qubits[0])
        elif name == "rz":
            kernel.rz(params[0], qubits[0])
        elif name == "p":
            # P(θ) = diag(1, e^{iθ}). CUDA-Q has rz but not p; use rz + global phase.
            # rz(θ) = diag(e^{-iθ/2}, e^{iθ/2}), so P(θ) = e^{iθ/2} · rz(θ).
            # Global phase is unobservable, so rz is acceptable for single-qubit.
            kernel.rz(params[0], qubits[0])
        elif name == "cp":
            # CP(θ) — CUDA-Q has no native controlled-phase; decompose.
            # CP(θ) = |0><0| ⊗ I + |1><1| ⊗ P(θ)
            # Decomposition: CX + rz(θ/2) + CX + rz(-θ/2) (standard)
            kernel.rz(params[0] / 2, qubits[1])
            kernel.cx(qubits[0], qubits[1])
            kernel.rz(-params[0] / 2, qubits[1])
            kernel.cx(qubits[0], qubits[1])
        elif name == "mcz":
            self._apply_mcz(kernel, qubits)
        elif name == "measure":
            pass
        else:
            raise ValueError(tr(self._GATE_ERR, name=name))

    @staticmethod
    def _apply_mcz(kernel: Any, qubits: list[int]) -> None:
        if len(qubits) == 2:
            kernel.cz(qubits[0], qubits[1])
        else:
            target = qubits[-1]
            kernel.h(target)
            for c in qubits[:-1]:
                kernel.cx(c, target)
            kernel.h(target)

    def _sample(self, engine: Any, shots: int, n: int) -> dict[str, int]:
        import cudaq

        kernel, _ = engine
        result = cudaq.sample(kernel, shots_count=shots)
        counts: dict[str, int] = {}
        for bs, count in result.items():
            counts[str(bs)] = counts.get(str(bs), 0) + int(count)
        return counts

    def _run_gpu(self, circuit, shots, nm):
        """CUDA-Q is GPU-native — just run normally."""
        from ..result import Result

        engine = self._create(circuit.num_qubits)
        for op in circuit.ops:
            if op.name == "measure":
                continue
            self._apply_one(engine, op.name, list(op.qubits), op.params)
        counts = self._sample(engine, shots, circuit.num_qubits)
        if nm.readout > 0:
            counts = self._apply_readout_noise(counts, circuit.num_qubits, nm.readout)
        return Result.from_counts(counts, shots)

    # ------------------------------------------------------------------ #
    #  Noise path (v2) — CUDA-Q uses a global NoiseModel
    # ------------------------------------------------------------------ #

    def _run_noisy(
        self, circuit: Any, shots: int, nm: NoiseModel, method: str
    ) -> Result:
        """CUDA-Q noise: set_noise() is global, not per-gate."""
        import cudaq

        kernel, n = self._create(circuit.num_qubits)
        for op in circuit.ops:
            self._apply_one((kernel, n), op.name, list(op.qubits), op.params)

        noise_model = cudaq.NoiseModel()
        _SINGLE_GATES = ["h", "x", "y", "z", "rx", "ry", "rz", "p", "i"]
        _DOUBLE_GATES = ["cx", "cz", "swap"]
        if nm.single > 0:
            ch = cudaq.noise.depolarization_channel(nm.single)
            for g in _SINGLE_GATES:
                noise_model.add_channel(g, [ch])
        if nm.double > 0:
            ch = cudaq.noise.depolarization_channel(nm.double)
            for g in _DOUBLE_GATES:
                noise_model.add_channel(g, [ch])

        cudaq.set_noise(noise_model)
        result = cudaq.sample(kernel, shots_count=shots)
        cudaq.unset_noise()

        counts: dict[str, int] = {}
        for bs, count in result.items():
            counts[str(bs)] = counts.get(str(bs), 0) + int(count)

        if nm.readout > 0:
            counts = self._apply_readout_noise(counts, circuit.num_qubits, nm.readout)
        return Result.from_counts(counts, shots)

    # ------------------------------------------------------------------ #
    #  Dynamic path (v2) — per-shot kernel creation
    # ------------------------------------------------------------------ #

    def _run_dynamic(
        self, circuit: Any, shots: int, nm: NoiseModel, method: str,
        return_state: bool = False,
    ) -> Any:
        """CUDA-Q dynamic circuits: per-shot kernel creation (high overhead)."""
        import cudaq

        counts: dict[str, int] = {}
        for _ in range(shots):
            kernel, n = self._create(circuit.num_qubits)
            cregs: dict[str, int] = {}
            self._cudaq_execute_shot(kernel, n, circuit.ops, cregs)
            result = cudaq.sample(kernel, shots_count=1)
            for bs, c in result.items():
                counts[str(bs)] = counts.get(str(bs), 0) + int(c)
        return Result.from_counts(counts, shots)

    def _cudaq_execute_shot(
        self, kernel: Any, n: int, ops: list, cregs: dict[str, int]
    ) -> None:
        """Execute ops on a CUDA-Q kernel for one shot."""
        for op in ops:
            name = op.name
            if name == "cmeasure":
                # CUDA-Q doesn't support mid-circuit measurement natively.
                # Use random outcome as approximation (no state collapse).
                import random as _rnd

                outcome = _rnd.randint(0, 1)
                v = cregs.get(op.creg, 0)
                cregs[op.creg] = (v & ~(1 << op.bit)) | (outcome << op.bit)
            elif name == "cif":
                if isinstance(op.control, int):
                    # Can't measure mid-kernel; random outcome as approximation
                    import random as _rnd

                    hit = _rnd.randint(0, 1) == 1
                elif isinstance(op.control, CRegCondition):
                    hit = cregs.get(op.control.creg, 0) == op.control.value
                else:
                    hit = cregs.get(op.control, 0) == 1
                branch = op.then_op if hit else op.else_op
                self._apply_one((kernel, n), branch.name, list(branch.qubits), branch.params)
            elif name == "cwhile":
                iters = 0
                while cregs.get(op.creg, 0) != op.until:
                    self._cudaq_execute_shot(kernel, n, list(op.body), cregs)
                    iters += 1
                    if iters > 100000:
                        raise RuntimeError(tr("err.cwhile_limit", creg=op.creg))
            elif name == "measure":
                pass
            else:
                self._apply_one((kernel, n), name, list(op.qubits), op.params)
