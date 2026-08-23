"""QuoNic's in-house backend: depends on no external quantum library, using four naive engines directly.

This is the "unbound + compose" fallback: whatever backend the user switches to, any
circuit suited to stabilizer / MPS or other non-statevector methods can fall back here
(only numpy is required).

Engines are imported lazily inside run(), so `import quonic` does not pull in numpy.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .._i18n import tr
from ..ir import Circuit, CRegCondition
from ..noise import NoiseModel, resolve_noise
from ..result import Result
from .base import Backend

_METHODS: tuple[str, ...] = (
    "statevector",
    "stabilizer",
    "matrix_product_state",
    "density_matrix",
)


class NativeBackend(Backend):
    name = "native"
    methods = frozenset(_METHODS)
    _CAPABILITIES = {"noise": True, "ctrl": True, "mid_measure": True, "gpu": False}

    def run(
        self,
        circuit: Circuit,
        shots: int = 1024,
        noise: NoiseModel | float | None = None,
        method: str = "statevector",
        return_state: bool = False,
    ) -> Any:
        if method == "gpu":
            raise NotImplementedError(tr("err.no_gpu", name=self.name))
        from ..simulators import (
            DensityMatrixEngine,
            MPSEngine,
            StabilizerEngine,
            StatevectorEngine,
        )

        nm = resolve_noise(noise)
        # Classical control flow (cif / creg / cwhile) needs per-shot simulation:
        # mid-circuit measurement irreversibly collapses the statevector
        if any(op.name in ("cif", "cmeasure", "cwhile") for op in circuit.ops):
            return self._run_dynamic(circuit, shots, nm, method)

        # Noise simulation requires the density-matrix method; the other engines do not support general noise models
        if nm.enabled:
            engine = DensityMatrixEngine(circuit.num_qubits, noise=nm)
        else:
            engines = {
                "statevector": StatevectorEngine,
                "stabilizer": StabilizerEngine,
                "matrix_product_state": MPSEngine,
                "density_matrix": DensityMatrixEngine,
            }
            if method not in engines:
                raise ValueError(
                    tr("err.native_method", method=method, engines=", ".join(sorted(engines)))
                )
            engine = engines[method](circuit.num_qubits)

        for op in circuit.ops:
            engine.apply(op.name, list(op.qubits), op.params)

        if return_state:
            from ..statevector import MixedState, StateVector
            if hasattr(engine, "rho"):
                return MixedState(engine.rho.copy())
            return StateVector(engine.state.copy())

        return Result.from_counts(engine.sample(shots), shots)

    @classmethod
    def _run_dynamic(
        cls, circuit: Circuit, shots: int, nm: NoiseModel, method: str
    ) -> Result:
        from ..simulators import DensityMatrixEngine, StatevectorEngine

        if nm.enabled:
            method = "density_matrix"
        if method == "density_matrix":
            def new_engine() -> Any:
                return DensityMatrixEngine(circuit.num_qubits, noise=nm)
        elif method == "statevector":
            def new_engine() -> Any:
                return StatevectorEngine(circuit.num_qubits)
        else:
            raise NotImplementedError(tr("err.native_ctrl", method=method))

        counts: dict[str, int] = {}
        for _ in range(shots):
            engine = new_engine()
            cregs: dict[str, int] = {}
            cls._execute(engine, circuit.ops, cregs)
            for bs, c in engine.sample(1).items():
                counts[bs] = counts.get(bs, 0) + c
        return Result.from_counts(counts, shots)

    @staticmethod
    def _execute(engine: Any, ops: Iterable[Any], cregs: dict[str, int]) -> None:
        """Execute a block of ops shot by shot, maintaining named classical registers
        cregs (name -> integer register value)."""
        for op in ops:
            name = op.name
            if name == "cmeasure":
                v = cregs.get(op.creg, 0)
                outcome = engine.measure_qubit(op.qubit)
                cregs[op.creg] = (v & ~(1 << op.bit)) | (outcome << op.bit)
            elif name == "cif":
                if isinstance(op.control, int):
                    outcome = engine.measure_qubit(op.control)
                    hit = outcome == 1
                elif isinstance(op.control, CRegCondition):
                    hit = cregs.get(op.control.creg, 0) == op.control.value
                else:
                    hit = cregs.get(op.control, 0) == 1
                branch = op.then_op if hit else op.else_op
                engine.apply(branch.name, list(branch.qubits), branch.params)
            elif name == "cwhile":
                iters = 0
                while cregs.get(op.creg, 0) != op.until:
                    NativeBackend._execute(engine, op.body, cregs)
                    iters += 1
                    if iters > 100000:
                        raise RuntimeError(tr("err.cwhile_limit", creg=op.creg))
            else:
                engine.apply(name, list(op.qubits), op.params)
