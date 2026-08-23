"""TensorCircuit backend adapter."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, ClassVar

from .._i18n import tr
from ..noise import NoiseModel
from .engine import EngineBackend

_tc_numpy_patched = False


def _ensure_tc_numpy_compat() -> None:
    """Patch numpy for TensorCircuit 0.12 + numpy 2.x compatibility.

    Applied once at first use.  The patches are **idempotent** — they translate
    the deprecated ``newshape`` kwarg to ``shape`` and restore ``ComplexWarning``.
    Neither change alters numpy's public API semantics, so other backends and
    user code are unaffected.
    """
    global _tc_numpy_patched
    if _tc_numpy_patched:
        return

    import numpy as np

    if np.__version__ >= "2":
        _orig_reshape = np.reshape

        def _compat_reshape(a, *args, **kwargs):
            if "newshape" in kwargs and "shape" not in kwargs:
                kwargs["shape"] = kwargs.pop("newshape")
            return _orig_reshape(a, *args, **kwargs)

        np.reshape = _compat_reshape

        if not hasattr(np, "ComplexWarning"):
            try:
                from numpy.exceptions import ComplexWarning
                np.ComplexWarning = ComplexWarning
            except ImportError:
                pass

    _tc_numpy_patched = True


@contextmanager
def _tc_compat():
    """Context manager variant: patches on entry, restores on exit.

    Use this in backend methods for process-level isolation.  For test helpers
    that need the patch for the entire session, use ``_ensure_tc_numpy_compat``
    instead (patches once, never restores).
    """
    import numpy as np

    patched = False
    patched_warn = False
    orig_reshape = np.reshape

    if np.__version__ >= "2":
        def _compat_reshape(a, *args, **kwargs):
            if "newshape" in kwargs and "shape" not in kwargs:
                kwargs["shape"] = kwargs.pop("newshape")
            return orig_reshape(a, *args, **kwargs)

        np.reshape = _compat_reshape
        patched = True

        if not hasattr(np, "ComplexWarning"):
            try:
                from numpy.exceptions import ComplexWarning
                np.ComplexWarning = ComplexWarning
                patched_warn = True
            except ImportError:
                pass

    try:
        yield
    finally:
        if patched:
            np.reshape = orig_reshape
        if patched_warn:
            try:
                del np.ComplexWarning
            except AttributeError:
                pass


class TensorCircuitBackend(EngineBackend):
    name = "tensorcircuit"
    _MISSING_ERR = "err.tensorcircuit_missing"
    _GATE_ERR = "err.tensorcircuit_gate"
    methods = frozenset({"statevector", "density_matrix"})
    _CAPABILITIES: ClassVar[dict[str, bool]] = {"noise": True, "ctrl": True, "mid_measure": True, "gpu": True}

    # ------------------------------------------------------------------ #
    #  Statevector path (v1, unchanged)
    # ------------------------------------------------------------------ #

    def _create(self, n: int) -> Any:
        with _tc_compat():
            try:
                import tensorcircuit as tc
            except ImportError as e:
                raise ImportError(tr(self._MISSING_ERR)) from e
        self._n = n
        return tc.Circuit(n)

    def _apply_one(
        self, engine: Any, name: str, qubits: list[int], params: tuple[float, ...]
    ) -> None:
        # Flip qubit indices: QuoNic 0=LSB → TC 0=MSB.
        # Dynamic path uses _apply_one_tc which does NOT flip (uses numpy SV directly).
        n = self._n
        q = [n - 1 - qi for qi in qubits]
        if name in ("identity", "i"):
            pass
        elif name == "h":
            engine.h(q[0])
        elif name == "x":
            engine.x(q[0])
        elif name == "y":
            engine.y(q[0])
        elif name == "z":
            engine.z(q[0])
        elif name == "cx":
            engine.cnot(q[0], q[1])
        elif name == "cz":
            engine.cz(q[0], q[1])
        elif name == "swap":
            engine.swap(q[0], q[1])
        elif name == "ccx":
            engine.toffoli(q[0], q[1], q[2])
        elif name == "rx":
            engine.rx(q[0], theta=params[0])
        elif name == "ry":
            engine.ry(q[0], theta=params[0])
        elif name == "rz":
            engine.rz(q[0], theta=params[0])
        elif name == "p":
            engine.phase(q[0], theta=params[0])
        elif name == "cp":
            engine.cphase(q[0], q[1], theta=params[0])
        elif name == "mcz":
            self._apply_mcz(engine, q)
        elif name == "measure":
            pass
        else:
            raise ValueError(tr(self._GATE_ERR, name=name))

    @staticmethod
    def _apply_mcz(engine: Any, qubits: list[int]) -> None:
        if len(qubits) == 2:
            engine.cz(qubits[0], qubits[1])
        else:
            target = qubits[-1]
            engine.h(target)
            for c in qubits[:-1]:
                engine.cnot(c, target)
            engine.h(target)

    def _sample(self, engine: Any, shots: int, n: int) -> dict[str, int]:
        raw = engine.sample(shots)
        # TC: qubit 0 = leftmost in bitstring (MSB).
        # QuoNic: qubit 0 = rightmost (LSB).  Reverse to match.
        counts: dict[str, int] = {}
        for val in raw:
            if isinstance(val, tuple):
                bits = val[0]
                bs = "".join(str(int(bits[i])) for i in range(n))[::-1]
            elif isinstance(val, str):
                bs = val[::-1]
            else:
                bs = "".join(str(int(val[i])) for i in range(n))[::-1]
            counts[bs] = counts.get(bs, 0) + 1
        return counts

    # ------------------------------------------------------------------ #
    #  Density-matrix path (v2)
    # ------------------------------------------------------------------ #

    def _create_dm(self, n: int) -> Any:
        """DMCircuit is a drop-in replacement for Circuit — same API."""
        with _tc_compat():
            try:
                import tensorcircuit as tc
            except ImportError as e:
                raise ImportError(tr(self._MISSING_ERR)) from e
        self._n = n
        return tc.DMCircuit(n)

    def _run_gpu(self, circuit, shots, nm):
        """Try JAX backend for GPU, fallback to CuPy."""
        with _tc_compat():
            try:
                import tensorcircuit as tc
                tc.set_backend("jax")
                # JAX backend active — use TC's native GPU path
                engine = tc.Circuit(circuit.num_qubits)
                for op in circuit.ops:
                    if op.name == "measure":
                        continue
                    self._apply_one(engine, op.name, list(op.qubits), op.params)
                raw = engine.sample(shots)
                counts: dict[str, int] = {}
                for val in raw:
                    if isinstance(val, tuple):
                        bits = val[0]
                        bs = "".join(str(int(bits[i])) for i in range(circuit.num_qubits))[::-1]
                    elif isinstance(val, str):
                        bs = val[::-1]
                    else:
                        bs = "".join(str(int(val[i])) for i in range(circuit.num_qubits))[::-1]
                    counts[bs] = counts.get(bs, 0) + 1
                if nm.readout > 0:
                    counts = self._apply_readout_noise(counts, circuit.num_qubits, nm.readout)
                from ..result import Result
                return Result.from_counts(counts, shots)
            except (ImportError, RuntimeError, ValueError):
                return super()._run_gpu(circuit, shots, nm)

    def _apply_noise_after_gate(
        self, engine: Any, qubits: list[int], nm: NoiseModel
    ) -> None:
        import tensorcircuit as tc

        p = nm.single if len(qubits) == 1 else nm.double
        if p > 0:
            # depolarizingchannel(px, py, pz) — symmetric: px=py=pz=p/3
            channel = tc.channels.depolarizingchannel(p / 3, p / 3, p / 3)
            for q in qubits:
                engine.apply_general_kraus(channel, [(q,)])

    def _measure_qubit(self, engine: Any, qubit: int) -> int:
        """Mid-circuit measurement via manual probability extraction from DM.

        TC uses qubit 0 = LSB (same as QuoNic). Standard bit extraction.
        """
        import numpy as np

        dm = engine.state()
        dm_np = dm.numpy() if hasattr(dm, "numpy") else np.asarray(dm)
        if dm_np.ndim == 1:
            dim = int(np.sqrt(len(dm_np)))
            dm_np = dm_np.reshape(dim, dim)
        diag = np.real(np.diag(dm_np))
        n = int(np.log2(len(diag)))
        idx = np.arange(2**n)
        bit = (idx >> qubit) & 1
        p0 = float(np.sum(diag[bit == 0]))
        return 0 if np.random.random() < p0 else 1

    # ------------------------------------------------------------------ #
    #  Dynamic path (v2) — segment-by-segment with state collapse
    # ------------------------------------------------------------------ #

    def _apply_one_tc(self, engine, name, qubits, params, n):
        """Apply gate to TC engine for dynamic path (numpy SV, 0=LSB)."""
        self._apply_one(engine, name, qubits, params)

    def _run_dynamic(self, circuit, shots, nm, method, return_state=False):
        """Override: segment-by-segment DMCircuit execution with mid-circuit collapse."""
        with _tc_compat():
            import numpy as np

            counts: dict[str, int] = {}
            n = circuit.num_qubits
            self._n = n

            for _ in range(shots):
                sv = self._execute_dynamic_segments(circuit.ops, n)
                # Sample from final state vector
                probs = np.abs(sv) ** 2
                probs = probs / probs.sum()
                idx = np.random.choice(2**n, p=probs)
                bs = format(idx, f"0{n}b")[::-1]
                counts[bs] = counts.get(bs, 0) + 1

            if nm.readout > 0:
                counts = self._apply_readout_noise(counts, n, nm.readout)
        from ..result import Result
        return Result.from_counts(counts, shots)

    def _execute_dynamic_segments(self, ops, n, cregs=None, current_sv=None):
        """Execute ops on a state vector (numpy), collapsing on measurement.

        Uses numpy directly — no TC DMCircuit injection (TC's internal tensor
        network doesn't support state injection).
        """
        import numpy as np

        if cregs is None:
            cregs: dict[str, int] = {}
        sv = current_sv
        if sv is None:
            sv = np.zeros(2**n, dtype=complex)
            sv[0] = 1.0

        # Split into segments: each segment ends at a measurement/cif/cmeasure/cwhile
        segments = []
        current_segment = []
        for op in ops:
            if op.name in ("cmeasure", "cif", "cwhile"):
                if current_segment:
                    segments.append(("gates", current_segment))
                    current_segment = []
                segments.append(("control", op))
            elif op.name == "measure":
                if current_segment:
                    segments.append(("gates", current_segment))
                    current_segment = []
                segments.append(("measure", op))
            else:
                current_segment.append(op)
        if current_segment:
            segments.append(("gates", current_segment))

        for seg_type, seg_data in segments:
            if seg_type == "gates":
                for op in seg_data:
                    sv = self._apply_gate_sv(sv, op.name, list(op.qubits), op.params, n)

            elif seg_type == "measure":
                outcome = self._measure_sv(sv, seg_data.qubit, n)
                sv = self._collapse_sv(sv, seg_data.qubit, n, outcome)

            elif seg_type == "control":
                op = seg_data
                if op.name == "cmeasure":
                    outcome = self._measure_sv(sv, op.qubit, n)
                    sv = self._collapse_sv(sv, op.qubit, n, outcome)
                    v = cregs.get(op.creg, 0)
                    cregs[op.creg] = (v & ~(1 << op.bit)) | (outcome << op.bit)

                elif op.name == "cif":
                    if isinstance(op.control, int):
                        outcome = self._measure_sv(sv, op.control, n)
                        sv = self._collapse_sv(sv, op.control, n, outcome)
                        hit = outcome == 1
                    elif isinstance(op.control, __import__("quonic.ir", fromlist=["CRegCondition"]).CRegCondition):
                        hit = cregs.get(op.control.creg, 0) == op.control.value
                    else:
                        hit = cregs.get(op.control, 0) == 1
                    branch = op.then_op if hit else op.else_op
                    sv = self._apply_gate_sv(sv, branch.name, list(branch.qubits), branch.params, n)

                elif op.name == "cwhile":
                    iters = 0
                    while cregs.get(op.creg, 0) != op.until:
                        sv = self._execute_dynamic_segments(op.body, n, cregs, sv)
                        iters += 1
                        if iters > 100000:
                            raise RuntimeError("cwhile limit exceeded")

        return sv

    def _measure_dm(self, dm_np, qubit, n):
        """Measure probability from density matrix.

        TC uses qubit 0 = LSB (same as QuoNic). Standard bit extraction.
        """
        import numpy as np
        diag = np.real(np.diag(dm_np))
        idx = np.arange(2**n)
        bit = (idx >> qubit) & 1
        p0 = float(np.sum(diag[bit == 0]))
        return 0 if np.random.random() < p0 else 1

    def _collapse_dm(self, dm_np, qubit, n, outcome=None):
        """Collapse density matrix after measurement."""
        import numpy as np
        if outcome is None:
            outcome = self._measure_dm(dm_np, qubit, n)
        idx = np.arange(2**n)
        bit = (idx >> qubit) & 1
        dm_new = dm_np.copy()
        dm_new[bit != outcome, :] = 0.0
        dm_new[:, bit != outcome] = 0.0
        tr_val = np.real(np.trace(dm_new))
        if tr_val > 0:
            dm_new /= tr_val
        return dm_new

    # ------------------------------------------------------------------ #
    #  Statevector helpers for dynamic path (numpy-based, no TC dependency)
    # ------------------------------------------------------------------ #

    def _apply_gate_sv(self, sv, name, qubits, params, n):
        """Apply a gate to a numpy state vector."""

        if name in ("measure", "identity", "i"):
            return sv

        # Build single-qubit unitary
        u = self._gate_matrix(name, params)
        if u is None:
            return sv

        if len(qubits) == 1:
            return self._apply_single_sv(sv, u, qubits[0], n)
        elif len(qubits) == 2 and name in ("cx", "cz", "cp"):
            return self._apply_two_qubit_sv(sv, name, qubits[0], qubits[1], params, n)
        elif len(qubits) == 3 and name == "ccx":
            return self._apply_ccx_sv(sv, qubits[0], qubits[1], qubits[2], n)
        elif name == "swap":
            return self._apply_swap_sv(sv, qubits[0], qubits[1], n)
        elif name == "mcz":
            return self._apply_mcz_sv(sv, qubits, n)
        return sv

    def _gate_matrix(self, name, params):
        """Return 2x2 unitary matrix for a single-qubit gate."""
        import numpy as np
        if name == "h":
            return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        if name == "x":
            return np.array([[0, 1], [1, 0]], dtype=complex)
        if name == "y":
            return np.array([[0, -1j], [1j, 0]], dtype=complex)
        if name == "z":
            return np.array([[1, 0], [0, -1]], dtype=complex)
        if name == "rx":
            t = params[0]
            return np.array([[np.cos(t/2), -1j*np.sin(t/2)], [-1j*np.sin(t/2), np.cos(t/2)]], dtype=complex)
        if name == "ry":
            t = params[0]
            return np.array([[np.cos(t/2), -np.sin(t/2)], [np.sin(t/2), np.cos(t/2)]], dtype=complex)
        if name == "rz":
            t = params[0]
            return np.array([[np.exp(-1j*t/2), 0], [0, np.exp(1j*t/2)]], dtype=complex)
        if name == "p":
            t = params[0]
            return np.array([[1, 0], [0, np.exp(1j*t)]], dtype=complex)
        if name in ("cx", "cz", "cp", "ccx", "swap", "mcz"):
            return None  # multi-qubit, handled separately
        return None

    def _apply_single_sv(self, sv, u, qubit, n):
        """Apply a single-qubit gate to a state vector."""
        new_sv = sv.copy()
        for i in range(2**n):
            if (i >> qubit) & 1 == 0:
                j = i | (1 << qubit)
                a, b = sv[i], sv[j]
                new_sv[i] = u[0, 0] * a + u[0, 1] * b
                new_sv[j] = u[1, 0] * a + u[1, 1] * b
        return new_sv

    def _apply_two_qubit_sv(self, sv, name, q0, q1, params, n):
        """Apply a two-qubit gate (CX, CZ, CP) to a state vector."""
        import numpy as np
        new_sv = sv.copy()
        for i in range(2**n):
            b0 = (i >> q0) & 1
            b1 = (i >> q1) & 1
            if name == "cx" and b0 == 1:
                j = i ^ (1 << q1)
                new_sv[i] = sv[j]
                new_sv[j] = sv[i]
            elif name == "cz" and b0 == 1 and b1 == 1:
                new_sv[i] = -sv[i]
            elif name == "cp" and b0 == 1 and b1 == 1:
                new_sv[i] = np.exp(1j * params[0]) * sv[i]
        return new_sv

    def _apply_ccx_sv(self, sv, c0, c1, target, n):
        """Apply Toffoli (CCX) gate."""
        new_sv = sv.copy()
        for i in range(2**n):
            if (i >> c0) & 1 == 1 and (i >> c1) & 1 == 1:
                j = i ^ (1 << target)
                new_sv[i] = sv[j]
                new_sv[j] = sv[i]
        return new_sv

    def _apply_swap_sv(self, sv, q0, q1, n):
        """Apply SWAP gate."""
        new_sv = sv.copy()
        for i in range(2**n):
            b0 = (i >> q0) & 1
            b1 = (i >> q1) & 1
            if b0 != b1:
                j = i ^ (1 << q0) ^ (1 << q1)
                new_sv[i] = sv[j]
                new_sv[j] = sv[i]
        return new_sv

    def _apply_mcz_sv(self, sv, qubits, n):
        """Apply multi-controlled Z gate."""
        new_sv = sv.copy()
        for i in range(2**n):
            if all((i >> q) & 1 == 1 for q in qubits):
                new_sv[i] = -sv[i]
        return new_sv

    def _measure_sv(self, sv, qubit, n):
        """Measure a qubit from a state vector, return 0 or 1."""
        import numpy as np
        idx = np.arange(2**n)
        bit = (idx >> qubit) & 1
        p0 = float(np.sum(np.abs(sv[bit == 0]) ** 2))
        return 0 if np.random.random() < p0 else 1

    def _collapse_sv(self, sv, qubit, n, outcome):
        """Collapse state vector after measurement."""
        import numpy as np
        idx = np.arange(2**n)
        bit = (idx >> qubit) & 1
        new_sv = sv.copy()
        new_sv[bit != outcome] = 0.0
        norm = np.linalg.norm(new_sv)
        if norm > 0:
            new_sv /= norm
        return new_sv
